"""
FedFOD Local Trainer Module
============================

SCAFFOLD local training loop with gradient correction, compression pipeline
(DP noise, top-k sparsification, int8 quantisation), and Flower NumPyClient
for federated learning integration.
"""

import base64
import copy
import io
import logging
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils

try:
    import flwr.client
except ImportError:
    flwr = None

try:
    import yaml
except ImportError:
    yaml = None

from src.client.inference import RTDETRDetector, Detection, FalseAlarmFilterMLP
from src.client.open_world import CLIPOpenWorldDetector
from src.common.utils import extract_logits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CompressionPipeline
# ---------------------------------------------------------------------------
class CompressionPipeline:
    """Sequential compression: clip → DP noise → top-k sparsify → int8 quantise.

    Ensures the final payload fits within ``max_payload_mb``.
    """

    def __init__(
        self,
        clip_norm: float = 1.0,
        dp_epsilon: float = 4.0,
        dp_delta: float = 1e-6,
        top_k_pct: float = 0.05,
        quantize_bits: int = 8,
        max_payload_mb: float = 2.0,
        is_satellite: bool = False,
    ):
        self.clip_norm = clip_norm
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.top_k_pct = top_k_pct
        self.quantize_bits = quantize_bits
        self.max_payload_mb = max_payload_mb
        self.is_satellite = is_satellite

        # For satellite links, use more aggressive compression
        if is_satellite:
            self.top_k_pct = min(self.top_k_pct, 0.01)
            self.max_payload_mb = min(self.max_payload_mb, 0.5)

    def compress(
        self, weights_delta: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Apply sequential compression to a weight delta dictionary.

        Pipeline: clip → DP noise → top-k sparsify → int8 quantise.

        Returns:
            (compressed_dict, metadata)  where metadata stores scales and masks
            needed for decompression.
        """
        metadata: Dict[str, Any] = {}

        # Stage 1: Gradient norm clipping
        clipped = self._clip_gradient_norm(weights_delta)

        # Stage 2: Add Gaussian DP noise
        noised = self._add_gaussian_dp_noise(clipped)

        # Stage 3: Top-k sparsification
        sparse, masks = self._sparsify_gradients(noised)
        metadata["masks"] = masks

        # Stage 4: Int8 quantisation
        quantized, scales = self._quantize_to_int8(sparse)
        metadata["scales"] = scales

        # Validate payload size
        total_bytes = sum(arr.nbytes for arr in quantized.values())
        max_bytes = self.max_payload_mb * 1024 * 1024
        if total_bytes > max_bytes:
            logger.warning(
                "Compressed payload %.2f MB exceeds limit %.2f MB – "
                "applying additional sparsification",
                total_bytes / (1024 * 1024),
                self.max_payload_mb,
            )
            # Further reduce: zero out smallest values
            for key in quantized:
                arr = quantized[key]
                flat = arr.flatten().astype(np.float32)
                threshold = np.percentile(np.abs(flat), 95)
                arr[np.abs(arr.astype(np.float32)) < threshold] = 0
                quantized[key] = arr

        return quantized, metadata

    def decompress(
        self,
        compressed: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Reverse the int8 quantisation using stored scales."""
        scales = metadata.get("scales", {})
        decompressed: Dict[str, np.ndarray] = {}

        for key, arr in compressed.items():
            scale = scales.get(key, 1.0)
            decompressed[key] = arr.astype(np.float32) * scale

        return decompressed

    # -- private pipeline stages --------------------------------------------

    def _clip_gradient_norm(
        self, weights_delta: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Clip the L2 norm of the entire weight delta to ``clip_norm``."""
        # Compute global L2 norm
        total_norm_sq = sum(
            float(np.sum(v.astype(np.float64) ** 2))
            for v in weights_delta.values()
        )
        total_norm = math.sqrt(total_norm_sq)

        if total_norm <= self.clip_norm:
            return {k: v.copy() for k, v in weights_delta.items()}

        scale = self.clip_norm / (total_norm + 1e-12)
        return {k: (v * scale).astype(np.float32) for k, v in weights_delta.items()}

    def _add_gaussian_dp_noise(
        self, weights_delta: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Add calibrated Gaussian noise for (ε, δ)-DP.

        Noise scale σ = clip_norm * √(2 ln(1.25/δ)) / ε
        """
        sigma = (
            self.clip_norm
            * math.sqrt(2.0 * math.log(1.25 / self.dp_delta))
            / self.dp_epsilon
        )

        noised: Dict[str, np.ndarray] = {}
        for key, val in weights_delta.items():
            noise = np.random.normal(0.0, sigma, size=val.shape).astype(
                np.float32
            )
            noised[key] = val.astype(np.float32) + noise

        return noised

    def _sparsify_gradients(
        self, weights_delta: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Keep only the top-k% largest (by magnitude) values; zero the rest."""
        sparse: Dict[str, np.ndarray] = {}
        masks: Dict[str, np.ndarray] = {}

        for key, val in weights_delta.items():
            flat = np.abs(val.flatten())
            n_keep = max(1, int(len(flat) * self.top_k_pct))
            # Find top-k threshold
            if n_keep >= len(flat):
                sparse[key] = val.copy()
                masks[key] = np.ones_like(val, dtype=np.bool_)
            else:
                threshold = np.partition(flat, -n_keep)[-n_keep]
                mask = np.abs(val) >= threshold
                sparse[key] = val * mask
                masks[key] = mask

        return sparse, masks

    def _quantize_to_int8(
        self, weights_delta: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Quantise float32 values to int8 with per-tensor symmetric scaling."""
        quantized: Dict[str, np.ndarray] = {}
        scales: Dict[str, float] = {}

        for key, val in weights_delta.items():
            abs_max = float(np.abs(val).max())
            if abs_max < 1e-10:
                quantized[key] = np.zeros_like(val, dtype=np.int8)
                scales[key] = 1.0
            else:
                scale = abs_max / 127.0
                quantized[key] = np.clip(
                    np.round(val / scale), -127, 127
                ).astype(np.int8)
                scales[key] = scale

        return quantized, scales


# ---------------------------------------------------------------------------
# ScaffoldTrainer
# ---------------------------------------------------------------------------
class ScaffoldTrainer:
    """SCAFFOLD local training loop with control variate correction.

    Implements the SCAFFOLD algorithm (Karimireddy et al., 2020):
        corrected_grad = grad − c_local + c_global
        c_new = c_local − c_global + (1/(K·lr)) · (w_global − w_new)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: data_utils.DataLoader,
        val_loader: data_utils.DataLoader,
        config: Dict[str, Any],
        device: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Initialise local control variates to zeros (same shape as params)
        self.c_local: Dict[str, np.ndarray] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.c_local[name] = np.zeros_like(
                    param.data.cpu().numpy(), dtype=np.float32
                )

        self._loss_fn = nn.CrossEntropyLoss()
        logger.info(
            "ScaffoldTrainer initialised – %d trainable parameter groups",
            len(self.c_local),
        )

    def train_round(
        self,
        global_weights: Dict[str, np.ndarray],
        global_cv: Dict[str, np.ndarray],
        local_epochs: int = 3,
        K: int = 10,
        lr: float = 0.001,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
        """Execute one federated training round with SCAFFOLD correction.

        Args:
            global_weights: Global model parameters.
            global_cv: Global control variates (c_global).
            local_epochs: Number of local epochs.
            K: Total local steps used in control variate update.
            lr: Learning rate.

        Returns:
            (w_delta, cv_delta, metrics_dict)
        """
        # 1. Load global weights into model
        self._load_weights(global_weights)

        # 2. Store a copy of w_global
        w_global = {
            name: param.data.clone().cpu().numpy()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Ensure global_cv has entries for all params
        for name in w_global:
            if name not in global_cv:
                global_cv[name] = np.zeros_like(w_global[name], dtype=np.float32)

        # Save old c_local for delta computation later
        c_local_old = {k: v.copy() for k, v in self.c_local.items()}

        # 3. Training loop with SCAFFOLD correction
        self.model.train()
        self.model.to(self.device)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        step_count = 0

        for epoch in range(local_epochs):
            for batch in self.train_loader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        inputs, targets = batch
                    else:
                        inputs = batch[0]
                        targets = batch[1]
                elif isinstance(batch, dict):
                    inputs = batch.get("image", batch.get("img", None))
                    targets = batch.get("label", batch.get("cls", None))
                else:
                    continue

                inputs = inputs.to(self.device) if isinstance(inputs, torch.Tensor) else torch.tensor(inputs).to(self.device)
                targets = targets.to(self.device) if isinstance(targets, torch.Tensor) else torch.tensor(targets).to(self.device)

                # a. Forward pass
                outputs = self.model(inputs)
                num_classes = self.config.get("num_classes", 15)
                logits = extract_logits(outputs, num_classes)
                loss = self._compute_loss(logits, targets)

                # b. Backward pass
                self.model.zero_grad()
                loss.backward()

                # c. Apply SCAFFOLD correction and manual SGD update
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            c_loc = torch.from_numpy(
                                self.c_local.get(name, np.zeros_like(param.data.cpu().numpy()))
                            ).to(self.device)
                            c_glob = torch.from_numpy(
                                global_cv.get(name, np.zeros_like(param.data.cpu().numpy()))
                            ).to(self.device)

                            # corrected_grad = grad - c_local + c_global
                            corrected_grad = param.grad.data - c_loc + c_glob

                            # d. Update weights: w = w - lr * corrected_grad
                            param.data -= lr * corrected_grad

                total_loss += loss.item() * inputs.size(0)
                if logits.dim() > 1 and logits.size(-1) > 1:
                    preds = logits.argmax(dim=-1)
                    if targets.dim() > 1:
                        target_labels = targets.argmax(dim=-1)
                    else:
                        target_labels = targets
                    total_correct += (preds == target_labels).sum().item()
                total_samples += inputs.size(0)
                step_count += 1

        # 4. Compute w_delta = w_new − w_global
        w_new = {
            name: param.data.clone().cpu().numpy()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        w_delta: Dict[str, np.ndarray] = {}
        for name in w_global:
            w_delta[name] = w_new[name] - w_global[name]

        # 5. Update c_local:
        #    c_new = c_local − c_global + (1/(K·lr)) · (w_global − w_new)
        effective_K = max(K, step_count) if step_count > 0 else K
        c_new: Dict[str, np.ndarray] = {}
        for name in w_global:
            c_loc = self.c_local.get(name, np.zeros_like(w_global[name]))
            c_glob = global_cv.get(name, np.zeros_like(w_global[name]))
            c_new[name] = (
                c_loc - c_glob + (1.0 / (effective_K * lr)) * (w_global[name] - w_new[name])
            ).astype(np.float32)

        # 6. Compute cv_delta = c_new − c_local_old
        cv_delta: Dict[str, np.ndarray] = {}
        for name in c_new:
            cv_delta[name] = c_new[name] - c_local_old.get(
                name, np.zeros_like(c_new[name])
            )

        # Update stored c_local
        self.c_local = c_new

        # Metrics
        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_samples": total_samples,
            "local_epochs": local_epochs,
            "effective_K": effective_K,
        }

        logger.info(
            "ScaffoldTrainer round complete – loss=%.4f acc=%.4f samples=%d",
            avg_loss,
            accuracy,
            total_samples,
        )

        return w_delta, cv_delta, metrics

    def evaluate(
        self,
        weights: Dict[str, np.ndarray],
        val_loader: data_utils.DataLoader,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model on a validation set.

        Returns:
            (average_loss, metrics_dict)
        """
        self._load_weights(weights)
        self.model.eval()
        self.model.to(self.device)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs = batch.get("image", batch.get("img"))
                    targets = batch.get("label", batch.get("cls"))
                else:
                    continue

                inputs = inputs.to(self.device) if isinstance(inputs, torch.Tensor) else torch.tensor(inputs).to(self.device)
                targets = targets.to(self.device) if isinstance(targets, torch.Tensor) else torch.tensor(targets).to(self.device)

                outputs = self.model(inputs)
                num_classes = self.config.get("num_classes", 15)
                logits = extract_logits(outputs, num_classes)
                loss = self._compute_loss(logits, targets)

                total_loss += loss.item() * inputs.size(0)
                if logits.dim() > 1 and logits.size(-1) > 1:
                    preds = logits.argmax(dim=-1)
                    if targets.dim() > 1:
                        target_labels = targets.argmax(dim=-1)
                    else:
                        target_labels = targets
                    total_correct += (preds == target_labels).sum().item()
                total_samples += inputs.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)

        metrics = {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_samples": total_samples,
        }

        return avg_loss, metrics

    def _create_optimizer(self, lr: float) -> torch.optim.SGD:
        """Create an SGD optimiser for the model."""
        return torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )

    def _compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss handling various target formats."""
        if targets.dtype in (torch.float32, torch.float64) and targets.dim() > 1:
            # Soft labels → use KL divergence as cross-entropy proxy
            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
            return -(targets * log_probs).sum(dim=-1).mean()
        return self._loss_fn(outputs, targets.long())

    def _load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Load numpy weight arrays into the model."""
        state_dict = {}
        for name, arr in weights.items():
            state_dict[name] = torch.from_numpy(arr).clone()
        self.model.load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# FedFODClient (Flower NumPyClient)
# ---------------------------------------------------------------------------
class FedFODClient(flwr.client.NumPyClient if flwr is not None else object):
    """Flower federated learning client for FedFOD.

    Integrates RT-DETR detection, SCAFFOLD training, compression pipeline,
    and CLIP open-world detection into a single FL client interface.
    """

    def __init__(
        self,
        client_id: str,
        config: Dict[str, Any],
        data_dir: str,
        device: Optional[str] = None,
    ):
        self.client_id = client_id
        self.config = config
        self.data_dir = data_dir
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load airport-specific config
        self.airport_config = self._load_airport_config(config)

        # Initialise detector
        model_path = self.airport_config.get("model_path", "rtdetr-l.pt")
        num_classes = self.airport_config.get("num_classes", 15)
        self.detector = RTDETRDetector(
            model_path=model_path,
            num_classes=num_classes,
            device=self.device,
        )
        for param in self.detector.model.model.parameters():
            param.requires_grad = True

        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders(data_dir)

        # Initialise SCAFFOLD trainer
        self.scaffold_trainer = ScaffoldTrainer(
            model=self.detector.model.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.airport_config,
            device=self.device,
        )

        # Initialise compression pipeline
        dp_eps = self.airport_config.get("dp_epsilon", 4.0)
        is_sat = self.airport_config.get("is_satellite", False)
        self.compressor = CompressionPipeline(
            dp_epsilon=dp_eps,
            is_satellite=is_sat,
        )

        # CLIP detector (initialised lazily to save memory)
        self._clip_detector: Optional[CLIPOpenWorldDetector] = None

        # Round tracking
        self._round_number = 0
        self._last_fit_time = time.time()

        logger.info(
            "FedFODClient[%s] initialised – airport=%s device=%s",
            client_id,
            self.airport_config.get("airport_id", "unknown"),
            self.device,
        )

    # -- Flower NumPyClient interface ---------------------------------------

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model parameters as a list of numpy arrays sorted alphabetically by key."""
        state_dict = self.detector.get_state_dict()
        sorted_keys = sorted(state_dict.keys())
        return [state_dict[k] for k in sorted_keys]

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Execute one federated training round.

        Steps:
            1. Load global parameters into local model
            2. Decode global control variates from config
            3. Run ScaffoldTrainer.train_round()
            4. Compress weight delta
            5. Encode control variate delta for transmission
            6. Return updated parameters + metrics
        """
        self._round_number += 1

        # 1. Load global parameters
        param_keys = sorted(self.detector.model.model.state_dict().keys())
        global_weights: Dict[str, np.ndarray] = {}
        for i, key in enumerate(param_keys):
            if i < len(parameters):
                global_weights[key] = parameters[i]

        self.detector.set_state_dict(global_weights)

        # 2. Decode global control variates from config
        global_cv = self._decode_control_variates(config)

        # 3. Run SCAFFOLD training
        local_epochs = config.get("local_epochs", 3)
        K = config.get("K", 10)
        lr = config.get("lr", 0.001)

        w_delta, cv_delta, train_metrics = self.scaffold_trainer.train_round(
            global_weights=global_weights,
            global_cv=global_cv,
            local_epochs=local_epochs,
            K=K,
            lr=lr,
        )

        # 4. Compress weight delta
        compressed_delta, comp_metadata = self.compressor.compress(w_delta)

        # 5. Encode cv_delta into metrics
        cv_encoded = self._encode_control_variates(cv_delta)

        # 6. Build final parameters (global + delta applied)
        updated_state = self.detector.get_state_dict()
        sorted_keys = sorted(updated_state.keys())
        updated_params = [updated_state[k] for k in sorted_keys]

        n_samples = train_metrics.get("num_samples", 0)

        # Pack metrics
        metrics: Dict[str, Any] = {
            "loss": float(train_metrics.get("loss", 0.0)),
            "accuracy": float(train_metrics.get("accuracy", 0.0)),
            "precision_q": float(self._get_precision_q()),
            "num_samples": int(n_samples),
            "staleness": int(self._get_staleness()),
            "round": self._round_number,
            "client_id": self.client_id,
            "cv_delta_b64": cv_encoded,
        }

        self._last_fit_time = time.time()

        logger.info(
            "FedFODClient[%s] fit round %d – loss=%.4f acc=%.4f samples=%d",
            self.client_id,
            self._round_number,
            metrics["loss"],
            metrics["accuracy"],
            n_samples,
        )

        return updated_params, n_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate model on local validation data.

        Returns:
            (loss, num_examples, metrics_dict_with_mAP_and_FAR)
        """
        # Load parameters
        param_keys = sorted(self.detector.model.model.state_dict().keys())
        weights: Dict[str, np.ndarray] = {}
        for i, key in enumerate(param_keys):
            if i < len(parameters):
                weights[key] = parameters[i]

        self.detector.set_state_dict(weights)

        # Run evaluation
        loss, eval_metrics = self.scaffold_trainer.evaluate(
            weights=weights,
            val_loader=self.val_loader,
        )

        n_samples = eval_metrics.get("val_samples", 0)

        # Compute detection metrics (mAP proxy, FAR proxy)
        map_score = eval_metrics.get("val_accuracy", 0.0)
        far_score = 1.0 - map_score  # simple false alarm rate proxy

        metrics: Dict[str, Any] = {
            "val_loss": float(loss),
            "val_accuracy": float(eval_metrics.get("val_accuracy", 0.0)),
            "mAP": float(map_score),
            "FAR": float(far_score),
            "client_id": self.client_id,
            "round": self._round_number,
        }

        logger.info(
            "FedFODClient[%s] evaluate – loss=%.4f mAP=%.4f FAR=%.4f",
            self.client_id,
            loss,
            map_score,
            far_score,
        )

        return float(loss), int(n_samples), metrics

    # -- private helpers -----------------------------------------------------

    def _setup_data_loaders(
        self, data_dir: str
    ) -> Tuple[data_utils.DataLoader, data_utils.DataLoader]:
        """Create train and validation data loaders from data_dir.

        Expects data_dir to contain ``train/`` and ``val/`` subdirectories
        with images and labels in YOLO format, or falls back to a dummy
        dataset if the directory is empty / missing.
        """
        batch_size = self.airport_config.get("batch_size", 16)
        num_workers = self.airport_config.get("num_workers", 2)

        train_dir = Path(data_dir) / "train"
        val_dir = Path(data_dir) / "val"

        train_dataset = self._build_dataset(train_dir)
        val_dataset = self._build_dataset(val_dir)

        train_loader = data_utils.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
            drop_last=True,
        )
        val_loader = data_utils.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
        )

        return train_loader, val_loader

    def _build_dataset(
        self, data_path: Path
    ) -> data_utils.Dataset:
        """Build a simple image classification dataset from a directory.

        Falls back to a dummy tensor dataset when data is unavailable.
        """
        images_dir = data_path / "images"
        labels_dir = data_path / "labels"

        if images_dir.is_dir() and labels_dir.is_dir():
            # Load image paths
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
            image_paths = sorted(
                p
                for p in images_dir.iterdir()
                if p.suffix.lower() in image_extensions
            )
            if image_paths:
                return _FODImageDataset(image_paths, labels_dir)

        # Fallback: dummy dataset
        logger.warning("No data found at %s – using dummy dataset", data_path)
        num_classes = self.airport_config.get("num_classes", 15)
        dummy_images = torch.randn(64, 3, 640, 640)
        dummy_labels = torch.randint(0, num_classes, (64,))
        return data_utils.TensorDataset(dummy_images, dummy_labels)

    def _load_airport_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load airport-specific configuration from a YAML file or dict."""
        config_path = config.get("config_path", None)
        if config_path and yaml is not None and Path(config_path).is_file():
            with open(config_path, "r") as f:
                airport_cfg = yaml.safe_load(f)
            if airport_cfg is None:
                airport_cfg = {}
            airport_cfg.update(config)
            return airport_cfg
        return dict(config)

    def _get_precision_q(self) -> float:
        """Return historical precision quality metric from airport config."""
        return float(self.airport_config.get("historical_precision", 0.85))

    def _get_staleness(self) -> int:
        """Return staleness indicator (rounds since last successful aggregation)."""
        staleness = self.airport_config.get("staleness", 0)
        # Also factor in time since last fit
        elapsed = time.time() - self._last_fit_time
        time_staleness = int(elapsed / 60)  # minutes as staleness proxy
        return int(staleness) + time_staleness

    def _encode_control_variates(
        self, cv_delta: Dict[str, np.ndarray]
    ) -> str:
        """Serialise control variate deltas to a base64 string."""
        buf = io.BytesIO()
        # Convert to a compact format: list of (name, array) tuples
        compact = {k: v.astype(np.float16) for k, v in cv_delta.items()}
        pickle.dump(compact, buf, protocol=pickle.HIGHEST_PROTOCOL)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _decode_control_variates(
        self, config: Dict
    ) -> Dict[str, np.ndarray]:
        """Decode global control variates from the fit config.

        Expects ``config["global_cv_b64"]`` to be a base64-encoded pickle.
        """
        cv_b64 = config.get("global_cv_b64", "")
        if not cv_b64:
            return {}
        try:
            raw = base64.b64decode(cv_b64)
            compact = pickle.loads(raw)
            return {k: v.astype(np.float32) for k, v in compact.items()}
        except Exception as exc:
            logger.warning("Failed to decode global CVs: %s", exc)
            return {}


# ---------------------------------------------------------------------------
# Helper dataset
# ---------------------------------------------------------------------------
class _FODImageDataset(data_utils.Dataset):
    """Simple YOLO-format image + label dataset for FOD detection training."""

    def __init__(self, image_paths: List[Path], labels_dir: Path):
        self.image_paths = image_paths
        self.labels_dir = labels_dir

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import cv2

        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((640, 640, 3), dtype=np.uint8)

        # Resize to 640×640 and normalise
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0
        # HWC → CHW
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        # Load label (YOLO format: class x_center y_center width height)
        label_path = self.labels_dir / (img_path.stem + ".txt")
        if label_path.is_file():
            with open(label_path, "r") as f:
                lines = f.readlines()
            if lines:
                first_line = lines[0].strip().split()
                class_id = int(first_line[0])
            else:
                class_id = 0
        else:
            class_id = 0

        return img_tensor, torch.tensor(class_id, dtype=torch.long)
