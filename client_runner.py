#!/usr/bin/env python3
"""FedFOD Client Runner — Live gRPC Deployment
================================================

Starts a single Flower federated learning client that connects to a
running FedFOD server over gRPC.  Each client represents one airport.

Usage:
    python client_runner.py --client-id 0 --server localhost:8080
    python client_runner.py --client-id 1 --server localhost:8080 \
        --airport-config config/airport_configs/airport_B.yaml \
        --data-dir data/airport_B
    python client_runner.py --client-id 0 --server localhost:50051 --dummy-model
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils

# ---- Optional imports ----
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

import flwr as fl
import flwr.client

# ---- Project imports ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.common.utils import extract_logits

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("fedfod.client_runner")


# ===================================================================== #
#                      DUMMY MODEL                                       #
# ===================================================================== #

class DummyFODModel(nn.Module):
    """Lightweight stand-in for RT-DETR-L when ultralytics is unavailable."""

    def __init__(self, num_classes: int = 15):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes * 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ===================================================================== #
#                  LIVE FEDERATED CLIENT                                 #
# ===================================================================== #

class LiveFedFODClient(fl.client.NumPyClient):
    """Flower NumPyClient for live gRPC deployment.

    This client:
    - Loads a model (DummyFODModel or RT-DETR-L)
    - Runs SCAFFOLD-corrected local training
    - Reports metrics back to the server
    """

    def __init__(
        self,
        client_id: str,
        config: Dict[str, Any],
        airport_config: Dict[str, Any],
        data_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_dummy_model: bool = False,
    ):
        super().__init__()
        self.client_id = client_id
        self.config = config
        self.airport_config = airport_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._round_number = 0
        self._use_dummy = use_dummy_model

        fl_cfg = config.get("fl", {})
        self.local_epochs = fl_cfg.get("local_epochs", 3)
        self.lr = fl_cfg.get("learning_rate", 0.001)
        self.K = fl_cfg.get("local_steps_K", 10)

        num_classes = config.get("model", {}).get("num_classes", 15)

        # ---- Model ----
        self.model = self._build_model(num_classes)

        # ---- Data ----
        self.train_loader = self._setup_data(data_dir, num_classes)

        # ---- SCAFFOLD control variates (zeros matching trainable params) ----
        self.c_local: Dict[str, np.ndarray] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.c_local[name] = np.zeros_like(
                    param.data.cpu().numpy(), dtype=np.float32
                )

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "[Client %s] Initialised — device=%s, trainable_params=%s, dummy=%s",
            client_id, self.device, f"{n_params:,}", use_dummy_model,
        )

    # ------------------------------------------------------------------ #
    #  Model construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_model(self, num_classes: int) -> nn.Module:
        """Build the model — DummyFODModel, RT-DETR, or YOLO inner module."""
        if self._use_dummy:
            logger.info("[Client %s] Using DummyFODModel.", self.client_id)
            return DummyFODModel(num_classes=num_classes).to(self.device)

        backbone = self.config.get("model", {}).get("backbone", "rtdetr-l")
        try:
            if "yolo" in backbone.lower():
                from ultralytics import YOLO
                model_name = backbone if backbone.endswith(".pt") else f"{backbone}.pt"
                yolo = YOLO(model_name)
                model = yolo.model
            else:
                from ultralytics import RTDETR
                model_name = backbone if backbone.endswith(".pt") else "rtdetr-l.pt"
                rtdetr = RTDETR(model_name)
                model = rtdetr.model

            for param in model.parameters():
                param.requires_grad = True
            model.to(self.device)
            logger.info("[Client %s] Loaded %s inner model.", self.client_id, backbone)
            return model
        except Exception as exc:
            logger.warning(
                "[Client %s] %s unavailable (%s), falling back to DummyFODModel.",
                self.client_id, backbone, exc,
            )
            self._use_dummy = True
            return DummyFODModel(num_classes=num_classes).to(self.device)

    # ------------------------------------------------------------------ #
    #  Flower NumPyClient interface                                       #
    # ------------------------------------------------------------------ #

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model parameters as a list of numpy arrays sorted alphabetically by key."""
        state_dict = self.model.state_dict()
        sorted_keys = sorted(state_dict.keys())
        return [state_dict[k].cpu().numpy() for k in sorted_keys]

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Execute one federated training round."""
        self._round_number += 1
        server_round = config.get("server_round", self._round_number)

        logger.info("[Client %s] Starting fit for round %d …",
                     self.client_id, server_round)

        # 1. Load global parameters into model
        self._set_parameters(parameters)

        # 2. Store pre-training weights
        w_global = {
            name: param.data.clone().cpu().numpy()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # 3. Decode global control variates (if available)
        global_cv = self._decode_cv_from_config(config)

        # 4. Local training with SCAFFOLD correction
        total_loss, total_samples, step_count = self._train_local(
            w_global, global_cv
        )

        # 5. Update SCAFFOLD control variates
        w_new = {
            name: param.data.clone().cpu().numpy()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        effective_K = max(self.K, step_count) if step_count > 0 else self.K
        c_new: Dict[str, np.ndarray] = {}
        for name in w_global:
            c_loc = self.c_local.get(name, np.zeros_like(w_global[name]))
            c_glob = global_cv.get(name, np.zeros_like(w_global[name]))
            c_new[name] = (
                c_loc - c_glob
                + (1.0 / (effective_K * self.lr))
                * (w_global[name] - w_new[name])
            ).astype(np.float32)
        self.c_local = c_new

        # 6. Return updated full parameters
        state_dict = self.model.state_dict()
        sorted_keys = sorted(state_dict.keys())
        updated_params = [state_dict[k].cpu().numpy() for k in sorted_keys]

        avg_loss = total_loss / max(total_samples, 1)
        precision_q = float(
            self.airport_config.get("data", {}).get("quality_score_q", 0.85)
        )
        metrics = {
            "loss": float(avg_loss),
            "num_samples": int(total_samples),
            "client_id": self.client_id,
            "round": int(server_round),
            "precision_q": precision_q,
        }

        logger.info(
            "[Client %s] Round %d done — loss=%.4f, samples=%d, steps=%d",
            self.client_id, server_round, avg_loss, total_samples, step_count,
        )

        return updated_params, total_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate model on local data."""
        self._set_parameters(parameters)
        self.model.eval()
        self.model.to(self.device)

        total_loss = 0.0
        total_samples = 0
        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                num_classes = self.config.get("model", {}).get("num_classes", 15)
                logits = extract_logits(outputs, num_classes)
                loss = loss_fn(logits, targets.long())
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        return float(avg_loss), total_samples, {"client_id": self.client_id}

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load a list of numpy arrays into the model state dict."""
        state_dict = self.model.state_dict()
        sorted_keys = sorted(state_dict.keys())
        new_state = {}
        for i, key in enumerate(sorted_keys):
            if i < len(parameters):
                new_state[key] = torch.from_numpy(parameters[i]).to(
                    dtype=state_dict[key].dtype
                )
            else:
                new_state[key] = state_dict[key]
        self.model.load_state_dict(new_state, strict=False)

    def _train_local(
        self,
        w_global: Dict[str, np.ndarray],
        global_cv: Dict[str, np.ndarray],
    ) -> Tuple[float, int, int]:
        """Run local epochs with SCAFFOLD gradient correction.

        Returns (total_loss, total_samples, step_count).
        """
        self.model.train()
        self.model.to(self.device)
        loss_fn = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_samples = 0
        step_count = 0

        for epoch in range(self.local_epochs):
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                num_classes = self.config.get("model", {}).get("num_classes", 15)
                logits = extract_logits(outputs, num_classes)
                loss = loss_fn(logits, targets.long())

                self.model.zero_grad()
                loss.backward()

                # SCAFFOLD correction: corrected = grad − c_local + c_global
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            c_loc = torch.from_numpy(
                                self.c_local.get(
                                    name,
                                    np.zeros_like(param.data.cpu().numpy()),
                                )
                            ).to(self.device)
                            c_glob = torch.from_numpy(
                                global_cv.get(
                                    name,
                                    np.zeros_like(param.data.cpu().numpy()),
                                )
                            ).to(self.device)
                            param.grad.data = param.grad.data - c_loc + c_glob
                            param.data -= self.lr * param.grad.data

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                step_count += 1

        return total_loss, total_samples, step_count

    def _setup_data(
        self, data_dir: Optional[str], num_classes: int
    ) -> data_utils.DataLoader:
        """Build a DataLoader from a YOLO directory or a dummy dataset."""
        from pathlib import Path

        if not self._use_dummy and data_dir and Path(data_dir).is_dir():
            images_dir = Path(data_dir) / "train" / "images"
            labels_dir = Path(data_dir) / "train" / "labels"
            if images_dir.is_dir() and labels_dir.is_dir():
                exts = {".jpg", ".jpeg", ".png", ".bmp"}
                paths = sorted(
                    p for p in images_dir.iterdir()
                    if p.suffix.lower() in exts
                )
                if paths:
                    logger.info(
                        "[Client %s] Found %d training images in %s",
                        self.client_id, len(paths), data_dir,
                    )
                    from src.client.local_trainer import _FODImageDataset
                    dataset = _FODImageDataset(paths, labels_dir)
                    return data_utils.DataLoader(
                        dataset, batch_size=16, shuffle=True, drop_last=True,
                    )

        # Fallback: dummy dataset matching the model's input shape
        logger.info(
            "[Client %s] No data at '%s' — using dummy dataset.",
            self.client_id, data_dir,
        )

        if self._use_dummy:
            # DummyFODModel expects 256-dim input
            dummy_images = torch.randn(64, 256)
        else:
            # RT-DETR expects 3×640×640 images
            dummy_images = torch.randn(64, 3, 640, 640)

        dummy_labels = torch.randint(0, num_classes, (64,))
        dataset = data_utils.TensorDataset(dummy_images, dummy_labels)
        return data_utils.DataLoader(
            dataset, batch_size=16, shuffle=True, drop_last=True,
        )

    def _decode_cv_from_config(self, config: Dict) -> Dict[str, np.ndarray]:
        """Decode base64-encoded global control variates from fit config."""
        import base64
        import pickle

        cv_b64 = config.get("global_cv_b64", "")
        if not cv_b64:
            return {}
        try:
            raw = base64.b64decode(cv_b64)
            compact = pickle.loads(raw)
            return {k: v.astype(np.float32) for k, v in compact.items()}
        except Exception as exc:
            logger.warning("[Client %s] Failed to decode CVs: %s",
                           self.client_id, exc)
            return {}


# ===================================================================== #
#                         MAIN                                           #
# ===================================================================== #

def load_yaml(path: str) -> dict:
    """Load a YAML config file."""
    if not HAS_YAML:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FedFOD — Federated Learning Client (gRPC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--client-id", type=str, required=True,
                        help="Unique client identifier (e.g. '0', '1', '2')")
    parser.add_argument("--config", type=str, default="config/global_config.yaml",
                        help="Path to global YAML config")
    parser.add_argument("--airport-config", type=str, default=None,
                        help="Path to airport-specific YAML config")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to local YOLO-format dataset directory")
    parser.add_argument("--server", type=str, default="localhost:8080",
                        help="Server address (host:port)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda:0 | cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--insecure", action="store_true", default=True,
                        help="Use insecure gRPC (no TLS)")
    parser.add_argument("--dummy-model", action="store_true",
                        help="Use DummyFODModel instead of RT-DETR-L (for testing)")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Model backbone to use (overrides config)")

    args = parser.parse_args()

    # ---- Reproducibility ----
    np.random.seed(args.seed + hash(args.client_id) % 1000)
    torch.manual_seed(args.seed + hash(args.client_id) % 1000)

    # ---- Load configs ----
    global_config = load_yaml(args.config)
    airport_config = load_yaml(args.airport_config) if args.airport_config else {}

    # Override backbone if provided
    if args.backbone:
        if "model" not in global_config:
            global_config["model"] = {}
        global_config["model"]["backbone"] = args.backbone

    # Infer data_dir from airport config if not specified
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = airport_config.get("data", {}).get(
            "data_dir", f"data/airport_{args.client_id}"
        )

    # ---- Build client ----
    client = LiveFedFODClient(
        client_id=args.client_id,
        config=global_config,
        airport_config=airport_config,
        data_dir=data_dir,
        device=args.device,
        use_dummy_model=args.dummy_model,
    )

    logger.info("=" * 60)
    logger.info("FedFOD Client — Live gRPC Deployment")
    logger.info("=" * 60)
    logger.info("  Client ID:    %s", args.client_id)
    logger.info("  Server:       %s", args.server)
    logger.info("  Data dir:     %s", data_dir)
    logger.info("  Device:       %s", client.device)
    logger.info("  Dummy model:  %s", args.dummy_model)
    logger.info("=" * 60)

    # ---- Connect to server ----
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client,
        grpc_max_message_length=1024 * 1024 * 512,  # 512 MB
        insecure=args.insecure,
    )

    logger.info("[Client %s] Finished. Shutting down.", args.client_id)


if __name__ == "__main__":
    main()
