"""
FedFOD Common Utilities
========================

Core dataclasses, SCAFFOLD helpers, gradient compression / quantisation,
differential-privacy noise injection, federated weighting schemes,
and general-purpose numeric helpers used throughout the pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device selection (CPU fallback)
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Constants used by server modules
# ---------------------------------------------------------------------------
STALENESS_DECAY: float = 0.85
MAX_GINI_COEFFICIENT: float = 0.35
SHAMIR_SAFE_PRIME: int = (1 << 61) - 1  # Mersenne prime 2^61-1
COSINE_SIM_ACCEPT_THRESHOLD: float = 0.70
COSINE_SIM_MERGE_THRESHOLD: float = 0.85
DEFAULT_EMBED_DIM: int = 512
PROTOTYPE_MAX_ENTRIES: int = 200
PROTOTYPE_STALE_ROUNDS: int = 50


# ===================================================================== #
#     CONFIG / METRICS DATACLASSES (used by aggregator + server)        #
# ===================================================================== #

@dataclass
class GlobalConfig:
    """Experiment-wide configuration passed to the aggregator."""

    num_rounds: int = 90
    num_clients: int = 3
    min_clients: int = 2
    min_available: int = 3
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    learning_rate: float = 0.001
    lr_decay: float = 0.995
    scaffold_enabled: bool = True
    dp_enabled: bool = True
    dp_epsilon: float = 4.0
    dp_delta: float = 1e-6
    wandb_enabled: bool = False
    wandb_project: str = "fedfod-research"
    experiment_name: str = "fedfod-run"
    checkpoint_dir: str = "checkpoints"


@dataclass
class ClientMetrics:
    """Per-client metrics reported back to the aggregator each round."""

    client_id: str = ""
    num_samples: int = 0
    precision_q: float = 1.0
    staleness: int = 0
    train_loss: float = 0.0
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    novel_detections: int = 0
    round_number: int = 0


@dataclass
class AggregationResult:
    """Summary of one aggregation round."""

    round_number: int = 0
    num_clients: int = 0
    effective_weights: List[float] = field(default_factory=list)
    gini_coefficient: float = 0.0
    gini_valid: bool = True
    global_loss: float = 0.0
    global_mAP50: float = 0.0
    global_mAP50_95: float = 0.0


@dataclass
class SMPCShare:
    """A single Shamir share (x, y) for one secret value."""

    x: int = 0
    y: int = 0


# ---------------------------------------------------------------------------
# Serialization helpers (base64 numpy encode/decode for Flower config dicts)
# ---------------------------------------------------------------------------

import base64
import io


def encode_ndarray_b64(arr: np.ndarray) -> str:
    """Serialize a numpy array to a base64 string for Flower config transport."""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_ndarray_b64(b64_str: str) -> np.ndarray:
    """Deserialize a base64 string back to a numpy array."""
    raw = base64.b64decode(b64_str.encode("ascii"))
    buf = io.BytesIO(raw)
    return np.load(buf, allow_pickle=False)


# ---------------------------------------------------------------------------
# Learning rate schedule + W&B helper
# ---------------------------------------------------------------------------

def get_lr_for_round(config: GlobalConfig, server_round: int) -> float:
    """Cosine-decay learning rate for a given round."""
    return config.learning_rate * (config.lr_decay ** server_round)


def staleness_weight(staleness: int, decay: float = STALENESS_DECAY) -> float:
    """Exponential staleness penalty (alias for compute_staleness_weight)."""
    capped = min(staleness, 15)
    return float(decay ** capped)


def try_import_wandb():
    """Try to import wandb; return module or None."""
    try:
        import wandb
        return wandb
    except ImportError:
        logger.info("wandb not installed — W&B logging disabled.")
        return None

# ===================================================================== #
#                           DATA-CLASSES                                #
# ===================================================================== #

@dataclass
class GradientPayload:
    """Encapsulates everything a client ships to the aggregator each round."""

    client_id: str
    round_number: int
    weights_delta: Dict[str, np.ndarray]
    control_variate_delta: Optional[Dict[str, np.ndarray]] = None
    rolling_precision_q: float = 1.0
    num_samples: int = 0
    is_async: bool = False
    staleness_rounds: int = 0
    payload_size_bytes: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class PrototypePayload:
    """A single FOD-candidate prototype shipped for novel-class discovery."""

    client_id: str
    embedding_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(512, dtype=np.float32)
    )
    detection_confidence: float = 0.0
    fod_candidate_class: str = "unknown"
    context_metadata: Dict[str, Any] = field(default_factory=dict)
    novelty_distance: float = 0.0


@dataclass
class GlobalUpdate:
    """Broadcast from aggregator → clients after a round of aggregation."""

    round_number: int
    global_weights: Dict[str, np.ndarray]
    global_control_variate: Optional[Dict[str, np.ndarray]] = None
    active_novel_prototypes: List[np.ndarray] = field(default_factory=list)
    novel_class_names: List[str] = field(default_factory=list)
    aggregation_metadata: Dict[str, Any] = field(default_factory=dict)


# ===================================================================== #
#                         SCAFFOLD HELPERS                              #
# ===================================================================== #

def scaffold_local_update(
    w_local: Dict[str, np.ndarray],
    w_global: Dict[str, np.ndarray],
    c_local: Dict[str, np.ndarray],
    c_global: Dict[str, np.ndarray],
    grad: Dict[str, np.ndarray],
    eta: float,
    K: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """One SCAFFOLD local correction step.

    w_new[k]  = w_local[k] - eta * (grad[k] - c_local[k] + c_global[k])
    c_new[k]  = c_local[k] - c_global[k] + (1/(K*eta)) * (w_global[k] - w_new[k])

    Returns
    -------
    w_new, c_new : dicts of np.ndarray
    """
    w_new: Dict[str, np.ndarray] = {}
    c_new: Dict[str, np.ndarray] = {}

    for k in w_local:
        corrected_grad = grad[k] - c_local[k] + c_global[k]
        w_new[k] = w_local[k] - eta * corrected_grad

        c_new[k] = (
            c_local[k]
            - c_global[k]
            + (1.0 / (K * eta)) * (w_global[k] - w_new[k])
        )

    return w_new, c_new


def compute_control_variate_delta(
    c_new: Dict[str, np.ndarray],
    c_old: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Δc = c_new − c_old  (per-layer)."""
    return {k: c_new[k] - c_old[k] for k in c_new}


def apply_global_control_update(
    c_global: Dict[str, np.ndarray],
    c_deltas: List[Dict[str, np.ndarray]],
    n_clients: int,
) -> Dict[str, np.ndarray]:
    """c_global_new = c_global + (1/n) * Σ Δc_i."""
    c_new: Dict[str, np.ndarray] = {}
    for k in c_global:
        summed = np.zeros_like(c_global[k])
        for delta in c_deltas:
            summed = summed + delta[k]
        c_new[k] = c_global[k] + summed / n_clients
    return c_new


# ===================================================================== #
#                    GRADIENT COMPRESSION / QUANTISATION                #
# ===================================================================== #

def sparsify_gradients(
    weights_delta: Dict[str, np.ndarray],
    top_k_pct: float = 0.05,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Top-k sparsification — keep only the largest ``top_k_pct`` entries.

    Returns
    -------
    sparse : dict of np.ndarray  (zeroed-out below threshold)
    masks  : dict of bool np.ndarray
    """
    sparse: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}

    for k, v in weights_delta.items():
        flat = np.abs(v).ravel()
        n_keep = max(1, int(math.ceil(flat.size * top_k_pct)))
        # partition so the top-n_keep values are at the end
        if n_keep >= flat.size:
            threshold = 0.0
        else:
            threshold = np.partition(flat, -n_keep)[-n_keep]

        mask = np.abs(v) >= threshold
        sparse[k] = v * mask
        masks[k] = mask

    return sparse, masks


def quantize_to_int8(
    weights_delta: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Symmetric min-max quantisation to int8.

    scale = max(|W|) / 127
    quantized = round(W / scale).clip(-128, 127).astype(int8)

    Returns
    -------
    quantized : dict of int8 np.ndarray
    scales    : dict of float64 np.ndarray (scalar per tensor)
    """
    quantized: Dict[str, np.ndarray] = {}
    scales: Dict[str, np.ndarray] = {}

    for k, v in weights_delta.items():
        abs_max = np.abs(v).max()
        if abs_max < 1e-12:
            scale = np.float64(1.0)
        else:
            scale = np.float64(abs_max / 127.0)

        q = np.round(v / scale).clip(-128, 127).astype(np.int8)
        quantized[k] = q
        scales[k] = np.array(scale)

    return quantized, scales


def dequantize_from_int8(
    quantized: Dict[str, np.ndarray],
    scales: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Reverse of :func:`quantize_to_int8`."""
    return {
        k: quantized[k].astype(np.float32) * float(scales[k])
        for k in quantized
    }


# ===================================================================== #
#                       PAYLOAD VALIDATION                              #
# ===================================================================== #

def compute_payload_size_bytes(weights_delta: Dict[str, np.ndarray]) -> int:
    """Sum of in-memory buffer sizes for every array in *weights_delta*."""
    return sum(v.nbytes for v in weights_delta.values())


def validate_payload_size(
    weights_delta: Dict[str, np.ndarray],
    max_mb: float = 2.0,
    is_satellite: bool = False,
) -> bool:
    """Return True if payload fits in the bandwidth budget.

    Satellite links get a stricter 0.5 MB default if *is_satellite* is set.
    """
    effective_max = 0.5 if is_satellite else max_mb
    size_mb = compute_payload_size_bytes(weights_delta) / (1024 * 1024)
    ok = size_mb <= effective_max
    if not ok:
        logger.warning(
            "Payload %.2f MB exceeds limit %.2f MB (satellite=%s)",
            size_mb,
            effective_max,
            is_satellite,
        )
    return ok


# ===================================================================== #
#               DIFFERENTIAL PRIVACY / GRADIENT CLIPPING               #
# ===================================================================== #

def add_gaussian_dp_noise(
    weights_delta: Dict[str, np.ndarray],
    epsilon: float = 4.0,
    delta: float = 1e-6,
    clip_norm: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Clip + add calibrated Gaussian noise for (ε, δ)-DP.

    σ = sqrt(2 · ln(1.25 / δ)) · clip_norm / ε
    """
    clipped = clip_gradient_norm(weights_delta, max_norm=clip_norm)
    sigma = math.sqrt(2.0 * math.log(1.25 / delta)) * clip_norm / epsilon
    logger.debug("DP noise σ = %.6f  (ε=%.2f, δ=%.1e)", sigma, epsilon, delta)

    noisy: Dict[str, np.ndarray] = {}
    for k, v in clipped.items():
        noise = np.random.normal(loc=0.0, scale=sigma, size=v.shape).astype(
            v.dtype
        )
        noisy[k] = v + noise
    return noisy


def clip_gradient_norm(
    weights_delta: Dict[str, np.ndarray],
    max_norm: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Per-update global L2 norm clipping."""
    total_norm_sq = sum(
        float(np.sum(v.astype(np.float64) ** 2)) for v in weights_delta.values()
    )
    total_norm = math.sqrt(total_norm_sq)
    clip_coeff = min(1.0, max_norm / (total_norm + 1e-12))

    if clip_coeff < 1.0:
        logger.debug(
            "Clipping gradient norm %.4f → %.4f", total_norm, max_norm
        )
        return {k: (v * clip_coeff).astype(v.dtype) for k, v in weights_delta.items()}
    return {k: v.copy() for k, v in weights_delta.items()}


# ===================================================================== #
#                      FEDERATED WEIGHTING                              #
# ===================================================================== #

def compute_staleness_weight(
    staleness_rounds: int,
    base: float = 0.85,
) -> float:
    """Exponential staleness discount: base^min(staleness, 15)."""
    capped = min(staleness_rounds, 15)
    return float(base ** capped)


def compute_fa_weight(
    historical_precision: float,
    num_samples: int,
    total_samples: int,
) -> float:
    """FedAvg-style quality weight: q_i = precision × (n_i / N)."""
    if total_samples <= 0:
        return 0.0
    return historical_precision * (num_samples / total_samples)


def normalize_weights(weights: List[float]) -> List[float]:
    """Normalise a list of non-negative weights so they sum to 1.0."""
    total = sum(weights)
    if total <= 0:
        n = len(weights)
        return [1.0 / n] * n if n > 0 else []
    return [w / total for w in weights]


# ===================================================================== #
#                     SIMILARITY / DISTANCE                             #
# ===================================================================== #

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat vectors."""
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    dot = float(np.dot(a_flat, b_flat))
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two flat vectors."""
    diff = a.ravel().astype(np.float64) - b.ravel().astype(np.float64)
    return float(np.linalg.norm(diff))


# ===================================================================== #
#                       WEIGHT PACKING                                  #
# ===================================================================== #

def pack_model_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Convert a PyTorch state_dict to a dict of NumPy arrays (on CPU)."""
    return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}


def unpack_model_weights(
    packed: Dict[str, np.ndarray],
    reference_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Rebuild a PyTorch state_dict from packed NumPy arrays.

    Shapes and dtypes are matched against *reference_state_dict*.
    """
    rebuilt: Dict[str, torch.Tensor] = {}
    for k, ref_tensor in reference_state_dict.items():
        arr = packed[k]
        tensor = torch.from_numpy(arr).to(dtype=ref_tensor.dtype)
        if tensor.shape != ref_tensor.shape:
            raise ValueError(
                f"Shape mismatch for '{k}': packed {tensor.shape} vs "
                f"reference {ref_tensor.shape}"
            )
        rebuilt[k] = tensor
    return rebuilt


def compute_weights_delta(
    w_new: Dict[str, np.ndarray],
    w_old: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Δw = w_new − w_old  (per-layer)."""
    return {k: w_new[k] - w_old[k] for k in w_new}


def apply_weights_delta(
    w_base: Dict[str, np.ndarray],
    delta: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """w_updated = w_base + delta  (per-layer)."""
    return {k: w_base[k] + delta[k] for k in w_base}


# ===================================================================== #
#                          STATISTICS                                   #
# ===================================================================== #

def gini_coefficient(values: List[float]) -> float:
    """Compute the Gini coefficient.

    0 → perfect equality, 1 → maximal inequality.
    Uses the mean-absolute-difference formula:
        G = Σ_i Σ_j |x_i − x_j| / (2 n² μ)
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0
    mu = arr.mean()
    if mu <= 0:
        return 0.0
    abs_diff_sum = float(np.sum(np.abs(arr[:, None] - arr[None, :])))
    return abs_diff_sum / (2.0 * n * n * mu)


def extract_logits(outputs: Any, num_classes: int = 15) -> torch.Tensor:
    """Extract standard classification logits from RT-DETR, YOLO, or dummy models."""
    if isinstance(outputs, dict) and "scores" in outputs:
        # YOLO in train mode: shape [batch, 80, 8400]
        return outputs["scores"].mean(dim=2)[:, :num_classes]
    elif isinstance(outputs, tuple) and len(outputs) == 2 and hasattr(outputs[0], "dim") and outputs[0].dim() == 3:
        # YOLO in eval mode: shape [batch, 84, 8400] where 84 = 4 (bbox) + 80 (scores)
        return outputs[0][:, 4:, :].mean(dim=2)[:, :num_classes]
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 5:
        # RT-DETR in train/eval mode: outputs[3] is shape [batch, 300, 80]
        return outputs[3].mean(dim=1)[:, :num_classes]
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        # General fallback for tuple/list
        item = outputs[0]
        if hasattr(item, "dim") and item.dim() > 1:
            return item[:, :num_classes]
        return item
    else:
        # Dummy model or standard tensor
        return outputs
