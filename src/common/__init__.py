"""
FedFOD Common Utilities
========================

Exports all shared utilities, dataclasses, privacy tools, and
protobuf-compatible message stubs used across the federated learning pipeline.
"""

from src.common.utils import (
    # --- Dataclasses ---
    GradientPayload,
    PrototypePayload,
    GlobalUpdate,
    # --- SCAFFOLD helpers ---
    scaffold_local_update,
    compute_control_variate_delta,
    apply_global_control_update,
    # --- Compression ---
    sparsify_gradients,
    quantize_to_int8,
    dequantize_from_int8,
    # --- Payload validation ---
    compute_payload_size_bytes,
    validate_payload_size,
    # --- DP / clipping ---
    add_gaussian_dp_noise,
    clip_gradient_norm,
    # --- Federated weighting ---
    compute_staleness_weight,
    compute_fa_weight,
    normalize_weights,
    # --- Similarity / distance ---
    cosine_similarity,
    euclidean_distance,
    # --- Weight packing ---
    pack_model_weights,
    unpack_model_weights,
    compute_weights_delta,
    apply_weights_delta,
    # --- Statistics ---
    gini_coefficient,
)

from src.common.dp_utils import (
    PrivacyAccountant,
    RDPAccountant,
    validate_psnr_bound,
    generate_privacy_report,
    compute_noise_multiplier,
)

from src.common.protocols_pb2 import (
    GradientMessage,
    GlobalUpdateMessage,
    PrototypeMessage,
    ClientRegistration,
    HeartbeatMessage,
)

__all__ = [
    # utils dataclasses
    "GradientPayload",
    "PrototypePayload",
    "GlobalUpdate",
    # SCAFFOLD
    "scaffold_local_update",
    "compute_control_variate_delta",
    "apply_global_control_update",
    # compression
    "sparsify_gradients",
    "quantize_to_int8",
    "dequantize_from_int8",
    # payload
    "compute_payload_size_bytes",
    "validate_payload_size",
    # DP
    "add_gaussian_dp_noise",
    "clip_gradient_norm",
    # weighting
    "compute_staleness_weight",
    "compute_fa_weight",
    "normalize_weights",
    # similarity
    "cosine_similarity",
    "euclidean_distance",
    # weight packing
    "pack_model_weights",
    "unpack_model_weights",
    "compute_weights_delta",
    "apply_weights_delta",
    # stats
    "gini_coefficient",
    # privacy
    "PrivacyAccountant",
    "RDPAccountant",
    "validate_psnr_bound",
    "generate_privacy_report",
    "compute_noise_multiplier",
    # protocols
    "GradientMessage",
    "GlobalUpdateMessage",
    "PrototypeMessage",
    "ClientRegistration",
    "HeartbeatMessage",
]
