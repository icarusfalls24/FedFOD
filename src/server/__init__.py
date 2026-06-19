"""
FedFOD Server Module
====================
Flower-compatible federated learning server components for FOD detection.

Exports:
    - FedFODAggregator:     FA-weighted SCAFFOLD aggregation strategy
    - ShamirSecretSharing:  Shamir (k, n) threshold secret sharing
    - SMPCNode:             Secure MPC computation node
    - SMPCGradientProtector: Gradient-level SMPC orchestrator
    - PrototypeMemoryBank:  Open-world CLIP prototype memory

Imports are lazy to avoid hard dependency on flwr at module load time.
"""


def __getattr__(name):
    """Lazy import to avoid ModuleNotFoundError when flwr is not installed."""
    if name == "FedFODAggregator":
        from src.server.aggregator import FedFODAggregator
        return FedFODAggregator
    if name == "ShamirSecretSharing":
        from src.server.smpc_node import ShamirSecretSharing
        return ShamirSecretSharing
    if name == "SMPCNode":
        from src.server.smpc_node import SMPCNode
        return SMPCNode
    if name == "SMPCGradientProtector":
        from src.server.smpc_node import SMPCGradientProtector
        return SMPCGradientProtector
    if name == "PrototypeMemoryBank":
        from src.server.prototype_memory import PrototypeMemoryBank
        return PrototypeMemoryBank
    if name == "PrototypeEntry":
        from src.server.prototype_memory import PrototypeEntry
        return PrototypeEntry
    raise AttributeError(f"module 'src.server' has no attribute {name!r}")


__all__ = [
    "FedFODAggregator",
    "ShamirSecretSharing",
    "SMPCNode",
    "SMPCGradientProtector",
    "PrototypeMemoryBank",
    "PrototypeEntry",
]
