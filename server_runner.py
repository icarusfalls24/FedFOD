#!/usr/bin/env python3
"""FedFOD Server Runner — Live gRPC Deployment
================================================

Starts the Flower federated learning server with the FedFODAggregator
strategy over a real gRPC transport.  Clients connect from separate
processes (or machines) via ``client_runner.py``.

Usage:
    python server_runner.py --config config/global_config.yaml
    python server_runner.py --config config/global_config.yaml --port 8080 --rounds 30
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ---- Optional imports ----
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

import flwr as fl
from flwr.common import ndarrays_to_parameters

# ---- Project imports ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.common.utils import GlobalConfig, pack_model_weights
from src.server.aggregator import FedFODAggregator

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("fedfod.server_runner")


# ===================================================================== #
#                         DUMMY MODEL                                    #
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
#                         HELPERS                                        #
# ===================================================================== #

def load_yaml(path: pathlib.Path) -> dict:
    """Load YAML config; return minimal fallback if pyyaml is missing."""
    if not HAS_YAML:
        return {
            "fl": {"num_rounds": 90, "num_clients": 3, "local_steps_K": 10,
                   "learning_rate": 0.001, "min_clients_per_round": 2,
                   "scaffold_correction": True},
            "model": {"num_classes": 15},
            "communication": {"grpc_port": 50051},
            "privacy": {"dp_epsilon": 4.0, "dp_delta": 1e-6,
                        "gradient_clip_norm": 1.0},
            "aggregation": {"staleness_penalty_base": 0.85},
            "logging": {"log_dir": "logs/", "checkpoint_dir": "checkpoints/",
                        "wandb_project": "fedfod-research"},
        }
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_global_config(raw: dict, num_rounds: int, min_clients: int,
                        log_wandb: bool) -> GlobalConfig:
    """Map a raw YAML dict to a GlobalConfig dataclass."""
    fl_cfg = raw.get("fl", {})
    priv_cfg = raw.get("privacy", {})
    log_cfg = raw.get("logging", {})

    return GlobalConfig(
        num_rounds=num_rounds,
        num_clients=fl_cfg.get("num_clients", 3),
        min_clients=min_clients,
        min_available=min_clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        learning_rate=fl_cfg.get("learning_rate", 0.001),
        lr_decay=fl_cfg.get("lr_decay", 0.995),
        scaffold_enabled=fl_cfg.get("scaffold_correction", True),
        dp_enabled=True,
        dp_epsilon=float(priv_cfg.get("dp_epsilon", 4.0)),
        dp_delta=float(priv_cfg.get("dp_delta", 1e-6)),
        wandb_enabled=log_wandb,
        wandb_project=log_cfg.get("wandb_project", "fedfod-research"),
        experiment_name=raw.get("project", "fedfod-live"),
        checkpoint_dir=log_cfg.get("checkpoint_dir", "checkpoints/"),
    )


def load_initial_parameters(num_classes: int = 15, use_dummy: bool = False, backbone: str = "rtdetr-l"):
    """Load RT-DETR-L or YOLO weights or fall back to DummyFODModel."""
    if not use_dummy:
        try:
            if "yolo" in backbone.lower():
                from ultralytics import YOLO
                model_name = backbone if backbone.endswith(".pt") else f"{backbone}.pt"
                model = YOLO(model_name)
                state_dict = model.model.state_dict()
            else:
                from ultralytics import RTDETR
                model_name = backbone if backbone.endswith(".pt") else "rtdetr-l.pt"
                model = RTDETR(model_name)
                state_dict = model.model.state_dict()

            sorted_keys = sorted(state_dict.keys())
            ndarrays = [state_dict[k].detach().cpu().numpy() for k in sorted_keys]
            logger.info("Loaded %s initial parameters (%d tensors).",
                         backbone, len(ndarrays))
            return ndarrays_to_parameters(ndarrays)
        except Exception as exc:
            logger.info("%s unavailable (%s), using DummyFODModel.", backbone, exc)

    model = DummyFODModel(num_classes=num_classes)
    state_dict = model.state_dict()
    sorted_keys = sorted(state_dict.keys())
    ndarrays = [state_dict[k].detach().cpu().numpy() for k in sorted_keys]
    logger.info("Using DummyFODModel initial parameters (%d tensors).", len(ndarrays))
    return ndarrays_to_parameters(ndarrays)


# ===================================================================== #
#                         MAIN                                           #
# ===================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FedFOD — Federated Learning Server (gRPC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/global_config.yaml",
                        help="Path to global YAML config")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server bind address")
    parser.add_argument("--port", type=int, default=8080,
                        help="Server port (Flower default is 8080)")
    parser.add_argument("--rounds", type=int, default=90,
                        help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=2,
                        help="Minimum clients required per round")
    parser.add_argument("--log-wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dummy-model", action="store_true",
                        help="Use DummyFODModel instead of RT-DETR-L (for testing)")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Model backbone to use (overrides config)")

    args = parser.parse_args()

    # ---- Reproducibility ----
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Config ----
    raw_config = load_yaml(pathlib.Path(args.config))
    num_classes = raw_config.get("model", {}).get("num_classes", 15)
    backbone = args.backbone or raw_config.get("model", {}).get("backbone", "rtdetr-l")

    g_cfg = build_global_config(
        raw_config,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        log_wandb=args.log_wandb,
    )

    # ---- Initial parameters ----
    initial_params = load_initial_parameters(
        num_classes=num_classes, use_dummy=args.dummy_model, backbone=backbone,
    )

    # ---- Strategy ----
    strategy = FedFODAggregator(
        global_config=g_cfg,
        initial_parameters=initial_params,
    )

    # ---- Logging setup ----
    log_dir = raw_config.get("logging", {}).get("log_dir", "logs/")
    os.makedirs(log_dir, exist_ok=True)

    server_address = f"{args.host}:{args.port}"
    logger.info("=" * 60)
    logger.info("FedFOD Server — Live gRPC Deployment")
    logger.info("=" * 60)
    logger.info("  Address:      %s", server_address)
    logger.info("  Rounds:       %d", args.rounds)
    logger.info("  Min clients:  %d", args.min_clients)
    logger.info("  SCAFFOLD:     %s", g_cfg.scaffold_enabled)
    logger.info("  DP ε:         %.2f", g_cfg.dp_epsilon)
    logger.info("  Seed:         %d", args.seed)
    logger.info("=" * 60)

    # ---- Start server ----
    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 512,  # 512 MB max message
    )

    # ---- Report ----
    logger.info("Server completed %d rounds.", args.rounds)
    if history.losses_distributed:
        final_loss = history.losses_distributed[-1]
        logger.info("Final distributed loss: round=%d, loss=%.4f",
                     final_loss[0], final_loss[1])
    if history.metrics_distributed:
        logger.info("Final distributed metrics: %s",
                     {k: v[-1] for k, v in history.metrics_distributed.items()})

    logger.info("Server shutting down.")


if __name__ == "__main__":
    main()
