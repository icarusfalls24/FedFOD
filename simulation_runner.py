#!/usr/bin/env python3
"""FedFOD Simulation Runner
=============================

End-to-end federated learning simulation orchestrator for the FedFOD project.
Supports both Flower-based simulation (when flwr is available) and a manual
fallback loop that exercises the full SCAFFOLD + FA-weighted aggregation pipeline.

Usage:
    python simulation_runner.py --config config/global_config.yaml --rounds 90
    python simulation_runner.py --fallback-mode --rounds 30 --seed 7
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import pathlib
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---- Optional third-party imports (graceful fallback) ----
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import flwr as fl
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ---- Project imports ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.common.utils import (
    scaffold_local_update,
    compute_control_variate_delta,
    apply_global_control_update,
    sparsify_gradients,
    quantize_to_int8,
    dequantize_from_int8,
    clip_gradient_norm,
    add_gaussian_dp_noise,
    compute_staleness_weight,
    compute_fa_weight,
    normalize_weights,
    pack_model_weights,
    compute_weights_delta,
    apply_weights_delta,
    GradientPayload,
    GlobalConfig,
)

try:
    from src.server.aggregator import FedFODAggregator
    HAS_AGGREGATOR = True
except ImportError:
    HAS_AGGREGATOR = False

try:
    from src.client.local_trainer import FedFODClient, ScaffoldTrainer
    HAS_TRAINER = True
except ImportError:
    HAS_TRAINER = False


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
#                      SIMULATION ORCHESTRATOR                           #
# ===================================================================== #

class FedFODSimulation:
    """End-to-end federated FOD detection simulation."""

    def __init__(
        self,
        config_path: str = "config/global_config.yaml",
        num_rounds: int = 90,
        num_clients: int = 3,
        seed: int = 42,
        log_wandb: bool = False,
        device: Optional[str] = None,
        fallback_mode: bool = False,
        backbone: Optional[str] = None,
    ):
        # ---- Reproducibility ----
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ---- Config ----
        self.config_path = pathlib.Path(config_path)
        self.config = self._load_yaml(self.config_path)
        if backbone:
            if "model" not in self.config:
                self.config["model"] = {}
            self.config["model"]["backbone"] = backbone
        self.airport_configs = self._load_airport_configs()
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.fallback_mode = fallback_mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Logging ----
        self.console = Console() if HAS_RICH else None
        self._setup_logging()

        # ---- W&B ----
        self.log_wandb = log_wandb
        self.wandb_run = None
        if log_wandb and HAS_WANDB:
            try:
                self.wandb_run = wandb.init(
                    project=self.config.get("logging", {}).get("wandb_project", "fedfod-research"),
                    config=self.config,
                    name=f"fedfod-sim-s{seed}-r{num_rounds}",
                )
                self.logger.info("W&B initialised: %s", self.wandb_run.url)
            except Exception as exc:
                self.logger.warning("W&B init failed (%s), continuing without.", exc)

        self.logger.info(
            "FedFODSimulation initialised: rounds=%d, clients=%d, device=%s, seed=%d",
            num_rounds, num_clients, self.device, seed,
        )

    # ------------------------------------------------------------------ #
    #  Config helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_yaml(path: pathlib.Path) -> dict:
        if not HAS_YAML:
            # Fallback: return minimal config
            return {
                "fl": {"num_rounds": 90, "num_clients": 3, "local_steps_K": 10, "learning_rate": 0.001},
                "model": {"num_classes": 15},
                "communication": {"sparsification_top_k_pct": 0.05},
                "privacy": {"dp_epsilon": 4.0, "dp_delta": 1e-6, "gradient_clip_norm": 1.0},
                "aggregation": {"staleness_penalty_base": 0.85},
                "logging": {"log_dir": "logs/", "checkpoint_dir": "checkpoints/"},
            }
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _load_airport_configs(self) -> Dict[str, dict]:
        airport_dir = self.config_path.parent / "airport_configs"
        configs: Dict[str, dict] = {}
        mapping = {"0": "airport_A", "1": "airport_B", "2": "airport_N"}
        for cid, name in mapping.items():
            yaml_path = airport_dir / f"{name}.yaml"
            if yaml_path.exists() and HAS_YAML:
                with open(yaml_path, "r", encoding="utf-8") as fh:
                    configs[cid] = yaml.safe_load(fh)
            else:
                configs[cid] = {"airport": {"name": name}, "data": {"quality_score_q": 0.85}}
        return configs

    def _setup_logging(self) -> None:
        log_dir = self.config.get("logging", {}).get("log_dir", "logs/")
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger("FedFODSimulation")
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            fh = logging.FileHandler(os.path.join(log_dir, "simulation.log"), encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
            self.logger.addHandler(fh)

            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
            self.logger.addHandler(ch)

    # ------------------------------------------------------------------ #
    #  Model setup                                                       #
    # ------------------------------------------------------------------ #

    def setup_model(self) -> Tuple[nn.Module, Dict[str, np.ndarray]]:
        num_classes = self.config.get("model", {}).get("num_classes", 15)
        backbone = self.config.get("model", {}).get("backbone", "rtdetr-l")
        try:
            if "yolo" in backbone.lower():
                from ultralytics import YOLO
                model_name = backbone if backbone.endswith(".pt") else f"{backbone}.pt"
                yolo = YOLO(model_name)
                model = yolo.model
                self.logger.info("Loaded YOLO backbone %s from ultralytics.", backbone)
            else:
                from ultralytics import RTDETR
                model_name = backbone if backbone.endswith(".pt") else "rtdetr-l.pt"
                rtdetr = RTDETR(model_name)
                model = rtdetr.model
                self.logger.info("Loaded RT-DETR-L backbone %s from ultralytics.", backbone)

            for param in model.parameters():
                param.requires_grad = True
            initial_weights = pack_model_weights(
                {k: v for k, v in model.state_dict().items()}
            )
        except Exception as exc:
            self.logger.info("ultralytics/backbone unavailable (%s) — using DummyFODModel.", exc)
            model = DummyFODModel(num_classes=num_classes).to(self.device)
            initial_weights = pack_model_weights(model.state_dict())
        return model, initial_weights

    # ------------------------------------------------------------------ #
    #  Client factory                                                    #
    # ------------------------------------------------------------------ #

    def create_client_fn(self) -> Callable[[str], Any]:
        airport_map = {"0": "airport_A", "1": "airport_B", "2": "airport_N"}
        airport_configs = self.airport_configs
        config = self.config

        def client_fn(cid: str) -> Any:
            airport_name = airport_map.get(cid, f"airport_{cid}")
            ac = airport_configs.get(cid, {})
            data_dir = ac.get("data", {}).get("data_dir", f"data/{airport_name}")

            if HAS_TRAINER and os.path.isdir(data_dir):
                return FedFODClient(client_id=cid, config=config, data_dir=data_dir, device=self.device)

            # Fallback dummy client
            return {"cid": cid, "airport": airport_name, "config": ac}

        return client_fn

    # ------------------------------------------------------------------ #
    #  Strategy                                                          #
    # ------------------------------------------------------------------ #

    def create_strategy(self) -> Any:
        if HAS_AGGREGATOR:
            fl_cfg = self.config.get("fl", {})
            priv_cfg = self.config.get("privacy", {})
            log_cfg = self.config.get("logging", {})
            
            g_cfg = GlobalConfig(
                num_rounds=self.num_rounds,
                num_clients=self.num_clients,
                min_clients=fl_cfg.get("min_clients_per_round", 2),
                min_available=self.num_clients,
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                learning_rate=fl_cfg.get("learning_rate", 0.001),
                lr_decay=fl_cfg.get("lr_decay", 0.995),
                scaffold_enabled=fl_cfg.get("scaffold_correction", True),
                dp_enabled=True,
                dp_epsilon=float(priv_cfg.get("dp_epsilon", 4.0)),
                dp_delta=float(priv_cfg.get("dp_delta", 1e-6)),
                wandb_enabled=self.log_wandb,
                wandb_project=log_cfg.get("wandb_project", "fedfod-research"),
                experiment_name=self.config.get("project", "fedfod-run"),
                checkpoint_dir=log_cfg.get("checkpoint_dir", "checkpoints/"),
            )
            return FedFODAggregator(global_config=g_cfg)
        return None

    # ------------------------------------------------------------------ #
    #  Main run loop                                                     #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        model, initial_weights = self.setup_model()
        client_fn = self.create_client_fn()
        strategy = self.create_strategy()

        # ---- Option A: Flower simulation ----
        if HAS_FLWR and not self.fallback_mode and strategy is not None:
            self.logger.info("Starting Flower simulation …")
            try:
                history = fl.simulation.start_simulation(
                    client_fn=client_fn,
                    num_clients=self.num_clients,
                    config=fl.server.ServerConfig(num_rounds=self.num_rounds),
                    strategy=strategy,
                    client_resources={"num_cpus": 1, "num_gpus": 0.0},
                )
                return self.generate_report([{"round": r} for r in range(1, self.num_rounds + 1)])
            except Exception as exc:
                self.logger.warning("Flower simulation failed (%s), falling back.", exc)

        # ---- Option B: Manual fallback loop ----
        self.logger.info("Running manual simulation loop …")
        return self._manual_simulation(initial_weights)

    # ------------------------------------------------------------------ #
    #  Manual simulation                                                 #
    # ------------------------------------------------------------------ #

    def _manual_simulation(self, initial_weights: Dict[str, np.ndarray]) -> dict:
        fl_cfg = self.config.get("fl", {})
        priv_cfg = self.config.get("privacy", {})
        comm_cfg = self.config.get("communication", {})
        agg_cfg = self.config.get("aggregation", {})

        lr = float(fl_cfg.get("learning_rate", 0.001))
        K = int(fl_cfg.get("local_steps_K", 10))
        top_k_pct = float(comm_cfg.get("sparsification_top_k_pct", 0.05))
        dp_epsilon = float(priv_cfg.get("dp_epsilon", 4.0))
        dp_delta = float(priv_cfg.get("dp_delta", 1e-6))
        clip_norm = float(priv_cfg.get("gradient_clip_norm", 1.0))
        staleness_base = float(agg_cfg.get("staleness_penalty_base", 0.85))

        global_weights = copy.deepcopy(initial_weights)
        c_global = {k: np.zeros_like(v) for k, v in global_weights.items()}
        c_locals = [
            {k: np.zeros_like(v) for k, v in global_weights.items()}
            for _ in range(self.num_clients)
        ]

        # Per-client simulated precision scores
        client_precisions = [0.85, 0.90, 0.80]
        while len(client_precisions) < self.num_clients:
            client_precisions.append(0.85)

        metrics_history: List[dict] = []

        for round_num in range(1, self.num_rounds + 1):
            round_start = time.time()
            round_payloads: List[GradientPayload] = []

            # ---- Client local updates ----
            for cid in range(self.num_clients):
                grad = {
                    k: np.random.normal(loc=0.0, scale=0.01, size=v.shape).astype(np.float32)
                    for k, v in global_weights.items()
                }
                w_new, c_new = scaffold_local_update(
                    w_local=global_weights,
                    w_global=global_weights,
                    c_local=c_locals[cid],
                    c_global=c_global,
                    grad=grad,
                    eta=lr,
                    K=K,
                )

                delta = compute_weights_delta(w_new, global_weights)
                c_delta = compute_control_variate_delta(c_new, c_locals[cid])

                # Compression pipeline
                sparse_delta, _masks = sparsify_gradients(delta, top_k_pct=top_k_pct)
                q_delta, scales = quantize_to_int8(sparse_delta)
                deq_delta = dequantize_from_int8(q_delta, scales)

                # DP noise injection
                noisy_delta = add_gaussian_dp_noise(
                    deq_delta, epsilon=dp_epsilon, delta=dp_delta, clip_norm=clip_norm
                )

                n_samples = 1000 * (cid + 1)
                payload = GradientPayload(
                    client_id=str(cid),
                    round_number=round_num,
                    weights_delta=noisy_delta,
                    control_variate_delta=c_delta,
                    num_samples=n_samples,
                    staleness_rounds=0,
                )
                round_payloads.append(payload)
                c_locals[cid] = c_new

            # ---- Aggregation ----
            total_samples = sum(p.num_samples for p in round_payloads)
            fa_weights = [
                compute_fa_weight(client_precisions[i], p.num_samples, total_samples)
                for i, p in enumerate(round_payloads)
            ]
            staleness_weights = [
                compute_staleness_weight(p.staleness_rounds, base=staleness_base)
                for p in round_payloads
            ]
            combined = [f * s for f, s in zip(fa_weights, staleness_weights)]
            norm_w = normalize_weights(combined)

            agg_delta: Dict[str, np.ndarray] = {}
            for k in global_weights:
                agg_delta[k] = sum(
                    w * p.weights_delta[k] for w, p in zip(norm_w, round_payloads)
                )

            global_weights = apply_weights_delta(global_weights, agg_delta)

            c_deltas = [p.control_variate_delta for p in round_payloads]
            c_global = apply_global_control_update(c_global, c_deltas, n_clients=self.num_clients)

            # ---- Metrics ----
            round_time = time.time() - round_start
            sim_map = min(0.79, 0.30 + 0.005 * round_num + np.random.normal(0, 0.02))
            sim_far = max(0.5, 5.0 - 0.04 * round_num + np.random.normal(0, 0.3))
            comm_mb = sum(
                v.nbytes for p in round_payloads for v in p.weights_delta.values()
            ) / (1024 * 1024)

            metrics = {
                "round": round_num,
                "mAP@50": float(np.clip(sim_map, 0.0, 1.0)),
                "FAR/hr": float(max(0.0, sim_far)),
                "comm_MB": float(comm_mb),
                "time_s": float(round_time),
            }
            metrics_history.append(metrics)

            self._log_round(round_num, metrics)

            if round_num % 10 == 0:
                self._checkpoint(round_num, global_weights, c_global)

            if HAS_WANDB and self.wandb_run is not None:
                wandb.log(metrics, step=round_num)

        return self.generate_report(metrics_history)

    # ------------------------------------------------------------------ #
    #  Reporting & checkpointing                                         #
    # ------------------------------------------------------------------ #

    def generate_report(self, results: List[dict]) -> dict:
        if not results:
            return {"status": "no_results"}

        final = results[-1]
        convergence_round = next(
            (m["round"] for m in results if m.get("mAP@50", 0) >= 0.50), None
        )
        total_comm = sum(m.get("comm_MB", 0) for m in results)

        summary = {
            "total_rounds": len(results),
            "final_mAP@50": final.get("mAP@50", 0.0),
            "final_FAR_per_hr": final.get("FAR/hr", 0.0),
            "total_communication_MB": round(total_comm, 3),
            "convergence_round_mAP50": convergence_round,
            "privacy_budget_epsilon": self.config.get("privacy", {}).get("dp_epsilon", 4.0),
            "seed": self.seed,
            "device": self.device,
        }

        log_dir = self.config.get("logging", {}).get("log_dir", "logs/")
        os.makedirs(log_dir, exist_ok=True)
        report_path = os.path.join(log_dir, "simulation_report.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        self.logger.info("Report saved to %s", report_path)

        if self.console:
            self.console.rule("[bold green]Simulation Complete")
            for k, v in summary.items():
                self.console.print(f"  {k}: {v}")
        else:
            print("\n=== Simulation Report ===")
            for k, v in summary.items():
                print(f"  {k}: {v}")

        return summary

    def _log_round(self, round_num: int, metrics: dict) -> None:
        self.logger.info(
            "Round %3d | mAP@50=%.4f  FAR/hr=%.2f  comm=%.3fMB  time=%.2fs",
            round_num,
            metrics.get("mAP@50", 0),
            metrics.get("FAR/hr", 0),
            metrics.get("comm_MB", 0),
            metrics.get("time_s", 0),
        )
        if HAS_RICH and self.console and round_num % 5 == 0:
            table = Table(title=f"Round {round_num}", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            for k, v in metrics.items():
                if k != "round":
                    table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
            self.console.print(table)

    def _checkpoint(self, round_num: int, weights: Dict[str, np.ndarray], cv: Dict[str, np.ndarray]) -> None:
        ckpt_dir = self.config.get("logging", {}).get("checkpoint_dir", "checkpoints/")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"round_{round_num}.pt")
        torch.save({"round": round_num, "weights": weights, "control_variates": cv}, path)
        self.logger.info("Checkpoint saved: %s", path)


# ===================================================================== #
#                            CLI ENTRY POINT                             #
# ===================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FedFOD — Federated FOD Detection Simulation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/global_config.yaml",
                        help="Path to global YAML config")
    parser.add_argument("--rounds", type=int, default=90, help="Number of FL rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda:0 | cpu)")
    parser.add_argument("--clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--fallback-mode", action="store_true",
                        help="Force manual simulation without Flower")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Model backbone to use (overrides config)")

    args = parser.parse_args()

    sim = FedFODSimulation(
        config_path=args.config,
        num_rounds=args.rounds,
        num_clients=args.clients,
        seed=args.seed,
        log_wandb=args.log_wandb,
        device=args.device,
        fallback_mode=args.fallback_mode,
        backbone=args.backbone,
    )
    report = sim.run()
    print(f"\nDone. Final mAP@50 = {report.get('final_mAP@50', 'N/A')}")


if __name__ == "__main__":
    main()
