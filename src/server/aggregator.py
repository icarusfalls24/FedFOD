"""
FedFOD Aggregator — FA-weighted SCAFFOLD Strategy
==================================================
Flower-compatible (v1.7 legacy API) federated aggregation strategy that
combines:
  * Federated Analytics (FA) precision-weighted averaging
  * SCAFFOLD control-variate bias correction
  * Staleness-aware discounting for asynchronous stragglers
  * Gini-coefficient fairness validation (< 0.35)

Usage:
    strategy = FedFODAggregator(global_config=cfg)
    fl.server.start_server(strategy=strategy, ...)
"""

from __future__ import annotations

import base64
import io
import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.common.utils import (
    AggregationResult,
    ClientMetrics,
    GlobalConfig,
    encode_ndarray_b64,
    decode_ndarray_b64,
    gini_coefficient,
    get_lr_for_round,
    staleness_weight,
    try_import_wandb,
    STALENESS_DECAY,
    MAX_GINI_COEFFICIENT,
)

logger = logging.getLogger("fedfod.aggregator")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def encode_numpy_to_config(name: str, arr: np.ndarray) -> Dict[str, str]:
    """Serialize a numpy array into a config-dict entry via base64 encoding.

    Args:
        name: Key name for the config entry.
        arr: Numpy array to serialize.

    Returns:
        Dict with a single entry  ``{name: base64_string}``.
    """
    return {name: encode_ndarray_b64(arr)}


def decode_numpy_from_config(config: Dict[str, Any], name: str) -> np.ndarray:
    """Deserialize a numpy array from a config-dict entry.

    Args:
        config: Configuration dict received from a client.
        name: Key name to look up.

    Returns:
        Decoded numpy array.

    Raises:
        KeyError: If *name* is not present in *config*.
    """
    if name not in config:
        raise KeyError(f"Expected key '{name}' in config dict, got keys: {list(config.keys())}")
    return decode_ndarray_b64(config[name])


def fa_weighted_aggregate(
    weight_deltas: List[List[np.ndarray]],
    effective_weights: List[float],
) -> List[np.ndarray]:
    """Compute FA-weighted average of model weight deltas.

    Each client's delta is weighted by its effective weight
    (precision × sample fraction × staleness discount), then the weighted
    sum is normalised so the weights sum to 1.

    Args:
        weight_deltas: Per-client list of parameter delta arrays.
        effective_weights: Per-client scalar effective weights.

    Returns:
        Aggregated parameter deltas as a list of numpy arrays.
    """
    if not weight_deltas:
        raise ValueError("No weight deltas to aggregate.")
    if len(weight_deltas) != len(effective_weights):
        raise ValueError(
            f"Mismatched lengths: {len(weight_deltas)} deltas vs "
            f"{len(effective_weights)} weights."
        )

    total_weight = sum(effective_weights)
    if total_weight < 1e-12:
        logger.warning("Total effective weight ≈ 0; falling back to uniform averaging.")
        n = len(weight_deltas)
        effective_weights = [1.0 / n] * n
        total_weight = 1.0

    normalised = [w / total_weight for w in effective_weights]

    # Initialise aggregated arrays with zeros matching the first client's shapes (using float64 to prevent casting errors)
    aggregated = [np.zeros_like(arr, dtype=np.float64) for arr in weight_deltas[0]]

    for client_idx, (deltas, norm_w) in enumerate(zip(weight_deltas, normalised)):
        for layer_idx, delta in enumerate(deltas):
            aggregated[layer_idx] += norm_w * delta.astype(np.float64)

    # Cast back to original dtypes
    aggregated = [
        agg.astype(weight_deltas[0][layer_idx].dtype)
        for layer_idx, agg in enumerate(aggregated)
    ]
    return aggregated


# ---------------------------------------------------------------------------
# FedFODAggregator
# ---------------------------------------------------------------------------


class FedFODAggregator(FedAvg):
    """Flower strategy implementing FA-weighted SCAFFOLD aggregation.

    This strategy extends ``FedAvg`` from Flower v1.7 (legacy API) with:

    1. **Precision-weighted aggregation** — each client's contribution is
       scaled by its local detection precision metric ``precision_q``.
    2. **Sample-proportional weighting** — standard FL proportional-to-data
       weighting ``n_i / N``.
    3. **Staleness discounting** — exponential decay for delayed updates.
    4. **SCAFFOLD bias correction** — global and per-client control variates
       to reduce client-drift in heterogeneous settings.
    5. **Gini fairness gate** — aggregation proceeds only when the Gini
       coefficient of effective weights is below a configurable threshold.
    """

    def __init__(
        self,
        global_config: GlobalConfig,
        initial_parameters: Optional[Parameters] = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the FedFOD aggregation strategy.

        Args:
            global_config: Experiment-wide configuration dataclass.
            initial_parameters: Optional pre-loaded initial parameters.
            **kwargs: Forwarded to ``FedAvg.__init__``.
        """
        super().__init__(
            fraction_fit=global_config.fraction_fit,
            fraction_evaluate=global_config.fraction_evaluate,
            min_fit_clients=global_config.min_clients,
            min_evaluate_clients=max(1, global_config.min_clients // 2),
            min_available_clients=global_config.min_available,
            initial_parameters=initial_parameters,
            **kwargs,
        )

        self.global_config = global_config

        # SCAFFOLD global control variate (initialised lazily on first aggregate)
        self._global_cv: Optional[List[np.ndarray]] = None

        # Current global model parameters (as ndarrays)
        self._global_weights: Optional[List[np.ndarray]] = None

        # Per-round metric history
        self._round_metrics: OrderedDict[int, AggregationResult] = OrderedDict()

        # Optional W&B
        self._wandb = None
        if global_config.wandb_enabled:
            self._wandb = try_import_wandb()
            if self._wandb is not None:
                self._wandb.init(
                    project=global_config.wandb_project,
                    name=global_config.experiment_name,
                    config={
                        "num_rounds": global_config.num_rounds,
                        "lr": global_config.learning_rate,
                        "scaffold": global_config.scaffold_enabled,
                        "dp_epsilon": global_config.dp_epsilon if global_config.dp_enabled else None,
                    },
                )

        logger.info(
            "FedFODAggregator initialised — SCAFFOLD=%s, rounds=%d, lr=%.2e",
            global_config.scaffold_enabled,
            global_config.num_rounds,
            global_config.learning_rate,
        )

    # ------------------------------------------------------------------
    # Flower overrides
    # ------------------------------------------------------------------

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Load initial RT-DETR-L weights or return pre-set parameters.

        If ``initial_parameters`` were provided at construction, those are
        used directly.  Otherwise we attempt to load a local RT-DETR-L
        checkpoint from the configured checkpoint directory.

        Args:
            client_manager: Flower client manager (unused but required by API).

        Returns:
            Flower ``Parameters`` object, or ``None`` to let clients init.
        """
        if self.initial_parameters is not None:
            logger.info("Using pre-supplied initial parameters.")
            self._global_weights = parameters_to_ndarrays(self.initial_parameters)
            return self.initial_parameters

        # Attempt to load RT-DETR-L checkpoint
        try:
            import torch
            from pathlib import Path

            ckpt_dir = Path(self.global_config.checkpoint_dir)
            ckpt_path = ckpt_dir / "rtdetr_l_initial.pth"

            if ckpt_path.exists():
                state_dict = torch.load(str(ckpt_path), map_location="cpu")
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                sorted_keys = sorted(state_dict.keys())
                ndarrays = [state_dict[k].cpu().numpy() for k in sorted_keys]
                self._global_weights = ndarrays
                params = ndarrays_to_parameters(ndarrays)
                logger.info(
                    "Loaded RT-DETR-L initial weights from %s (%d tensors).",
                    ckpt_path,
                    len(ndarrays),
                )
                return params
            else:
                logger.warning(
                    "RT-DETR-L checkpoint not found at %s; "
                    "clients will initialise their own weights.",
                    ckpt_path,
                )
                return None

        except ImportError:
            logger.warning(
                "PyTorch not available; cannot load RT-DETR-L checkpoint. "
                "Clients will initialise their own weights."
            )
            return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Prepare fit instructions for each selected client.

        Packs into the config dict:
          * ``server_round`` — current round number
          * ``learning_rate`` — scheduled LR for this round
          * ``scaffold_enabled`` — whether clients should apply CV correction
          * ``global_cv_<i>`` — base64-encoded global control variate layers
            (only when SCAFFOLD is enabled and CVs are initialised)

        Args:
            server_round: Current server round (1-indexed).
            parameters: Current global model parameters.
            client_manager: Flower client manager.

        Returns:
            List of (ClientProxy, FitIns) tuples.
        """
        config: Dict[str, Scalar] = {
            "server_round": server_round,
            "learning_rate": get_lr_for_round(self.global_config, server_round),
            "scaffold_enabled": self.global_config.scaffold_enabled,
            "num_rounds": self.global_config.num_rounds,
        }

        # Pack global control variates
        if self.global_config.scaffold_enabled and self._global_cv is not None:
            config["global_cv_count"] = len(self._global_cv)
            for i, cv_arr in enumerate(self._global_cv):
                config[f"global_cv_{i}"] = encode_ndarray_b64(cv_arr)

        fit_ins = FitIns(parameters, config)

        # Sample clients using the parent FedAvg sampling logic
        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num,
        )

        logger.info(
            "Round %d — configure_fit: %d clients, lr=%.2e",
            server_round,
            len(clients),
            config["learning_rate"],
        )

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client fit results using FA-weighted SCAFFOLD averaging.

        Pipeline:
          1. Extract per-client weight deltas and CV deltas from ``FitRes.metrics``
          2. Compute effective weights: ``q_i = precision_i × (n_i/N) × staleness_w``
          3. Validate Gini coefficient < threshold
          4. FA-weighted average of weight deltas
          5. Update global control variates
          6. Apply aggregated delta to global model

        Args:
            server_round: Current server round.
            results: Successful client fit results.
            failures: Failed client fit results.

        Returns:
            Tuple of (aggregated Parameters, aggregation metrics dict).
        """
        if not results:
            logger.error("Round %d — no successful results to aggregate.", server_round)
            return None, {}

        if failures:
            logger.warning(
                "Round %d — %d client failures (proceeding with %d successes).",
                server_round,
                len(failures),
                len(results),
            )

        # ----- Step 1: Extract data from results -----
        client_weight_deltas: List[List[np.ndarray]] = []
        client_cv_deltas: List[List[np.ndarray]] = []
        client_metrics_list: List[ClientMetrics] = []

        for client_proxy, fit_res in results:
            metrics = fit_res.metrics if fit_res.metrics else {}

            # Parse client metrics
            cm = ClientMetrics(
                client_id=metrics.get("client_id", client_proxy.cid),
                num_samples=fit_res.num_examples,
                precision_q=float(metrics.get("precision_q", 1.0)),
                staleness=int(metrics.get("staleness", 0)),
                train_loss=float(metrics.get("train_loss", 0.0)),
                mAP50=float(metrics.get("mAP50", 0.0)),
                mAP50_95=float(metrics.get("mAP50_95", 0.0)),
                novel_detections=int(metrics.get("novel_detections", 0)),
                round_number=server_round,
            )
            client_metrics_list.append(cm)

            # Extract weight deltas (transmitted as parameters)
            weight_delta = parameters_to_ndarrays(fit_res.parameters)
            client_weight_deltas.append(weight_delta)

            # Extract CV deltas if SCAFFOLD is active
            if self.global_config.scaffold_enabled:
                cv_count = int(metrics.get("cv_delta_count", 0))
                if cv_count > 0:
                    cv_delta = []
                    for i in range(cv_count):
                        cv_arr = decode_ndarray_b64(metrics[f"cv_delta_{i}"])
                        cv_delta.append(cv_arr)
                    client_cv_deltas.append(cv_delta)

        # ----- Step 2: Compute effective weights -----
        effective_weights = self._compute_effective_weights(
            client_metrics_list
        )

        # ----- Step 3: Gini validation -----
        gini_valid = self._validate_gini(effective_weights)
        current_gini = gini_coefficient(effective_weights)

        if not gini_valid:
            logger.warning(
                "Round %d — Gini coefficient %.4f exceeds threshold %.2f. "
                "Clipping weights to uniform to reduce dominance.",
                server_round,
                current_gini,
                MAX_GINI_COEFFICIENT,
            )
            # Fallback: blend toward uniform weights
            n = len(effective_weights)
            uniform = [1.0 / n] * n
            blend = 0.5  # 50% uniform, 50% original
            effective_weights = [
                blend * u + (1 - blend) * w
                for u, w in zip(uniform, effective_weights)
            ]

        # ----- Step 4: FA-weighted aggregation -----
        aggregated_delta = fa_weighted_aggregate(
            client_weight_deltas, effective_weights
        )

        # Apply delta to global weights
        if self._global_weights is None:
            # First round — treat deltas as absolute weights
            self._global_weights = aggregated_delta
        else:
            self._global_weights = [
                gw + ad.astype(gw.dtype)
                for gw, ad in zip(self._global_weights, aggregated_delta)
            ]

        # ----- Step 5: Update global control variates -----
        if self.global_config.scaffold_enabled and client_cv_deltas:
            n_clients = len(client_cv_deltas)

            if self._global_cv is None:
                # Initialise global CV to zeros matching the first client's CV shape
                self._global_cv = [
                    np.zeros_like(cv) for cv in client_cv_deltas[0]
                ]

            for layer_idx in range(len(self._global_cv)):
                cv_sum = np.zeros_like(self._global_cv[layer_idx], dtype=np.float64)
                for client_cv in client_cv_deltas:
                    if layer_idx < len(client_cv):
                        cv_sum += client_cv[layer_idx].astype(np.float64)
                self._global_cv[layer_idx] = (
                    self._global_cv[layer_idx] + cv_sum / n_clients
                ).astype(self._global_cv[layer_idx].dtype)

        # ----- Step 6: Build result and metrics -----
        aggregated_params = ndarrays_to_parameters(self._global_weights)

        # Aggregate metrics
        avg_loss = float(np.mean([cm.train_loss for cm in client_metrics_list]))
        avg_mAP50 = float(np.mean([cm.mAP50 for cm in client_metrics_list]))
        avg_mAP50_95 = float(np.mean([cm.mAP50_95 for cm in client_metrics_list]))
        total_novel = sum(cm.novel_detections for cm in client_metrics_list)

        round_result = AggregationResult(
            round_number=server_round,
            num_clients=len(results),
            effective_weights=effective_weights,
            gini_coefficient=current_gini,
            gini_valid=gini_valid,
            global_loss=avg_loss,
            global_mAP50=avg_mAP50,
            global_mAP50_95=avg_mAP50_95,
        )

        self._log_round_metrics(server_round, round_result)
        self._round_metrics[server_round] = round_result

        metrics_dict: Dict[str, Scalar] = {
            "train_loss": avg_loss,
            "mAP50": avg_mAP50,
            "mAP50_95": avg_mAP50_95,
            "gini": current_gini,
            "gini_valid": int(gini_valid),
            "num_clients": len(results),
            "novel_detections": total_novel,
        }

        return aggregated_params, metrics_dict

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, Any]]:
        """Configure evaluation instructions for selected clients.

        Args:
            server_round: Current server round.
            parameters: Current global model parameters.
            client_manager: Flower client manager.

        Returns:
            List of (ClientProxy, EvaluateIns) tuples.
        """
        from flwr.common import EvaluateIns

        config: Dict[str, Scalar] = {
            "server_round": server_round,
        }

        eval_ins = EvaluateIns(parameters, config)

        sample_size, min_num = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num,
        )

        logger.info(
            "Round %d — configure_evaluate: %d clients selected.",
            server_round,
            len(clients),
        )

        return [(client, eval_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Any]],
        failures: List[Union[Tuple[ClientProxy, Any], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients.

        Computes weighted-average loss and metrics across all reporting
        clients.

        Args:
            server_round: Current server round.
            results: Successful evaluation results.
            failures: Failed evaluation results.

        Returns:
            Tuple of (weighted loss, metrics dict).
        """
        if not results:
            logger.warning("Round %d — no evaluation results.", server_round)
            return None, {}

        total_examples = 0
        weighted_loss = 0.0
        mAP50_sum = 0.0
        mAP50_95_sum = 0.0

        for client_proxy, eval_res in results:
            n = eval_res.num_examples
            total_examples += n
            weighted_loss += eval_res.loss * n

            metrics = eval_res.metrics if eval_res.metrics else {}
            mAP50_sum += float(metrics.get("mAP50", 0.0)) * n
            mAP50_95_sum += float(metrics.get("mAP50_95", 0.0)) * n

        if total_examples == 0:
            return None, {}

        avg_loss = weighted_loss / total_examples
        avg_mAP50 = mAP50_sum / total_examples
        avg_mAP50_95 = mAP50_95_sum / total_examples

        metrics_dict: Dict[str, Scalar] = {
            "eval_loss": avg_loss,
            "eval_mAP50": avg_mAP50,
            "eval_mAP50_95": avg_mAP50_95,
            "eval_num_clients": len(results),
        }

        logger.info(
            "Round %d — evaluate: loss=%.4f, mAP50=%.4f, mAP50-95=%.4f (%d clients)",
            server_round,
            avg_loss,
            avg_mAP50,
            avg_mAP50_95,
            len(results),
        )

        # W&B logging
        if self._wandb is not None:
            self._wandb.log(
                {
                    "eval/loss": avg_loss,
                    "eval/mAP50": avg_mAP50,
                    "eval/mAP50_95": avg_mAP50_95,
                    "round": server_round,
                },
                step=server_round,
            )

        return avg_loss, metrics_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_effective_weights(
        self, client_metrics: List[ClientMetrics]
    ) -> List[float]:
        """Compute effective aggregation weight for each client.

        Formula:
            q_i = precision_i × (n_i / N_total) × staleness_decay^staleness_i

        Args:
            client_metrics: Per-client metrics.

        Returns:
            List of effective weights (unnormalised).
        """
        total_samples = sum(cm.num_samples for cm in client_metrics)
        if total_samples == 0:
            n = len(client_metrics)
            return [1.0 / n] * n

        weights = []
        for cm in client_metrics:
            sample_fraction = cm.num_samples / total_samples
            stale_w = staleness_weight(cm.staleness, STALENESS_DECAY)
            q_i = cm.precision_q * sample_fraction * stale_w
            weights.append(q_i)

        return weights

    def _validate_gini(self, weights: List[float]) -> bool:
        """Check whether the Gini coefficient of the weights is acceptable.

        Args:
            weights: Effective aggregation weights.

        Returns:
            True if Gini < MAX_GINI_COEFFICIENT.
        """
        g = gini_coefficient(weights)
        return g < MAX_GINI_COEFFICIENT

    def _log_round_metrics(
        self, server_round: int, result: AggregationResult
    ) -> None:
        """Log round metrics to console and optionally to W&B.

        Args:
            server_round: Current round number.
            result: Aggregation result for this round.
        """
        logger.info(
            "Round %d — aggregated: %d clients, loss=%.4f, "
            "mAP50=%.4f, mAP50-95=%.4f, Gini=%.4f (valid=%s)",
            server_round,
            result.num_clients,
            result.global_loss,
            result.global_mAP50,
            result.global_mAP50_95,
            result.gini_coefficient,
            result.gini_valid,
        )

        if self._wandb is not None:
            self._wandb.log(
                {
                    "train/loss": result.global_loss,
                    "train/mAP50": result.global_mAP50,
                    "train/mAP50_95": result.global_mAP50_95,
                    "train/gini": result.gini_coefficient,
                    "train/num_clients": result.num_clients,
                    "round": server_round,
                },
                step=server_round,
            )

    # ------------------------------------------------------------------
    # Public utilities
    # ------------------------------------------------------------------

    def get_round_metrics(self, server_round: int) -> Optional[AggregationResult]:
        """Retrieve stored metrics for a specific round.

        Args:
            server_round: Round number to query.

        Returns:
            AggregationResult or None if the round hasn't been recorded.
        """
        return self._round_metrics.get(server_round)

    def get_all_round_metrics(self) -> Dict[int, AggregationResult]:
        """Return metrics for all completed rounds.

        Returns:
            OrderedDict mapping round number → AggregationResult.
        """
        return dict(self._round_metrics)

    def get_global_control_variates(self) -> Optional[List[np.ndarray]]:
        """Return a copy of the current global control variates.

        Returns:
            List of numpy arrays, or None if not yet initialised.
        """
        if self._global_cv is None:
            return None
        return [cv.copy() for cv in self._global_cv]
