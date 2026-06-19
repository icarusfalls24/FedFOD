"""
FedFOD Differential Privacy Utilities
=======================================

Privacy accounting (simple sequential composition and Rényi DP),
noise-multiplier calibration, PSNR-bound validation, and human-readable
privacy reports for the federated learning pipeline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ===================================================================== #
#                     SIMPLE (ε, δ) ACCOUNTANT                          #
# ===================================================================== #

class PrivacyAccountant:
    """Tracks cumulative privacy spend across FL rounds via basic sequential
    composition  (ε_total = Σ ε_i).
    """

    def __init__(
        self,
        target_epsilon: float,
        target_delta: float,
        n_rounds: int,
    ) -> None:
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.n_rounds = n_rounds

        self.epsilon_spent: float = 0.0
        self.per_round_epsilon: List[float] = []
        self._round_idx: int = 0

    # ---- public API ----

    def update(self, epsilon_spent: float) -> float:
        """Record privacy spend for one round and return remaining budget."""
        self._round_idx += 1
        self.per_round_epsilon.append(epsilon_spent)
        self.epsilon_spent += epsilon_spent

        remaining = max(0.0, self.target_epsilon - self.epsilon_spent)
        logger.info(
            "PrivacyAccountant  round %d: ε_spent=%.4f  cumulative=%.4f  remaining=%.4f",
            self._round_idx,
            epsilon_spent,
            self.epsilon_spent,
            remaining,
        )
        return remaining

    def is_budget_exhausted(self) -> bool:
        """True when cumulative spend ≥ target."""
        return self.epsilon_spent >= self.target_epsilon

    def get_report(self) -> Dict:
        """Return a structured dict summarising the privacy state."""
        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "n_rounds": self.n_rounds,
            "rounds_completed": self._round_idx,
            "epsilon_spent": self.epsilon_spent,
            "remaining_epsilon": max(
                0.0, self.target_epsilon - self.epsilon_spent
            ),
            "budget_exhausted": self.is_budget_exhausted(),
            "per_round_epsilon": list(self.per_round_epsilon),
        }


# ===================================================================== #
#                      RÉNYI DP ACCOUNTANT                              #
# ===================================================================== #

class RDPAccountant:
    """Rényi Differential Privacy (RDP) accountant.

    Tracks RDP guarantees across multiple compositions and converts to
    (ε, δ)-DP via the optimal-order conversion.

    Reference:  Mironov (2017), *Rényi Differential Privacy*.
    """

    def __init__(
        self,
        orders: Optional[List[float]] = None,
    ) -> None:
        self.orders: List[float] = list(
            orders or [2, 4, 8, 16, 32, 64, 128, 256]
        )
        # accumulated RDP values — one per order
        self.rdp_accumulated: np.ndarray = np.zeros(
            len(self.orders), dtype=np.float64
        )
        self._n_compositions: int = 0

    # ---- per-round RDP ----

    def compute_rdp_per_round(
        self,
        sigma: float,
        sensitivity: float = 1.0,
        sample_rate: float = 1.0,
    ) -> np.ndarray:
        """Compute RDP guarantee for a single Gaussian mechanism round.

        For the sampled Gaussian mechanism with sampling rate q and noise
        multiplier σ we use the dominant-term bound:

            ε_α ≤  q² α / (2 σ²)       for α > 1

        When q = 1 (full participation) this simplifies to the standard
        Gaussian RDP bound  α / (2 σ²).
        """
        rdp_values = np.zeros(len(self.orders), dtype=np.float64)
        adj_sigma = sigma / sensitivity if sensitivity > 0 else 1e12

        for i, alpha in enumerate(self.orders):
            if sample_rate >= 1.0:
                # full-batch: standard Gaussian RDP
                rdp_values[i] = alpha / (2.0 * adj_sigma ** 2)
            else:
                # sub-sampled Gaussian (dominant-term bound)
                rdp_values[i] = (
                    sample_rate ** 2 * alpha / (2.0 * adj_sigma ** 2)
                )

        return rdp_values

    # ---- accumulation ----

    def accumulate(self, rdp_values: np.ndarray) -> None:
        """Add per-round RDP values to the running total (composition)."""
        if rdp_values.shape != self.rdp_accumulated.shape:
            raise ValueError(
                f"RDP shape mismatch: got {rdp_values.shape}, "
                f"expected {self.rdp_accumulated.shape}"
            )
        self.rdp_accumulated += rdp_values
        self._n_compositions += 1

    # ---- RDP → (ε, δ) conversion ----

    def rdp_to_dp_epsilon(self, target_delta: float) -> float:
        """Convert accumulated RDP to (ε, δ)-DP via the optimal order.

        ε = min_α { ε_α  +  ln(1/δ) / (α − 1) }
        """
        best_epsilon = float("inf")
        log_inv_delta = math.log(1.0 / target_delta)

        for i, alpha in enumerate(self.orders):
            if alpha <= 1.0:
                continue
            eps_candidate = (
                self.rdp_accumulated[i] + log_inv_delta / (alpha - 1.0)
            )
            best_epsilon = min(best_epsilon, eps_candidate)

        return best_epsilon


# ===================================================================== #
#                         PSNR VALIDATION                               #
# ===================================================================== #

def validate_psnr_bound(
    original_gradient: Dict[str, np.ndarray],
    noised_gradient: Dict[str, np.ndarray],
    max_psnr_db: float = 12.0,
) -> bool:
    """Ensure that DP noise is *sufficient* — PSNR must stay **below** the
    threshold (lower PSNR ⇒ more noise ⇒ more privacy).

    PSNR = 10 · log10(MAX² / MSE)

    Returns True if the PSNR is within the acceptable bound.
    """
    mse_sum = 0.0
    n_elements = 0
    max_val = 0.0

    for k in original_gradient:
        orig = original_gradient[k].astype(np.float64)
        nois = noised_gradient[k].astype(np.float64)
        diff = orig - nois
        mse_sum += float(np.sum(diff ** 2))
        n_elements += orig.size
        max_val = max(max_val, float(np.abs(orig).max()))

    if n_elements == 0 or max_val < 1e-12:
        return True  # degenerate case — nothing to protect

    mse = mse_sum / n_elements
    if mse < 1e-30:
        # almost no noise was added ⇒ PSNR is extremely high ⇒ FAIL
        logger.warning("PSNR check failed: MSE ≈ 0  → noise is too small")
        return False

    psnr = 10.0 * math.log10((max_val ** 2) / mse)
    ok = psnr <= max_psnr_db
    if not ok:
        logger.warning(
            "PSNR %.2f dB exceeds bound %.2f dB — noise may be insufficient",
            psnr,
            max_psnr_db,
        )
    else:
        logger.debug("PSNR %.2f dB is within bound (≤ %.2f dB)", psnr, max_psnr_db)
    return ok


# ===================================================================== #
#                        REPORTING                                      #
# ===================================================================== #

def generate_privacy_report(
    accountant: PrivacyAccountant,
    round_number: int,
) -> str:
    """Pretty-print a human-readable privacy status report."""
    report = accountant.get_report()
    lines = [
        "=" * 60,
        f"  Privacy Report — Round {round_number}",
        "=" * 60,
        f"  Target  (ε, δ)  : ({report['target_epsilon']}, {report['target_delta']})",
        f"  Planned rounds  : {report['n_rounds']}",
        f"  Rounds completed: {report['rounds_completed']}",
        f"  ε spent so far  : {report['epsilon_spent']:.6f}",
        f"  ε remaining     : {report['remaining_epsilon']:.6f}",
        f"  Budget exhausted: {report['budget_exhausted']}",
        "-" * 60,
        "  Per-round ε breakdown:",
    ]
    for idx, eps in enumerate(report["per_round_epsilon"], start=1):
        lines.append(f"    Round {idx:4d}:  ε = {eps:.6f}")
    lines.append("=" * 60)
    return "\n".join(lines)


# ===================================================================== #
#                   NOISE-MULTIPLIER CALIBRATION                        #
# ===================================================================== #

def compute_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    n_rounds: int,
    sample_rate: float = 1.0,
) -> float:
    """Binary-search for the Gaussian noise multiplier σ such that
    *n_rounds* compositions yield the desired (ε, δ)-DP via the RDP
    accountant.

    Returns the calibrated σ.
    """
    sigma_lo = 0.01
    sigma_hi = 500.0
    tol = 1e-4
    max_iter = 200

    for _ in range(max_iter):
        sigma_mid = (sigma_lo + sigma_hi) / 2.0

        acc = RDPAccountant()
        rdp_round = acc.compute_rdp_per_round(
            sigma=sigma_mid, sensitivity=1.0, sample_rate=sample_rate
        )
        for _ in range(n_rounds):
            acc.accumulate(rdp_round)
        eps = acc.rdp_to_dp_epsilon(target_delta)

        if eps < target_epsilon:
            # noise is more than enough — try less noise
            sigma_hi = sigma_mid
        else:
            sigma_lo = sigma_mid

        if sigma_hi - sigma_lo < tol:
            break

    calibrated = (sigma_lo + sigma_hi) / 2.0
    logger.info(
        "Calibrated σ = %.4f for target (ε=%.2f, δ=%.1e) over %d rounds "
        "(q=%.3f)",
        calibrated,
        target_epsilon,
        target_delta,
        n_rounds,
        sample_rate,
    )
    return calibrated
