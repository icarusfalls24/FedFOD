"""
FedFOD SMPC Node — Shamir Secret Sharing for Gradient Privacy
==============================================================
Implements (k, n)-threshold Shamir Secret Sharing to protect gradient
updates transmitted between FL clients and the aggregation server.

Architecture:
  * **ShamirSecretSharing** — core maths: polynomial generation, share
    splitting, Lagrange-interpolation reconstruction.
  * **SMPCNode** — one of three trusted computation nodes (ICAO,
    Consortium, NeutralParty) that holds gradient shares.
  * **SMPCGradientProtector** — orchestrator that splits/reconstructs
    full gradient dictionaries across the three nodes.
"""

from __future__ import annotations

import logging
import math
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.utils import SHAMIR_SAFE_PRIME, SMPCShare

logger = logging.getLogger("fedfod.smpc")

# ---------------------------------------------------------------------------
# ShamirSecretSharing
# ---------------------------------------------------------------------------


class ShamirSecretSharing:
    """Shamir (k, n)-threshold secret sharing over a prime field.

    Secrets (non-negative integers) are split into *n* shares such that any
    *k* shares suffice to reconstruct the original value, while fewer than
    *k* shares reveal nothing.

    Args:
        n_shares: Total number of shares to generate (n).
        k_threshold: Minimum shares required for reconstruction (k).
        prime: Prime modulus for the finite field.  Defaults to 2^61 − 1
               (a Mersenne prime large enough for 64-bit float precision).
    """

    def __init__(
        self,
        n_shares: int = 3,
        k_threshold: int = 2,
        prime: Optional[int] = None,
    ) -> None:
        if k_threshold > n_shares:
            raise ValueError(
                f"Threshold k={k_threshold} must be ≤ n_shares={n_shares}."
            )
        if k_threshold < 2:
            raise ValueError("Threshold must be at least 2.")

        self.n_shares = n_shares
        self.k_threshold = k_threshold
        self.prime = prime if prime is not None else SHAMIR_SAFE_PRIME

    # ----- Core polynomial ops -----

    def _generate_polynomial(self, secret: int, degree: int) -> List[int]:
        """Generate a random polynomial of the given degree with ``coeffs[0] = secret``.

        Args:
            secret: The free term (the secret to hide).
            degree: Polynomial degree (= k_threshold − 1).

        Returns:
            List of coefficients ``[secret, a_1, ..., a_{degree}]``.
        """
        coeffs = [secret % self.prime]
        for _ in range(degree):
            coeffs.append(secrets.randbelow(self.prime))
        return coeffs

    def _evaluate_polynomial(
        self, coeffs: List[int], x: int, prime: int
    ) -> int:
        """Evaluate a polynomial at point *x* modulo *prime* (Horner's method).

        Args:
            coeffs: Coefficients ``[a_0, a_1, ..., a_d]``.
            x: Evaluation point.
            prime: Prime modulus.

        Returns:
            ``P(x) mod prime``.
        """
        result = 0
        for coeff in reversed(coeffs):
            result = (result * x + coeff) % prime
        return result

    def split_secret(self, secret_int: int) -> List[Tuple[int, int]]:
        """Split a secret integer into *n* Shamir shares.

        Args:
            secret_int: Non-negative integer to split.

        Returns:
            List of ``(x, y)`` share tuples where ``x ∈ {1, ..., n}``.
        """
        secret_mod = secret_int % self.prime
        degree = self.k_threshold - 1
        coeffs = self._generate_polynomial(secret_mod, degree)

        shares = []
        for i in range(1, self.n_shares + 1):
            y = self._evaluate_polynomial(coeffs, i, self.prime)
            shares.append((i, y))
        return shares

    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> int:
        """Reconstruct a secret from *k* or more shares via Lagrange interpolation.

        Args:
            shares: At least *k_threshold* ``(x, y)`` tuples.

        Returns:
            Reconstructed secret integer.

        Raises:
            ValueError: If fewer than *k_threshold* shares are provided.
        """
        if len(shares) < self.k_threshold:
            raise ValueError(
                f"Need at least {self.k_threshold} shares, got {len(shares)}."
            )

        # Use only the first k_threshold shares
        shares = shares[: self.k_threshold]
        secret = 0

        for j in range(len(shares)):
            basis = self._lagrange_basis(shares, j, 0, self.prime)
            secret = (secret + shares[j][1] * basis) % self.prime

        return secret

    def _lagrange_basis(
        self, shares: List[Tuple[int, int]], j: int, x: int, prime: int
    ) -> int:
        """Compute the Lagrange basis polynomial L_j(x) mod prime.

        Args:
            shares: List of ``(x_i, y_i)`` share tuples.
            j: Index of the basis polynomial.
            x: Point at which to evaluate.
            prime: Prime modulus.

        Returns:
            L_j(x) mod prime.
        """
        x_j = shares[j][0]
        numerator = 1
        denominator = 1

        for m in range(len(shares)):
            if m == j:
                continue
            x_m = shares[m][0]
            numerator = (numerator * (x - x_m)) % prime
            denominator = (denominator * (x_j - x_m)) % prime

        # Modular inverse via Fermat's little theorem: a^{-1} = a^{p-2} mod p
        inv_denom = pow(denominator, prime - 2, prime)
        return (numerator * inv_denom) % prime


# ---------------------------------------------------------------------------
# SMPCNode
# ---------------------------------------------------------------------------


class SMPCNode:
    """A single SMPC computation node that stores gradient shares.

    In the FedFOD architecture there are three nodes:
      * **ICAO** — regulatory authority node
      * **Consortium** — airport consortium node
      * **NeutralParty** — independent auditor node

    Each node stores its assigned share for each (round, client) pair and
    releases them on request during reconstruction.

    Args:
        node_id: Unique integer identifier (0, 1, or 2).
        node_name: Human-readable node name.
        sharing_scheme: Reference to the ``ShamirSecretSharing`` instance.
    """

    def __init__(
        self,
        node_id: int,
        node_name: str,
        sharing_scheme: ShamirSecretSharing,
    ) -> None:
        self.node_id = node_id
        self.node_name = node_name
        self.sharing_scheme = sharing_scheme
        self.online = True

        # Storage: { round_id: { client_id: { param_key: (x, y) } } }
        self._shares: Dict[int, Dict[str, Dict[str, Tuple[int, int]]]] = {}

        logger.debug("SMPCNode '%s' (id=%d) initialised.", node_name, node_id)

    def hold_share(
        self,
        round_id: int,
        client_id: str,
        share: Dict[str, Tuple[int, int]],
    ) -> None:
        """Store a share for a specific round and client.

        Args:
            round_id: Federated round number.
            client_id: Identifier of the contributing client.
            share: Mapping of ``param_key → (x, y)`` share tuple.
        """
        if round_id not in self._shares:
            self._shares[round_id] = {}
        self._shares[round_id][client_id] = share
        logger.debug(
            "Node '%s': stored %d shares for round=%d, client=%s.",
            self.node_name,
            len(share),
            round_id,
            client_id,
        )

    def release_shares(
        self, round_id: int, client_id: str
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """Release stored shares for reconstruction.

        Args:
            round_id: Federated round number.
            client_id: Client identifier.

        Returns:
            Dict of ``param_key → (x, y)`` tuples, or None if unavailable.
        """
        if not self.online:
            logger.warning(
                "Node '%s' is OFFLINE — cannot release shares.", self.node_name
            )
            return None

        round_data = self._shares.get(round_id)
        if round_data is None:
            logger.warning(
                "Node '%s': no shares for round %d.", self.node_name, round_id
            )
            return None

        client_data = round_data.get(client_id)
        if client_data is None:
            logger.warning(
                "Node '%s': no shares for client '%s' in round %d.",
                self.node_name,
                client_id,
                round_id,
            )
            return None

        return client_data

    def clear_round(self, round_id: int) -> None:
        """Remove all stored shares for a completed round.

        Args:
            round_id: Round number to clear.
        """
        if round_id in self._shares:
            del self._shares[round_id]
            logger.debug(
                "Node '%s': cleared shares for round %d.", self.node_name, round_id
            )


# ---------------------------------------------------------------------------
# SMPCGradientProtector
# ---------------------------------------------------------------------------


class SMPCGradientProtector:
    """Orchestrator that splits and reconstructs gradient dictionaries across
    three SMPC nodes using Shamir Secret Sharing.

    Gradient float values are quantised to integers (via a configurable scale
    factor), split into shares, distributed to the three nodes, and later
    reconstructed from any *k* available nodes.

    Args:
        n_shares: Number of shares (should be 3 for the FedFOD architecture).
        k_threshold: Reconstruction threshold (default 2).
    """

    SCALE_FACTOR: float = 1e6  # float → int quantisation scale

    NODE_NAMES = ["ICAO", "Consortium", "NeutralParty"]

    def __init__(self, n_shares: int = 3, k_threshold: int = 2) -> None:
        self.sharing = ShamirSecretSharing(
            n_shares=n_shares, k_threshold=k_threshold
        )
        self.nodes: List[SMPCNode] = [
            SMPCNode(node_id=i, node_name=name, sharing_scheme=self.sharing)
            for i, name in enumerate(self.NODE_NAMES[:n_shares])
        ]
        self.n_shares = n_shares
        self.k_threshold = k_threshold

        logger.info(
            "SMPCGradientProtector initialised with %d nodes (threshold=%d): %s",
            n_shares,
            k_threshold,
            [n.node_name for n in self.nodes],
        )

    # ----- Float ↔ int conversion -----

    @staticmethod
    def _float_to_int(f: float, scale: float = 1e6) -> int:
        """Convert a float to a non-negative integer for secret sharing.

        Negative values are handled via an offset scheme:
        the sign is encoded in the high bits.

        Args:
            f: Float value to convert.
            scale: Scaling factor for precision.

        Returns:
            Non-negative integer encoding of *f*.
        """
        scaled = int(round(f * scale))
        # Encode sign: positive → 2*abs, negative → 2*abs - 1
        if scaled >= 0:
            return 2 * scaled
        else:
            return 2 * abs(scaled) - 1

    @staticmethod
    def _int_to_float(i: int, scale: float = 1e6) -> float:
        """Convert a non-negative integer back to a float.

        Args:
            i: Non-negative integer (produced by ``_float_to_int``).
            scale: Scaling factor (must match the one used for encoding).

        Returns:
            Reconstructed float value.
        """
        if i % 2 == 0:
            return (i // 2) / scale
        else:
            return -((i + 1) // 2) / scale

    # ----- Split -----

    def split_gradient_shares(
        self,
        gradient_dict: Dict[str, float],
        round_id: int,
        client_id: str,
    ) -> None:
        """Split a gradient dictionary across the SMPC nodes.

        Each float value is quantised and split into *n* Shamir shares.
        Share *i* is distributed to node *i*.

        Args:
            gradient_dict: Mapping of parameter key → gradient float value.
            round_id: Current federated round.
            client_id: Identifier of the contributing client.
        """
        # Prepare per-node share dicts
        per_node: List[Dict[str, Tuple[int, int]]] = [
            {} for _ in range(self.n_shares)
        ]

        for param_key, grad_value in gradient_dict.items():
            secret_int = self._float_to_int(grad_value, self.SCALE_FACTOR)
            shares = self.sharing.split_secret(secret_int)

            for node_idx, (x, y) in enumerate(shares):
                per_node[node_idx][param_key] = (x, y)

        # Distribute to nodes
        for node_idx, node in enumerate(self.nodes):
            node.hold_share(round_id, client_id, per_node[node_idx])

        logger.info(
            "Split %d gradient values for client '%s' (round %d) across %d nodes.",
            len(gradient_dict),
            client_id,
            round_id,
            self.n_shares,
        )

    # ----- Reconstruct -----

    def reconstruct_gradient(
        self,
        round_id: int,
        client_id: str,
        available_node_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Reconstruct a gradient dictionary from available node shares.

        Args:
            round_id: Federated round.
            client_id: Client identifier.
            available_node_ids: Indices of nodes to query.  If None, all
                online nodes are used.

        Returns:
            Reconstructed gradient dict mapping param key → float.

        Raises:
            ValueError: If fewer than *k_threshold* nodes are available.
        """
        if available_node_ids is None:
            available_node_ids = [
                n.node_id for n in self.nodes if n.online
            ]

        if len(available_node_ids) < self.k_threshold:
            raise ValueError(
                f"Need at least {self.k_threshold} available nodes, "
                f"got {len(available_node_ids)}: {available_node_ids}."
            )

        # Collect shares from available nodes
        node_shares: List[Optional[Dict[str, Tuple[int, int]]]] = []
        active_node_ids: List[int] = []

        for nid in available_node_ids:
            node = self.nodes[nid]
            shares = node.release_shares(round_id, client_id)
            if shares is not None:
                node_shares.append(shares)
                active_node_ids.append(nid)

        if len(node_shares) < self.k_threshold:
            raise ValueError(
                f"Only {len(node_shares)} nodes returned shares; "
                f"need {self.k_threshold}."
            )

        # Determine parameter keys from the first node's shares
        param_keys = list(node_shares[0].keys())
        reconstructed: Dict[str, float] = {}

        for param_key in param_keys:
            shares_for_key: List[Tuple[int, int]] = []
            for ns in node_shares:
                if param_key in ns:
                    shares_for_key.append(ns[param_key])

            if len(shares_for_key) < self.k_threshold:
                logger.warning(
                    "Insufficient shares for param '%s'; skipping.", param_key
                )
                continue

            secret_int = self.sharing.reconstruct_secret(shares_for_key)
            reconstructed[param_key] = self._int_to_float(
                secret_int, self.SCALE_FACTOR
            )

        logger.info(
            "Reconstructed %d gradient values for client '%s' (round %d) "
            "from %d nodes.",
            len(reconstructed),
            client_id,
            round_id,
            len(node_shares),
        )
        return reconstructed

    # ----- Fault simulation -----

    def simulate_node_failure(self, node_id: int) -> None:
        """Mark a node as offline to simulate network partition or failure.

        Args:
            node_id: Index of the node to take offline.
        """
        if 0 <= node_id < len(self.nodes):
            self.nodes[node_id].online = False
            logger.warning(
                "Node '%s' (id=%d) marked OFFLINE.",
                self.nodes[node_id].node_name,
                node_id,
            )
        else:
            raise IndexError(
                f"Node id {node_id} out of range [0, {len(self.nodes) - 1}]."
            )

    def restore_node(self, node_id: int) -> None:
        """Bring an offline node back online.

        Args:
            node_id: Index of the node to restore.
        """
        if 0 <= node_id < len(self.nodes):
            self.nodes[node_id].online = True
            logger.info(
                "Node '%s' (id=%d) restored ONLINE.",
                self.nodes[node_id].node_name,
                node_id,
            )
        else:
            raise IndexError(
                f"Node id {node_id} out of range [0, {len(self.nodes) - 1}]."
            )

    def clear_round(self, round_id: int) -> None:
        """Clear all stored shares for a completed round across all nodes.

        Args:
            round_id: Round number to clear.
        """
        for node in self.nodes:
            node.clear_round(round_id)
        logger.debug("Cleared round %d data from all SMPC nodes.", round_id)
