"""
FedFOD Prototype Memory Bank
=============================
Open-world prototype memory for CLIP embeddings that enables the federated
system to discover and track novel FOD categories at inference time.

Each prototype is a 512-dimensional CLIP embedding associated with a class
name, confidence score, and provenance metadata.  The memory bank supports:

  * **Novelty processing** — incoming embeddings are classified as
    merge / accept-as-variant / reject based on cosine similarity thresholds.
  * **Running-average updates** — prototypes are refined over rounds as
    more observations arrive.
  * **Staleness eviction** — prototypes that haven't been updated within a
    configurable window are evicted.
  * **FIFO capacity enforcement** — oldest entries are evicted when the
    bank exceeds its maximum size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.utils import (
    PrototypePayload,
    cosine_similarity,
    COSINE_SIM_ACCEPT_THRESHOLD,
    COSINE_SIM_MERGE_THRESHOLD,
    DEFAULT_EMBED_DIM,
    PROTOTYPE_MAX_ENTRIES,
    PROTOTYPE_STALE_ROUNDS,
)

logger = logging.getLogger("fedfod.prototype_memory")

# ---------------------------------------------------------------------------
# PrototypeEntry
# ---------------------------------------------------------------------------


@dataclass
class PrototypeEntry:
    """A single prototype in the memory bank.

    Attributes:
        embedding: CLIP embedding vector (``embed_dim``-dimensional).
        class_name: Canonical class name or ``novel_<id>`` for discoveries.
        confidence: Detection confidence, updated as running average.
        source_client: ID of the client that first reported this prototype.
        created_round: Federated round when this prototype was created.
        last_updated_round: Last round in which this prototype was updated.
        update_count: Number of times this prototype has been updated.
    """

    embedding: np.ndarray
    class_name: str
    confidence: float = 0.0
    source_client: str = ""
    created_round: int = 0
    last_updated_round: int = 0
    update_count: int = 1


# ---------------------------------------------------------------------------
# PrototypeMemoryBank
# ---------------------------------------------------------------------------


class PrototypeMemoryBank:
    """Bounded memory bank for open-world CLIP prototypes.

    Args:
        max_entries: Maximum number of prototypes to store.
        stale_threshold_rounds: Number of rounds without update before a
            prototype is considered stale and eligible for eviction.
        embed_dim: Dimensionality of CLIP embedding vectors.
    """

    def __init__(
        self,
        max_entries: int = PROTOTYPE_MAX_ENTRIES,
        stale_threshold_rounds: int = PROTOTYPE_STALE_ROUNDS,
        embed_dim: int = DEFAULT_EMBED_DIM,
    ) -> None:
        self.max_entries = max_entries
        self.stale_threshold_rounds = stale_threshold_rounds
        self.embed_dim = embed_dim

        # Storage: class_name → PrototypeEntry
        self._bank: Dict[str, PrototypeEntry] = {}

        # Insertion order tracking for FIFO eviction
        self._insertion_order: List[str] = []

        logger.info(
            "PrototypeMemoryBank initialised: max=%d, stale_threshold=%d rounds, dim=%d.",
            max_entries,
            stale_threshold_rounds,
            embed_dim,
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_prototype(self, entry: PrototypeEntry) -> bool:
        """Add a new prototype to the memory bank.

        A prototype is accepted if:
          * It matches an existing prototype with cosine similarity ≥ 0.70
            (in which case it is merged), **or**
          * It is genuinely novel (no existing prototype has similarity ≥ 0.70).

        Prototypes with similarity between 0.70 and 0.85 to an existing entry
        are added as separate variants.

        Args:
            entry: The prototype to add.

        Returns:
            True if the prototype was accepted (added or merged), False if
            rejected.
        """
        if entry.embedding.shape[-1] != self.embed_dim:
            logger.warning(
                "Embedding dimension mismatch: expected %d, got %s.",
                self.embed_dim,
                entry.embedding.shape,
            )
            return False

        # Check for similarity with existing prototypes
        best_sim = -1.0
        best_name: Optional[str] = None

        for name, existing in self._bank.items():
            sim = self._cosine_similarity(entry.embedding, existing.embedding)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= COSINE_SIM_MERGE_THRESHOLD and best_name is not None:
            # High similarity → merge into existing prototype
            self.update_prototype(
                best_name, entry.embedding, entry.last_updated_round
            )
            logger.debug(
                "Merged new prototype into existing '%s' (sim=%.4f).",
                best_name,
                best_sim,
            )
            return True

        if best_sim >= COSINE_SIM_ACCEPT_THRESHOLD and best_name is not None:
            # Moderate similarity → accept as variant with distinct name
            variant_name = entry.class_name
            if variant_name in self._bank:
                variant_name = f"{entry.class_name}_v{len(self._bank)}"
            self._bank[variant_name] = entry
            self._insertion_order.append(variant_name)
            self._enforce_capacity()
            logger.info(
                "Accepted variant prototype '%s' (sim=%.4f with '%s').",
                variant_name,
                best_sim,
                best_name,
            )
            return True

        if best_sim < COSINE_SIM_ACCEPT_THRESHOLD:
            # Truly novel — accept as new class
            name = entry.class_name
            if name in self._bank:
                name = f"{entry.class_name}_{len(self._bank)}"
            self._bank[name] = entry
            self._insertion_order.append(name)
            self._enforce_capacity()
            logger.info(
                "Added novel prototype '%s' (best existing sim=%.4f).",
                name,
                best_sim,
            )
            return True

        return False

    def update_prototype(
        self,
        class_name: str,
        new_embedding: np.ndarray,
        round_number: int,
    ) -> None:
        """Update an existing prototype via running average.

        The embedding is updated as:
            ``e_new = (count * e_old + e_incoming) / (count + 1)``

        Args:
            class_name: Name of the prototype to update.
            new_embedding: New observation embedding.
            round_number: Current federated round.

        Raises:
            KeyError: If *class_name* is not in the bank.
        """
        if class_name not in self._bank:
            raise KeyError(f"Prototype '{class_name}' not found in memory bank.")

        entry = self._bank[class_name]
        count = entry.update_count
        entry.embedding = (
            (count * entry.embedding.astype(np.float64) + new_embedding.astype(np.float64))
            / (count + 1)
        ).astype(np.float32)
        entry.update_count = count + 1
        entry.last_updated_round = round_number

        # Normalise the embedding to unit length
        norm = np.linalg.norm(entry.embedding)
        if norm > 1e-12:
            entry.embedding = entry.embedding / norm

        logger.debug(
            "Updated prototype '%s': count=%d, round=%d.",
            class_name,
            entry.update_count,
            round_number,
        )

    def query_nearest(
        self, embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find the *top_k* most similar prototypes to a query embedding.

        Args:
            embedding: Query embedding vector.
            top_k: Number of nearest neighbours to return.

        Returns:
            List of ``(class_name, cosine_similarity)`` tuples, sorted
            by descending similarity.
        """
        if not self._bank:
            return []

        similarities: List[Tuple[str, float]] = []
        for name, entry in self._bank.items():
            sim = self._cosine_similarity(embedding, entry.embedding)
            similarities.append((name, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_prototype(self, class_name: str) -> Optional[PrototypeEntry]:
        """Retrieve a prototype by class name.

        Args:
            class_name: Name of the prototype.

        Returns:
            PrototypeEntry or None if not found.
        """
        return self._bank.get(class_name)

    def evict_stale(self, current_round: int) -> int:
        """Remove prototypes that haven't been updated recently.

        A prototype is stale if:
            ``current_round − last_updated_round > stale_threshold_rounds``

        Args:
            current_round: Current federated round.

        Returns:
            Number of evicted prototypes.
        """
        stale_names: List[str] = []
        for name, entry in self._bank.items():
            if (current_round - entry.last_updated_round) > self.stale_threshold_rounds:
                stale_names.append(name)

        for name in stale_names:
            del self._bank[name]
            if name in self._insertion_order:
                self._insertion_order.remove(name)

        if stale_names:
            logger.info(
                "Evicted %d stale prototypes at round %d: %s",
                len(stale_names),
                current_round,
                stale_names,
            )
        return len(stale_names)

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity in [-1, 1].
        """
        return cosine_similarity(a, b)

    # ------------------------------------------------------------------
    # Bulk accessors
    # ------------------------------------------------------------------

    def get_all_prototypes(self) -> Dict[str, PrototypeEntry]:
        """Return a shallow copy of the entire prototype bank.

        Returns:
            Dict mapping class_name → PrototypeEntry.
        """
        return dict(self._bank)

    def get_novel_prototypes(self) -> List[PrototypeEntry]:
        """Return prototypes whose class name starts with ``novel_``.

        Returns:
            List of PrototypeEntry instances for novel detections.
        """
        return [
            entry
            for name, entry in self._bank.items()
            if name.startswith("novel_")
        ]

    # ------------------------------------------------------------------
    # Novelty report processing
    # ------------------------------------------------------------------

    def process_novelty_report(
        self, payload: PrototypePayload, current_round: int
    ) -> str:
        """Process a novelty report from a client.

        Decision logic:
          * cosine_sim ≥ 0.85 with existing → **merge** into that prototype
          * 0.70 ≤ cosine_sim < 0.85 → **accept** as a separate variant
          * cosine_sim < 0.70 → **reject** (too dissimilar, may be noise)

        Args:
            payload: Novelty report from a client.
            current_round: Current federated round.

        Returns:
            Action taken: ``"merged"``, ``"accepted"``, or ``"rejected"``.
        """
        if payload.embedding_vector is None:
            logger.warning("Novelty report from '%s' has no embedding; rejecting.", payload.client_id)
            return "reject"

        embedding = payload.embedding_vector
        if embedding.shape[-1] != self.embed_dim:
            logger.warning(
                "Embedding dimension mismatch in novelty report: %s vs %d.",
                embedding.shape,
                self.embed_dim,
            )
            return "rejected"

        # Find best match
        best_sim = -1.0
        best_name: Optional[str] = None

        for name, entry in self._bank.items():
            sim = self._cosine_similarity(embedding, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= COSINE_SIM_MERGE_THRESHOLD and best_name is not None:
            # Merge
            self.update_prototype(best_name, embedding, current_round)
            logger.info(
                "Novelty report from '%s': MERGED into '%s' (sim=%.4f).",
                payload.client_id,
                best_name,
                best_sim,
            )
            return "merge"

        if best_sim >= COSINE_SIM_ACCEPT_THRESHOLD:
            # Accept as variant
            variant_name = payload.fod_candidate_class
            if not variant_name:
                variant_name = f"novel_{len(self._bank)}"
            if variant_name in self._bank:
                variant_name = f"{variant_name}_v{len(self._bank)}"

            entry = PrototypeEntry(
                embedding=embedding.astype(np.float32),
                class_name=variant_name,
                confidence=payload.detection_confidence,
                source_client=payload.client_id,
                created_round=current_round,
                last_updated_round=current_round,
                update_count=1,
            )
            self._bank[variant_name] = entry
            self._insertion_order.append(variant_name)
            self._enforce_capacity()

            logger.info(
                "Novelty report from '%s': ACCEPTED as '%s' (sim=%.4f with '%s').",
                payload.client_id,
                variant_name,
                best_sim,
                best_name,
            )
            return "accept"

        # Reject — too dissimilar to anything known
        logger.info(
            "Novelty report from '%s': REJECTED (best_sim=%.4f < %.2f).",
            payload.client_id,
            best_sim,
            COSINE_SIM_ACCEPT_THRESHOLD,
        )
        return "reject"

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of prototypes in the bank."""
        return len(self._bank)

    def __repr__(self) -> str:
        return (
            f"PrototypeMemoryBank(entries={len(self._bank)}, "
            f"max={self.max_entries}, dim={self.embed_dim})"
        )

    # ------------------------------------------------------------------
    # Capacity management
    # ------------------------------------------------------------------

    def _enforce_capacity(self) -> None:
        """Evict oldest entries (FIFO) if the bank exceeds ``max_entries``."""
        while len(self._bank) > self.max_entries:
            if not self._insertion_order:
                # Fallback: remove an arbitrary entry
                oldest_name = next(iter(self._bank))
            else:
                oldest_name = self._insertion_order.pop(0)

            if oldest_name in self._bank:
                del self._bank[oldest_name]
                logger.debug(
                    "FIFO eviction: removed prototype '%s' (bank size=%d).",
                    oldest_name,
                    len(self._bank),
                )
