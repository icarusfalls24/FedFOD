"""Tests for PrototypeMemoryBank and PrototypeEntry."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.server.prototype_memory import PrototypeMemoryBank, PrototypeEntry
from src.common.utils import PrototypePayload


# ---- Fixtures ----

@pytest.fixture
def bank():
    return PrototypeMemoryBank(max_entries=200)


def _make_entry(class_name="metal_fastener", created_round=0, seed=0):
    rng = np.random.RandomState(seed)
    emb = rng.randn(512).astype(np.float32)
    return PrototypeEntry(
        class_name=class_name,
        embedding=emb,
        confidence=0.9,
        source_client="test_client",
        created_round=created_round,
        last_updated_round=created_round,
        update_count=1,
    )


# ---- Basic Operations ----

def test_add_prototype(bank):
    entry = _make_entry()
    bank.add_prototype(entry)
    assert len(bank) == 1


def test_query_nearest(bank):
    # Add 3 prototypes with known embeddings
    e1 = _make_entry("class_a", seed=1)
    e1.embedding = np.array([1.0] + [0.0]*511, dtype=np.float32)
    e2 = _make_entry("class_b", seed=2)
    e2.embedding = np.array([0.0, 1.0] + [0.0]*510, dtype=np.float32)
    e3 = _make_entry("class_c", seed=3)
    e3.embedding = np.array([0.0, 0.0, 1.0] + [0.0]*509, dtype=np.float32)
    bank.add_prototype(e1)
    bank.add_prototype(e2)
    bank.add_prototype(e3)
    # Query with vector similar to e1
    query = np.array([0.95, 0.05] + [0.0]*510, dtype=np.float32)
    results = bank.query_nearest(query, top_k=1)
    assert len(results) >= 1
    # query_nearest returns List[Tuple[str, float]] = [(class_name, similarity)]
    nearest_class, sim = results[0]
    assert nearest_class == "class_a", f"Expected class_a, got {nearest_class}"


def test_cosine_threshold(bank):
    e = _make_entry("test")
    e.embedding = np.array([1.0] + [0.0]*511, dtype=np.float32)
    bank.add_prototype(e)
    # Query with orthogonal vector
    query = np.zeros(512, dtype=np.float32)
    query[500] = 1.0
    results = bank.query_nearest(query, top_k=1)
    _, sim = results[0]
    assert sim < 0.5, f"Orthogonal vectors should have low similarity, got {sim}"


# ---- Eviction ----

def test_stale_eviction(bank):
    entry = _make_entry(created_round=0)
    bank.add_prototype(entry)
    assert len(bank) == 1
    evicted = bank.evict_stale(current_round=60)
    assert evicted >= 1, "Entry from round 0 should be evicted at round 60 (>50 stale threshold)"
    assert len(bank) == 0


def test_no_stale_eviction_recent(bank):
    entry = _make_entry(created_round=40)
    bank.add_prototype(entry)
    evicted = bank.evict_stale(current_round=60)
    assert evicted == 0, "Entry only 20 rounds old should NOT be evicted"
    assert len(bank) == 1


def test_max_capacity():
    small_bank = PrototypeMemoryBank(max_entries=200)
    for i in range(210):
        entry = _make_entry(class_name=f"class_{i}", seed=i)
        small_bank.add_prototype(entry)
    assert len(small_bank) <= 200, f"Expected <=200 (FIFO eviction), got {len(small_bank)}"


# ---- Update ----

def test_update_prototype(bank):
    entry = _make_entry(class_name="metal_fastener")
    bank.add_prototype(entry)
    original_emb = entry.embedding.copy()
    new_emb = np.random.RandomState(99).randn(512).astype(np.float32)
    bank.update_prototype("metal_fastener", new_emb, round_number=5)
    # Get the updated prototype
    updated = bank.get_prototype("metal_fastener")
    assert updated is not None, "Prototype should still exist after update"
    assert not np.array_equal(updated.embedding, original_emb), "Embedding should change after update"


# ---- Novelty Processing ----

def test_process_novelty_merge(bank):
    base = np.random.RandomState(10).randn(512).astype(np.float32)
    base = base / np.linalg.norm(base)
    entry = _make_entry("known")
    entry.embedding = base
    bank.add_prototype(entry)
    # Very similar embedding -> merge
    similar = base + np.random.RandomState(11).randn(512).astype(np.float32) * 0.01
    similar = similar / np.linalg.norm(similar)
    payload = PrototypePayload(
        client_id="test",
        embedding_vector=similar,
        detection_confidence=0.9,
        fod_candidate_class="known",
    )
    action = bank.process_novelty_report(payload, current_round=1)
    assert action == "merge", f"Expected 'merge' for very similar embedding, got '{action}'"


def test_process_novelty_accept(bank):
    base = np.random.RandomState(20).randn(512).astype(np.float32)
    base = base / np.linalg.norm(base)
    entry = _make_entry("known")
    entry.embedding = base
    bank.add_prototype(entry)
    # Moderately similar (mix base with random)
    noise = np.random.RandomState(21).randn(512).astype(np.float32)
    mixed = 0.78 * base + 0.625 * noise / np.linalg.norm(noise)
    mixed = mixed / np.linalg.norm(mixed)
    payload = PrototypePayload(
        client_id="test",
        embedding_vector=mixed,
        detection_confidence=0.8,
        fod_candidate_class="maybe_new",
    )
    action = bank.process_novelty_report(payload, current_round=1)
    assert action == "accept", f"Expected 'accept' for moderately similar embedding, got '{action}'"


def test_process_novelty_reject(bank):
    base = np.random.RandomState(30).randn(512).astype(np.float32)
    base = base / np.linalg.norm(base)
    entry = _make_entry("known")
    entry.embedding = base
    bank.add_prototype(entry)
    # Very different embedding
    different = np.random.RandomState(31).randn(512).astype(np.float32)
    different = different / np.linalg.norm(different)
    payload = PrototypePayload(
        client_id="test",
        embedding_vector=different,
        detection_confidence=0.5,
        fod_candidate_class="unknown",
    )
    action = bank.process_novelty_report(payload, current_round=1)
    assert action == "reject", f"Expected 'reject' for dissimilar embedding, got '{action}'"
