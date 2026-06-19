"""Tests for SCAFFOLD math and utility functions in src.common.utils."""
import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    gini_coefficient,
)


# ---- Fixtures ----

@pytest.fixture
def dummy_weights():
    """Create reproducible dummy weight dicts for SCAFFOLD tests."""
    rng = np.random.RandomState(0)
    keys = ['layer1', 'layer2']
    w_local = {k: rng.randn(4, 4).astype(np.float32) for k in keys}
    w_global = {k: rng.randn(4, 4).astype(np.float32) for k in keys}
    c_local = {k: rng.randn(4, 4).astype(np.float32) for k in keys}
    c_global = {k: rng.randn(4, 4).astype(np.float32) for k in keys}
    grad = {k: rng.randn(4, 4).astype(np.float32) for k in keys}
    return w_local, w_global, c_local, c_global, grad


# ---- SCAFFOLD Tests ----

def test_scaffold_local_update_changes_weights(dummy_weights):
    w_local, w_global, c_local, c_global, grad = dummy_weights
    w_new, c_new = scaffold_local_update(w_local, w_global, c_local, c_global, grad, eta=0.01, K=10)
    any_changed = any(not np.array_equal(w_new[k], w_local[k]) for k in w_local)
    assert any_changed, "SCAFFOLD update should change weights"


def test_scaffold_correction_formula(dummy_weights):
    w_local, w_global, c_local, c_global, grad = dummy_weights
    eta = 0.01
    K = 10
    w_new, c_new = scaffold_local_update(w_local, w_global, c_local, c_global, grad, eta=eta, K=K)
    for k in w_local:
        expected_w = w_local[k] - eta * (grad[k] - c_local[k] + c_global[k])
        np.testing.assert_allclose(w_new[k], expected_w, rtol=1e-5, atol=1e-7,
                                    err_msg=f"SCAFFOLD correction formula mismatch for {k}")


def test_control_variate_update(dummy_weights):
    w_local, w_global, c_local, c_global, grad = dummy_weights
    eta = 0.01
    K = 10
    w_new, c_new = scaffold_local_update(w_local, w_global, c_local, c_global, grad, eta=eta, K=K)
    for k in w_local:
        expected_c = c_local[k] - c_global[k] + (1.0 / (K * eta)) * (w_global[k] - w_new[k])
        np.testing.assert_allclose(c_new[k], expected_c, rtol=1e-5, atol=1e-7,
                                    err_msg=f"Control variate formula mismatch for {k}")


def test_control_variate_delta(dummy_weights):
    w_local, w_global, c_local, c_global, grad = dummy_weights
    _, c_new = scaffold_local_update(w_local, w_global, c_local, c_global, grad, eta=0.01, K=10)
    delta = compute_control_variate_delta(c_new, c_local)
    for k in c_new:
        np.testing.assert_allclose(delta[k], c_new[k] - c_local[k], rtol=1e-6)


def test_global_control_update():
    rng = np.random.RandomState(1)
    keys = ['a', 'b']
    c_global = {k: rng.randn(3, 3).astype(np.float32) for k in keys}
    delta1 = {k: rng.randn(3, 3).astype(np.float32) for k in keys}
    delta2 = {k: rng.randn(3, 3).astype(np.float32) for k in keys}
    c_new = apply_global_control_update(c_global, [delta1, delta2], n_clients=2)
    for k in keys:
        expected = c_global[k] + (delta1[k] + delta2[k]) / 2.0
        np.testing.assert_allclose(c_new[k], expected, rtol=1e-5)


# ---- Compression Tests ----

def test_sparsify_top_k():
    rng = np.random.RandomState(2)
    arr = rng.randn(100).astype(np.float32)
    delta = {'w': arr}
    sparse, masks = sparsify_gradients(delta, top_k_pct=0.50)
    nnz = np.count_nonzero(sparse['w'])
    assert 45 <= nnz <= 55, f"Expected ~50 non-zeros at 50% top-k, got {nnz}"


def test_quantize_dequantize_roundtrip():
    rng = np.random.RandomState(3)
    original = {'layer': rng.randn(8, 8).astype(np.float32)}
    q, scales = quantize_to_int8(original)
    restored = dequantize_from_int8(q, scales)
    np.testing.assert_allclose(restored['layer'], original['layer'], atol=0.1,
                                err_msg="Quantize-dequantize roundtrip error too large")


# ---- DP / Clipping Tests ----

def test_clip_gradient_norm():
    large = {'w': np.ones((50, 50), dtype=np.float32) * 10.0}
    clipped = clip_gradient_norm(large, max_norm=1.0)
    total_norm = np.sqrt(sum(np.sum(v.astype(np.float64) ** 2) for v in clipped.values()))
    assert total_norm <= 1.0 + 1e-6, f"Clipped norm {total_norm} exceeds max_norm 1.0"


def test_dp_noise_changes_values():
    rng_state = np.random.get_state()
    original = {'w': np.ones((4, 4), dtype=np.float32)}
    noisy = add_gaussian_dp_noise(original, epsilon=4.0, delta=1e-6, clip_norm=1.0)
    assert not np.array_equal(noisy['w'], original['w']), "DP noise should change values"
    np.random.set_state(rng_state)  # restore state


# ---- Staleness Tests ----

def test_staleness_weight_decay():
    w0 = compute_staleness_weight(0)
    w5 = compute_staleness_weight(5)
    assert w0 > w5, "Weight should decrease with staleness"
    assert abs(w0 - 1.0) < 1e-9, "Zero staleness should give weight 1.0"


def test_staleness_quarantine():
    # staleness > 15 is capped at 15
    w16 = compute_staleness_weight(16)
    w15 = compute_staleness_weight(15)
    assert abs(w16 - w15) < 1e-9, "Staleness > 15 should be capped at 15"
    expected = 0.85 ** 15
    assert abs(w16 - expected) < 1e-6, f"Expected {expected}, got {w16}"


# ---- FA Weight Tests ----

def test_fa_weight_computation():
    q = compute_fa_weight(historical_precision=0.9, num_samples=100, total_samples=1000)
    expected = 0.9 * (100 / 1000)
    assert abs(q - expected) < 1e-9, f"FA weight: expected {expected}, got {q}"


def test_normalize_weights():
    w = normalize_weights([3.0, 1.0, 1.0])
    assert abs(sum(w) - 1.0) < 1e-9, "Normalized weights must sum to 1.0"
    assert abs(w[0] - 0.6) < 1e-9
    assert abs(w[1] - 0.2) < 1e-9


# ---- Gini Tests ----

def test_gini_coefficient_equal():
    g = gini_coefficient([10.0, 10.0, 10.0, 10.0])
    assert abs(g) < 1e-9, f"Equal values should give Gini ≈ 0, got {g}"


def test_gini_coefficient_unequal():
    g = gini_coefficient([0.0, 0.0, 0.0, 100.0])
    assert g > 0.7, f"Highly unequal distribution should give Gini > 0.7, got {g}"
    assert g < 1.0, f"Gini should be < 1.0, got {g}"
