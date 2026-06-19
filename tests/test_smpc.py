"""Tests for Shamir Secret Sharing and SMPC gradient protection."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.server.smpc_node import ShamirSecretSharing, SMPCNode, SMPCGradientProtector


# ---- Fixtures ----

@pytest.fixture
def shamir():
    """Create a ShamirSecretSharing instance with n=3, k=2."""
    return ShamirSecretSharing(n_shares=3, k_threshold=2)


@pytest.fixture
def protector():
    """Create an SMPCGradientProtector instance."""
    return SMPCGradientProtector(n_shares=3, k_threshold=2)


# ---- ShamirSecretSharing Tests ----

def test_split_and_reconstruct_exact(shamir):
    """Split a secret into 3 shares, reconstruct with all 3."""
    secret = 12345
    shares = shamir.split_secret(secret)
    assert len(shares) == 3
    reconstructed = shamir.reconstruct_secret(shares)
    assert reconstructed == secret, f"Expected {secret}, got {reconstructed}"


def test_reconstruct_with_k_shares(shamir):
    """Reconstruct with exactly k=2 shares."""
    secret = 12345
    shares = shamir.split_secret(secret)
    reconstructed = shamir.reconstruct_secret(shares[:2])
    assert reconstructed == secret, f"Expected {secret}, got {reconstructed}"


def test_k_minus_1_insufficient(shamir):
    """With k-1=1 share, reconstruction should NOT equal original."""
    secret = 12345
    shares = shamir.split_secret(secret)
    single = [shares[0]]
    try:
        result = shamir.reconstruct_secret(single)
        # If it doesn't raise, result should differ from secret
        assert result != secret, "Single share should not reconstruct the secret"
    except (ValueError, ZeroDivisionError, IndexError) as e:
        # Raising an error is also acceptable behaviour
        assert e is not None


def test_different_share_combinations(shamir):
    """All valid k-of-n combinations should give the same secret."""
    secret = 12345
    shares = shamir.split_secret(secret)
    combos = [
        shares[0:2],              # shares 0,1
        shares[1:3],              # shares 1,2
        [shares[0], shares[2]],   # shares 0,2
    ]
    for i, combo in enumerate(combos):
        result = shamir.reconstruct_secret(combo)
        assert result == secret, f"Combo {i} gave {result}, expected {secret}"


def test_large_secret():
    """Test with a very large integer value."""
    shamir = ShamirSecretSharing(n_shares=5, k_threshold=3)
    secret = 10**15
    shares = shamir.split_secret(secret)
    assert len(shares) == 5
    reconstructed = shamir.reconstruct_secret(shares[:3])
    assert reconstructed == secret, f"Large secret: expected {secret}, got {reconstructed}"


# ---- SMPCGradientProtector Tests ----

def test_gradient_protector_roundtrip(protector):
    """Split a gradient dict into shares, reconstruct, verify closeness."""
    rng = np.random.RandomState(42)
    grad = {
        'layer1': float(rng.randn()),
        'layer2': float(rng.randn()),
    }
    round_id = 1
    client_id = "test_client"
    protector.split_gradient_shares(grad, round_id, client_id)
    reconstructed = protector.reconstruct_gradient(round_id, client_id)
    for k in grad:
        assert abs(reconstructed[k] - grad[k]) < 0.01, (
            f"Gradient protector roundtrip failed for {k}: "
            f"expected {grad[k]}, got {reconstructed[k]}"
        )


def test_node_failure_still_works(protector):
    """Simulate 1 node failure — reconstruction with remaining 2 nodes."""
    rng = np.random.RandomState(7)
    grad = {'w': float(rng.randn())}
    round_id = 2
    client_id = "test_client2"
    protector.split_gradient_shares(grad, round_id, client_id)
    # Simulate node 2 (index 2) going offline
    protector.simulate_node_failure(2)
    # Should still reconstruct with nodes 0 and 1 (k=2)
    reconstructed = protector.reconstruct_gradient(
        round_id, client_id, available_node_ids=[0, 1]
    )
    for k in grad:
        assert abs(reconstructed[k] - grad[k]) < 0.01, (
            f"Reconstruction after node failure failed for {k}"
        )
    # Restore node for other tests
    protector.restore_node(2)


def test_multiple_clients_same_round(protector):
    """Split gradients for 2 different clients independently."""
    rng = np.random.RandomState(99)
    grad_a = {'w': float(rng.randn())}
    grad_b = {'w': float(rng.randn())}
    round_id = 3

    protector.split_gradient_shares(grad_a, round_id, "client_a")
    protector.split_gradient_shares(grad_b, round_id, "client_b")

    recon_a = protector.reconstruct_gradient(round_id, "client_a")
    recon_b = protector.reconstruct_gradient(round_id, "client_b")

    for k in grad_a:
        assert abs(recon_a[k] - grad_a[k]) < 0.01
        assert abs(recon_b[k] - grad_b[k]) < 0.01
        # They should differ from each other
        assert abs(recon_a[k] - recon_b[k]) > 0.001, (
            "Different clients should have different gradients"
        )
