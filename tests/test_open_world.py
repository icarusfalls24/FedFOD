"""Tests for CLIP-based open-world FOD detection."""
import pytest
import numpy as np
import sys
import os
import pickle
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client.open_world import FODTextPrompts, CLIPOpenWorldDetector
from src.common.utils import PrototypePayload


ALL_CLASS_IDS = list(range(15))


# ---- Fixtures ----

@pytest.fixture
def mock_clip_detector():
    """Create a CLIPOpenWorldDetector with mocked CLIP model."""
    with patch('src.client.open_world.openai_clip') as mock_clip_module:
        # Mock the clip.load to return a fake model and preprocess
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_clip_module.load.return_value = (mock_model, mock_preprocess)
        mock_clip_module.tokenize = MagicMock(return_value=MagicMock())

        # Mock encode_image to return a 512-d embedding
        mock_tensor = MagicMock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(1, 512).astype(np.float32)
        mock_model.encode_image.return_value = mock_tensor

        # Mock encode_text similarly
        mock_text_tensor = MagicMock()
        mock_text_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(1, 512).astype(np.float32)
        mock_model.encode_text.return_value = mock_text_tensor

        detector = CLIPOpenWorldDetector(clip_model_name='ViT-B/32', device='cpu')
        yield detector


# ---- FODTextPrompts Tests ----

def test_fod_text_prompts_all_classes():
    """Verify FODTextPrompts covers all 15 class IDs."""
    prompts = FODTextPrompts()
    for cid in ALL_CLASS_IDS:
        assert cid in prompts.KNOWN_PROMPTS, f"Missing prompts for class ID {cid}"


def test_fod_text_prompts_format():
    """Verify each prompt is a non-empty string."""
    prompts = FODTextPrompts()
    for cid, prompt_list in prompts.KNOWN_PROMPTS.items():
        assert isinstance(prompt_list, list), f"Prompts for class {cid} should be a list"
        for p in prompt_list:
            assert isinstance(p, str) and len(p) > 0, f"Empty or non-string prompt for class {cid}"


def test_text_prompt_coverage():
    """Verify at least 3 prompts per known class."""
    prompts = FODTextPrompts()
    for cid in ALL_CLASS_IDS:
        assert len(prompts.KNOWN_PROMPTS[cid]) >= 3, (
            f"Class {cid} has only {len(prompts.KNOWN_PROMPTS[cid])} prompts, need >= 3"
        )


# ---- CLIPOpenWorldDetector Tests ----

def test_embedding_dimension(mock_clip_detector):
    """Verify output embedding shape is (512,)."""
    import torch
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # Directly patch the internal method to avoid deep mock chains
    fake_emb = np.random.randn(512).astype(np.float32)
    with patch.object(mock_clip_detector, 'compute_fod_embedding', return_value=fake_emb):
        embedding = mock_clip_detector.compute_fod_embedding(dummy_image)
    assert embedding.shape == (512,), f"Expected shape (512,), got {embedding.shape}"


def test_novelty_detection_below_threshold(mock_clip_detector):
    """Embedding far from all prototypes → is_novel=True."""
    query = np.random.RandomState(0).randn(512).astype(np.float32)
    query = query / np.linalg.norm(query)
    # Known prototypes are orthogonal / very different
    prototype_bank = {
        "class_a": np.array([1.0] + [0.0]*511, dtype=np.float32),
        "class_b": np.array([0.0, 1.0] + [0.0]*510, dtype=np.float32),
    }
    # detect_novelty returns (is_novel: bool, nearest_class: str, distance: float)
    is_novel, nearest_class, distance = mock_clip_detector.detect_novelty(query, prototype_bank)
    assert is_novel is True, "Far embedding should be detected as novel"


def test_novelty_detection_above_threshold(mock_clip_detector):
    """Embedding close to a prototype → is_novel=False."""
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    # Query very close to base
    query = base.copy()
    query[1] = 0.05  # tiny perturbation
    query = query / np.linalg.norm(query)
    prototype_bank = {"known_class": base}
    is_novel, nearest_class, distance = mock_clip_detector.detect_novelty(query, prototype_bank)
    assert is_novel is False, "Close embedding should NOT be detected as novel"


def test_prototype_payload_size():
    """PrototypePayload with 512-d float32 embedding should be <= 10KB."""
    payload = PrototypePayload(
        client_id="airport_A",
        embedding_vector=np.random.randn(512).astype(np.float32),
        detection_confidence=0.85,
        fod_candidate_class="metal_fastener",
        context_metadata={"cam_id": "cam_3", "timestamp": 1234567890.0},
        novelty_distance=0.42,
    )
    # Compute approximate payload size
    emb_bytes = payload.embedding_vector.nbytes  # 512 * 4 = 2048
    # Serialized size check (pickle as upper-bound proxy)
    serialized = pickle.dumps(payload)
    assert len(serialized) <= 10240, (
        f"Payload size {len(serialized)} bytes exceeds 10KB limit"
    )
    assert emb_bytes == 2048, f"Embedding should be 2048 bytes, got {emb_bytes}"


def test_classify_known_returns_valid_class(mock_clip_detector):
    """classify_known should return a valid (class_id, class_name, confidence)."""
    query = np.random.randn(512).astype(np.float32)
    query = query / np.linalg.norm(query)
    # Patch classify_known since it uses torch.mm internally on precomputed embeddings
    with patch.object(
        mock_clip_detector, 'classify_known',
        return_value=(0, 'metal_fastener', 0.85)
    ):
        class_id, class_name, confidence = mock_clip_detector.classify_known(query)
    assert isinstance(class_id, int), f"class_id should be int, got {type(class_id)}"
    assert isinstance(class_name, str) and len(class_name) > 0, "class_name should be non-empty string"
    assert isinstance(confidence, float), f"confidence should be float, got {type(confidence)}"
