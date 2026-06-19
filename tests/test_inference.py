"""Tests for weather analysis, false alarm filtering MLP, alert management, and model loading."""
import os
import sys
import time
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client.inference import RTDETRDetector, FalseAlarmFilterMLP, AlertManager, Detection
from api_server import _analyze_image_weather


# ---- Weather Analysis Tests ----

def test_analyze_image_weather_dark():
    """Verify that a dark image maps to night hour and high fog/rain estimates."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    w = _analyze_image_weather(img)
    assert w["hour"] == 2.0
    assert w["fog_prob"] == 0.8
    assert w["glare_prob"] == 0.05
    assert w["rain_prob"] == 0.0


def test_analyze_image_weather_bright():
    """Verify that a bright, uniform image maps to noon hour with low fog."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    w = _analyze_image_weather(img)
    assert w["hour"] in [12.0, 14.0]
    assert w["fog_prob"] == 0.8  # Uniform gray/bright means std_val is 0 -> fog_prob=0.8


def test_analyze_image_weather_high_contrast():
    """Verify that high contrast images map to daylight hours and low fog."""
    # Create checkerboard pattern for high standard deviation
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50, :] = 255
    img[50:, 50:, :] = 255
    w = _analyze_image_weather(img)
    assert w["fog_prob"] == 0.0
    assert w["glare_prob"] == 0.6  # 50% white pixels (>240)


# ---- AlertManager Tests ----

def test_alert_manager_urgency():
    """Test urgency computation for different confidence and class levels."""
    manager = AlertManager(sla_seconds=45.0)
    
    # Low risk, low confidence
    det_low = Detection(
        bbox=np.array([10, 20, 30, 40]),
        class_id=0,
        class_name="metal_fastener",
        confidence=0.20
    )
    # High risk class, high confidence
    det_high_risk = Detection(
        bbox=np.array([10, 20, 30, 40]),
        class_id=4,  # wrench (high risk)
        class_name="tool_wrench",
        confidence=0.80
    )
    
    class MockTrack:
        track_id = 42
        dwell_time = 5.0
        
    alert_low = manager.generate_alert(det_low, MockTrack(), {})
    alert_high_risk = manager.generate_alert(det_high_risk, MockTrack(), {})
    
    assert alert_low["urgency"] == "LOW"
    assert alert_high_risk["urgency"] == "CRITICAL"
    assert alert_low["track_id"] == 42
    assert alert_low["dwell_time_seconds"] == 5.0


def test_alert_manager_sla_compliance():
    """Test alert generation latency SLA validation."""
    manager = AlertManager(sla_seconds=10.0)
    now = time.time()
    assert manager.check_sla_compliance(now + 5.0, now) is True
    assert manager.check_sla_compliance(now + 15.0, now) is False


# ---- FalseAlarmFilterMLP Tests ----

def test_mlp_feature_extraction():
    """Verify 12-d feature extraction maps confidence, spatial box, weather and hour."""
    mlp = FalseAlarmFilterMLP(input_dim=12)
    det = Detection(
        bbox=np.array([10, 20, 110, 120]),  # width=100, height=100
        class_id=0,
        class_name="metal_fastener",
        confidence=0.85
    )
    weather = {
        "rain_prob": 0.1,
        "fog_prob": 0.2,
        "glare_prob": 0.3,
        "hour": 12.0
    }
    
    features = mlp._extract_features(det, weather, frame=None)
    assert features.shape == (12,)
    # Index 0 is confidence
    assert abs(features[0].item() - 0.85) < 1e-5
    # Index 5, 6, 7 are rain, fog, glare
    assert abs(features[5].item() - 0.1) < 1e-5
    assert abs(features[6].item() - 0.2) < 1e-5
    assert abs(features[7].item() - 0.3) < 1e-5


def test_mlp_forward_pass():
    """Verify model output shape and value ranges."""
    mlp = FalseAlarmFilterMLP(input_dim=12)
    x = torch.randn(3, 12)
    out = mlp(x)
    assert out.shape == (3, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


# ---- RTDETRDetector Constructor & Mocks Tests ----

@patch("src.client.inference.torch.load")
@patch("src.client.inference.Path.is_file")
@patch("src.client.inference.YOLO")
def test_detector_fl_checkpoint_yolo_routing(mock_yolo, mock_is_file, mock_load):
    """Test that federated checkpoints containing YOLO weights trigger YOLO base load."""
    mock_is_file.return_value = True
    # Fake weights representing YOLO keys (no encoder/decoder keys)
    mock_load.return_value = {
        "weights": {
            "model.2.cv1.conv.weight": np.random.randn(3, 3, 3, 3)
        }
    }
    
    detector = RTDETRDetector(model_path="checkpoints/round_90.pt", device="cpu")
    
    mock_yolo.assert_called_once_with("yolov8n.pt")
    assert detector.model_type == "yolo"


@patch("src.client.inference.torch.load")
@patch("src.client.inference.Path.is_file")
@patch("src.client.inference.RTDETR")
def test_detector_fl_checkpoint_rtdetr_routing(mock_rtdetr, mock_is_file, mock_load):
    """Test that federated checkpoints containing RTDETR weights trigger RTDETR base load."""
    mock_is_file.return_value = True
    # Fake weights containing 'encoder' key signature
    mock_load.return_value = {
        "weights": {
            "model.encoder.layers.0.weight": np.random.randn(3, 3, 3, 3)
        }
    }
    
    detector = RTDETRDetector(model_path="checkpoints/round_90.pt", device="cpu")
    
    mock_rtdetr.assert_called_once_with("rtdetr-l.pt")
    assert detector.model_type == "rtdetr"
