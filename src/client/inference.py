"""
FedFOD Inference Module
=======================

RT-DETR-L inference engine, weather-aware false alarm filtering MLP,
and alert management for the FedFOD client.
"""

import math
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from ultralytics import RTDETR, YOLO
except ImportError:
    RTDETR = None
    YOLO = None

logger = logging.getLogger(__name__)

# Default FOD class names (15 classes)
DEFAULT_CLASS_NAMES: Dict[int, str] = {
    0: "metal_fastener",
    1: "wire_fragment",
    2: "rubber_gasket",
    3: "plastic_debris",
    4: "tool_wrench",
    5: "tool_screwdriver",
    6: "pavement_chunk",
    7: "luggage_fragment",
    8: "engine_blade",
    9: "safety_cone",
    10: "bird_remains",
    11: "ice_chunk",
    12: "tire_fragment",
    13: "light_cover",
    14: "unknown_fod",
}


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """Single object detection result."""

    bbox: np.ndarray  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    embedding: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# RTDETRDetector
# ---------------------------------------------------------------------------
class RTDETRDetector:
    """Ultralytics RT-DETR-L inference wrapper.

    Notes:
        * ``model.model`` is the underlying ``nn.Module`` with ``state_dict()``.
        * ``amp=False`` is enforced to avoid NaN issues in bipartite matching.
        * ``.clone()`` is used when loading weights to prevent autograd problems.
    """

    def __init__(
        self,
        model_path: str = "rtdetr-l.pt",
        num_classes: int = 15,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        class_names: Optional[Dict[int, str]] = None,
    ):
        self.model_path = model_path
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.class_names = class_names or DEFAULT_CLASS_NAMES

        # Check if the path points to a federated learning checkpoint
        is_fl_ckpt = False
        fl_weights = None
        
        # Load file with torch first to check if it's our FL checkpoint format
        if Path(model_path).is_file():
            try:
                ckpt = torch.load(model_path, map_location="cpu")
                if isinstance(ckpt, dict) and "weights" in ckpt:
                    is_fl_ckpt = True
                    fl_weights = ckpt["weights"]
            except Exception:
                pass

        if is_fl_ckpt and fl_weights:
            # Determine if it's YOLO or RT-DETR from weight key signatures
            # RT-DETR has keys containing 'encoder' or 'decoder'
            is_yolo = not any("encoder" in k or "decoder" in k for k in fl_weights.keys())
            base_model_path = "yolov8n.pt" if is_yolo else "rtdetr-l.pt"
            
            # Load the base model first
            if is_yolo:
                if YOLO is None:
                    raise ImportError("ultralytics is required for YOLO model")
                self.model = YOLO(base_model_path)
                self.model_type = "yolo"
            else:
                if RTDETR is None:
                    raise ImportError("ultralytics is required for RTDETR model")
                self.model = RTDETR(base_model_path)
                self.model_type = "rtdetr"
                
            # Convert weight state_dict values to torch.Tensor and load
            torch_state = {}
            for name, val in fl_weights.items():
                if isinstance(val, np.ndarray):
                    torch_state[name] = torch.from_numpy(val).clone()
                elif isinstance(val, torch.Tensor):
                    torch_state[name] = val.clone()
                else:
                    torch_state[name] = torch.tensor(val)
                    
            # Load weights state dict strict=False to ensure it works
            self.model.model.load_state_dict(torch_state, strict=False)
            logger.info("RTDETRDetector: loaded FL checkpoint %s onto base %s", model_path, base_model_path)
        else:
            # Robust loading with fallbacks
            loaded = False
            last_err = None

            if "yolo" in model_path.lower():
                if YOLO is not None:
                    try:
                        self.model = YOLO(model_path)
                        self.model_type = "yolo"
                        loaded = True
                    except Exception as e:
                        last_err = e
                if not loaded and RTDETR is not None:
                    try:
                        self.model = RTDETR(model_path)
                        self.model_type = "rtdetr"
                        loaded = True
                    except Exception as e:
                        last_err = e
            else:
                if RTDETR is not None:
                    try:
                        self.model = RTDETR(model_path)
                        self.model_type = "rtdetr"
                        loaded = True
                    except Exception as e:
                        last_err = e
                if not loaded and YOLO is not None:
                    try:
                        self.model = YOLO(model_path)
                        self.model_type = "yolo"
                        loaded = True
                    except Exception as e:
                        last_err = e

            if not loaded:
                raise RuntimeError(
                    f"Failed to load model from path '{model_path}' as YOLO or RTDETR. "
                    f"Underlying error: {last_err}"
                )

            logger.info(
                "RTDETRDetector loaded model=%s (type=%s) on device=%s",
                model_path,
                self.model_type,
                self.device,
            )

    # -- public API ----------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run single-frame inference and return parsed detections."""
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            half=False,  # amp=False equivalent
        )
        return self._parse_results(results)

    def detect_batch(
        self, frames: List[np.ndarray]
    ) -> List[List[Detection]]:
        """Run inference on a batch of frames."""
        all_detections: List[List[Detection]] = []
        results = self.model.predict(
            source=frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            half=False,
        )
        for result in results:
            all_detections.append(self._parse_results([result]))
        return all_detections

    def get_state_dict(self) -> Dict[str, np.ndarray]:
        """Return the model weights as a dict of ``str → np.ndarray``."""
        state_dict: Dict[str, np.ndarray] = {}
        for name, param in self.model.model.state_dict().items():
            state_dict[name] = param.clone().cpu().numpy()
        return state_dict

    def set_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        """Load weights from a ``{name: np.ndarray}`` dict.

        Uses ``.clone()`` to avoid autograd issues.
        """
        torch_state: Dict[str, torch.Tensor] = {}
        for name, arr in state_dict.items():
            torch_state[name] = torch.from_numpy(arr).clone()
        self.model.model.load_state_dict(torch_state, strict=False)
        logger.info("RTDETRDetector: loaded %d parameter tensors", len(torch_state))

    def warmup(self, imgsz: int = 640) -> None:
        """Run a dummy forward pass to warm up the model."""
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model.predict(
            source=dummy,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            half=False,
        )
        logger.info("RTDETRDetector warmup complete (imgsz=%d)", imgsz)

    # -- private helpers -----------------------------------------------------

    def _parse_results(self, results) -> List[Detection]:
        """Convert Ultralytics Results objects to a list of Detection."""
        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()   # (N, 4)
        confs = boxes.conf.cpu().numpy()  # (N,)
        cls_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)

        for i in range(len(xyxy)):
            cid = int(cls_ids[i])
            det = Detection(
                bbox=xyxy[i].copy(),
                class_id=cid,
                class_name=self.model.names.get(cid, self.class_names.get(cid, f"class_{cid}")),
                confidence=float(confs[i]),
            )
            detections.append(det)

        return detections


# ---------------------------------------------------------------------------
# FalseAlarmFilterMLP
# ---------------------------------------------------------------------------
class FalseAlarmFilterMLP(nn.Module):
    """4-layer MLP for filtering weather-induced false positive detections.

    Input feature vector (12-d):
        [confidence, bbox_area, bbox_aspect_ratio, x_center, y_center,
         rain_prob, fog_prob, glare_prob, hour_sin, hour_cos,
         luminance_mean, luminance_std]
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        self.device_target = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = input_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → probability that the detection is a *real* FOD."""
        return self.network(x)

    def filter_detections(
        self,
        detections: List[Detection],
        weather_ctx: Dict,
        frame: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """Run the MLP on each detection and keep those scored > 0.5."""
        if not detections:
            return []

        self.eval()
        features = torch.stack(
            [
                self._extract_features(det, weather_ctx, frame)
                for det in detections
            ]
        )
        device = next(self.parameters()).device
        features = features.to(device)

        with torch.no_grad():
            scores = self.forward(features).squeeze(-1)  # (N,)

        filtered: List[Detection] = []
        for det, score in zip(detections, scores):
            if score.item() > 0.5:
                filtered.append(det)

        return filtered

    def _extract_features(
        self,
        detection: Detection,
        weather_ctx: Dict,
        frame: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Build the 12-d feature tensor for a single detection."""
        x1, y1, x2, y2 = detection.bbox
        width = max(x2 - x1, 1e-6)
        height = max(y2 - y1, 1e-6)
        area = width * height
        aspect_ratio = width / height
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0

        # Weather features
        rain_prob = weather_ctx.get("rain_prob", 0.0)
        fog_prob = weather_ctx.get("fog_prob", 0.0)
        glare_prob = weather_ctx.get("glare_prob", 0.0)

        # Temporal features (hour encoded as sin/cos)
        hour = weather_ctx.get("hour", 12.0)
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)

        # Luminance features from the detection crop
        if frame is not None:
            crop = frame[
                max(0, int(y1)) : max(1, int(y2)),
                max(0, int(x1)) : max(1, int(x2)),
            ]
            if crop.size > 0:
                gray = crop.mean(axis=2) if crop.ndim == 3 else crop
                luminance_mean = float(gray.mean()) / 255.0
                luminance_std = float(gray.std()) / 255.0
            else:
                luminance_mean = weather_ctx.get("luminance_mean", 0.5)
                luminance_std = weather_ctx.get("luminance_std", 0.1)
        else:
            luminance_mean = weather_ctx.get("luminance_mean", 0.5)
            luminance_std = weather_ctx.get("luminance_std", 0.1)

        # Normalise spatial features (assume 640×640 image)
        area_norm = area / (640.0 * 640.0)
        x_center_norm = x_center / 640.0
        y_center_norm = y_center / 640.0

        features = [
            detection.confidence,
            area_norm,
            aspect_ratio,
            x_center_norm,
            y_center_norm,
            rain_prob,
            fog_prob,
            glare_prob,
            hour_sin,
            hour_cos,
            luminance_mean,
            luminance_std,
        ]

        return torch.tensor(features, dtype=torch.float32)

    def save_model(self, path: str) -> None:
        """Save MLP weights to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)
        logger.info("FalseAlarmFilterMLP saved to %s", save_path)

    def load_model(self, path: str) -> None:
        """Load MLP weights from disk."""
        load_path = Path(path)
        state = torch.load(load_path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        logger.info("FalseAlarmFilterMLP loaded from %s", load_path)


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------
class AlertManager:
    """Generate and manage FOD alerts with SLA compliance tracking."""

    # Urgency thresholds for RT-DETR detection confidence
    _URGENCY_THRESHOLDS: Dict[str, float] = {
        "CRITICAL": 0.90,
        "HIGH": 0.75,
        "MEDIUM": 0.55,
    }

    # Classes that are inherently high-risk (engine parts, large debris)
    _HIGH_RISK_CLASSES: set = {4, 5, 8, 12}  # wrench, screwdriver, engine blade, tire

    def __init__(self, sla_seconds: float = 45.0):
        self.sla_seconds = sla_seconds
        self._alert_history: List[Dict] = []

    def generate_alert(
        self,
        detection: Detection,
        track: object,
        weather_ctx: Dict,
    ) -> Dict:
        """Create a structured alert dictionary.

        Args:
            detection: The triggering detection.
            track: The associated tracked object (must have ``track_id``,
                ``dwell_time`` attributes).
            weather_ctx: Current weather context dict.
        """
        urgency = self._compute_urgency(detection)
        now = time.time()

        track_id = getattr(track, "track_id", -1)
        dwell_time = getattr(track, "dwell_time", 0.0)

        alert: Dict = {
            "alert_id": f"FOD-{int(now * 1000)}",
            "timestamp": now,
            "timestamp_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)
            ),
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": round(detection.confidence, 4),
            "bbox": detection.bbox.tolist(),
            "location": {
                "x_center": float((detection.bbox[0] + detection.bbox[2]) / 2),
                "y_center": float((detection.bbox[1] + detection.bbox[3]) / 2),
            },
            "urgency": urgency,
            "track_id": track_id,
            "dwell_time_seconds": round(dwell_time, 2),
            "weather": {
                "rain_prob": weather_ctx.get("rain_prob", 0.0),
                "fog_prob": weather_ctx.get("fog_prob", 0.0),
                "glare_prob": weather_ctx.get("glare_prob", 0.0),
            },
            "sla_deadline": now + self.sla_seconds,
        }

        self._alert_history.append(alert)
        return alert

    def check_sla_compliance(
        self, alert_timestamp: float, detection_timestamp: float
    ) -> bool:
        """Check whether the alert was generated within the SLA window."""
        latency = alert_timestamp - detection_timestamp
        return latency <= self.sla_seconds

    def _compute_urgency(self, detection: Detection) -> str:
        """Determine alert urgency from detection confidence and class.

        Returns one of ``'CRITICAL'``, ``'HIGH'``, ``'MEDIUM'``, ``'LOW'``.
        """
        conf = detection.confidence

        # Automatically escalate high-risk FOD classes
        if detection.class_id in self._HIGH_RISK_CLASSES:
            if conf >= self._URGENCY_THRESHOLDS["HIGH"]:
                return "CRITICAL"
            return "HIGH"

        if conf >= self._URGENCY_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        if conf >= self._URGENCY_THRESHOLDS["HIGH"]:
            return "HIGH"
        if conf >= self._URGENCY_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        return "LOW"
