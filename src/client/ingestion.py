"""
FedFOD Ingestion Module
=======================

Runway camera frame ingestion pipeline with multi-camera mosaic composition,
weather adaptation, and multi-object tracking via ByteTrack-style tracker
with Kalman filtering and Hungarian matching.
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter
except ImportError:
    FilterPyKalmanFilter = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FrameBuffer
# ---------------------------------------------------------------------------
class FrameBuffer:
    """Ring buffer for storing recent video frames using deque."""

    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)

    def push(self, frame: np.ndarray) -> None:
        """Add a frame to the ring buffer."""
        self._buffer.append(frame.copy())

    def get_recent(self, n: int = 5) -> List[np.ndarray]:
        """Return the *n* most recent frames (newest last)."""
        n = min(n, len(self._buffer))
        return [self._buffer[-(n - i)] for i in range(n)]

    def get_temporal_average(self) -> np.ndarray:
        """Compute a temporal average of all buffered frames (background estimation).

        Returns a float32 image averaged across the buffer.  If the buffer is
        empty a single-pixel black image is returned to avoid downstream errors.
        """
        if len(self._buffer) == 0:
            return np.zeros((1, 1, 3), dtype=np.float32)
        accumulator = np.zeros_like(self._buffer[0], dtype=np.float64)
        for frame in self._buffer:
            accumulator += frame.astype(np.float64)
        averaged = (accumulator / len(self._buffer)).astype(np.float32)
        return averaged

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# RunwayMosaicIngestion
# ---------------------------------------------------------------------------
class RunwayMosaicIngestion:
    """Multi-camera runway mosaic composition and preprocessing."""

    def __init__(
        self,
        camera_configs: List[Dict],
        target_size: Tuple[int, int] = (640, 640),
    ):
        """
        Args:
            camera_configs: List of dicts, each with keys like
                ``camera_id``, ``roi`` (x,y,w,h), ``weight``, etc.
            target_size: (width, height) of the final output tensor.
        """
        self.camera_configs = camera_configs
        self.target_w, self.target_h = target_size
        self.num_cameras = len(camera_configs)
        self._frame_buffers: Dict[str, FrameBuffer] = {
            cfg.get("camera_id", str(i)): FrameBuffer()
            for i, cfg in enumerate(camera_configs)
        }

    # -- public API ----------------------------------------------------------

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalise, and letterbox a single frame to target_size."""
        letterboxed, _ratio, _pad = self._letterbox(
            frame, new_shape=(self.target_h, self.target_w)
        )
        # Normalise to [0, 1] float32
        normalized = letterboxed.astype(np.float32) / 255.0
        return normalized

    def compose_mosaic(self, frames: List[np.ndarray]) -> np.ndarray:
        """Stitch *frames* from multiple cameras into a single mosaic image.

        The layout is a 2×N grid where N = ceil(num_frames / 2).
        Each sub-tile is resized to fit the overall target_size evenly.
        """
        n = len(frames)
        if n == 0:
            return np.full(
                (self.target_h, self.target_w, 3), 114, dtype=np.uint8
            )
        if n == 1:
            return cv2.resize(frames[0], (self.target_w, self.target_h))

        # Determine grid layout
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        tile_w = self.target_w // cols
        tile_h = self.target_h // rows

        mosaic = np.full(
            (self.target_h, self.target_w, 3), 114, dtype=np.uint8
        )

        for idx, frame in enumerate(frames):
            r, c = divmod(idx, cols)
            resized = cv2.resize(frame, (tile_w, tile_h))
            y_start = r * tile_h
            x_start = c * tile_w
            mosaic[y_start : y_start + tile_h, x_start : x_start + tile_w] = (
                resized
            )

        return mosaic

    def apply_weather_adaptation(
        self, frame: np.ndarray, weather_ctx: Dict
    ) -> np.ndarray:
        """Adjust exposure / contrast based on weather context.

        ``weather_ctx`` may contain:
        - ``rain_prob``: 0-1 probability of rain → increase contrast
        - ``fog_prob``: 0-1 probability of fog → apply CLAHE
        - ``glare_prob``: 0-1 probability of glare → reduce brightness
        - ``luminance_mean``: ambient luminance metric
        """
        adapted = frame.copy()

        # --- Fog: histogram equalisation via CLAHE ---
        fog_prob = weather_ctx.get("fog_prob", 0.0)
        if fog_prob > 0.3:
            lab = cv2.cvtColor(adapted, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clip_limit = 2.0 + fog_prob * 4.0  # stronger for denser fog
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit, tileGridSize=(8, 8)
            )
            l_channel = clahe.apply(l_channel)
            merged = cv2.merge([l_channel, a_channel, b_channel])
            adapted = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # --- Rain: sharpen + contrast boost ---
        rain_prob = weather_ctx.get("rain_prob", 0.0)
        if rain_prob > 0.3:
            alpha = 1.0 + rain_prob * 0.5  # contrast factor
            beta = 10  # brightness offset
            adapted = cv2.convertScaleAbs(adapted, alpha=alpha, beta=beta)

        # --- Glare: reduce overexposed regions ---
        glare_prob = weather_ctx.get("glare_prob", 0.0)
        if glare_prob > 0.3:
            hsv = cv2.cvtColor(adapted, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            reduction = int(glare_prob * 60)
            v = np.clip(v.astype(np.int16) - reduction, 0, 255).astype(
                np.uint8
            )
            hsv_merged = cv2.merge([h, s, v])
            adapted = cv2.cvtColor(hsv_merged, cv2.COLOR_HSV2BGR)

        # --- Low-light: increase brightness ---
        luminance_mean = weather_ctx.get("luminance_mean", 128)
        if luminance_mean < 60:
            gamma = max(0.5, luminance_mean / 128.0)
            inv_gamma = 1.0 / gamma
            table = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
            ).astype(np.uint8)
            adapted = cv2.LUT(adapted, table)

        return adapted

    # -- private helpers -----------------------------------------------------

    def _letterbox(
        self,
        image: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """Letterbox an image to *new_shape* preserving aspect ratio.

        Returns:
            (letterboxed_image, (ratio_w, ratio_h), (pad_w, pad_h))
        """
        h, w = image.shape[:2]
        target_h, target_w = new_shape

        ratio = min(target_w / w, target_h / h)
        new_unpad_w = int(round(w * ratio))
        new_unpad_h = int(round(h * ratio))

        pad_w = (target_w - new_unpad_w) // 2
        pad_h = (target_h - new_unpad_h) // 2

        if (w, h) != (new_unpad_w, new_unpad_h):
            resized = cv2.resize(
                image,
                (new_unpad_w, new_unpad_h),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            resized = image

        top, bottom = pad_h, target_h - new_unpad_h - pad_h
        left, right = pad_w, target_w - new_unpad_w - pad_w
        letterboxed = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        return letterboxed, (ratio, ratio), (pad_w, pad_h)


# ---------------------------------------------------------------------------
# TrackedObject
# ---------------------------------------------------------------------------
@dataclass
class TrackedObject:
    """Represents a single tracked object across frames."""

    track_id: int
    bbox: np.ndarray  # (x1, y1, x2, y2)
    class_id: int = 0
    confidence: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    dwell_time: float = 0.0
    kalman_state: Optional[np.ndarray] = None
    is_fod_candidate: bool = False
    _lost_count: int = field(default=0, repr=False)
    _kalman_filter: object = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# ByteTracker
# ---------------------------------------------------------------------------
class ByteTracker:
    """Simplified ByteTrack-style multi-object tracker.

    Uses a constant-velocity Kalman filter for state prediction and the
    Hungarian algorithm (via ``scipy.optimize.linear_sum_assignment``) for
    detection-to-track association based on IoU cost.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        dwell_threshold_sec: float = 30.0,
        max_lost: int = 30,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.dwell_threshold_sec = dwell_threshold_sec
        self.max_lost = max_lost

        self.tracks: List[TrackedObject] = []
        self._next_id: int = 1

    # -- public API ----------------------------------------------------------

    def update(
        self, detections: List[Dict], timestamp: float
    ) -> List[TrackedObject]:
        """Associate new *detections* with existing tracks.

        Each detection dict must contain at least:
        ``bbox`` (x1,y1,x2,y2), ``confidence``, and optionally ``class_id``.
        """
        # 1. Predict existing tracks
        for track in self.tracks:
            predicted_bbox = self._predict_kalman(track)
            track.bbox = predicted_bbox

        # 2. Filter detections by confidence threshold
        high_dets = [d for d in detections if d.get("confidence", 0) >= self.track_thresh]
        low_dets = [d for d in detections if d.get("confidence", 0) < self.track_thresh]

        # 3. Compute IoU cost matrix
        if self.tracks and high_dets:
            cost_matrix = self._compute_iou_matrix(self.tracks, high_dets)
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_dets = set()

            for r, c in zip(row_indices, col_indices):
                if cost_matrix[r, c] < (1.0 - self.match_thresh):
                    continue  # IoU too low → skip
                track = self.tracks[r]
                det = high_dets[c]
                self._update_kalman(track, np.array(det["bbox"], dtype=np.float64))
                track.bbox = np.array(det["bbox"], dtype=np.float64)
                track.confidence = det.get("confidence", track.confidence)
                track.class_id = det.get("class_id", track.class_id)
                track.last_seen = timestamp
                track._lost_count = 0
                matched_tracks.add(r)
                matched_dets.add(c)

            # Unmatched tracks from high detections — try matching with low dets
            unmatched_track_ids = [
                i for i in range(len(self.tracks)) if i not in matched_tracks
            ]
            unmatched_det_ids = [
                i for i in range(len(high_dets)) if i not in matched_dets
            ]
        else:
            unmatched_track_ids = list(range(len(self.tracks)))
            unmatched_det_ids = list(range(len(high_dets)))
            matched_dets = set()

        # Second-round matching with low confidence detections for remaining tracks
        if unmatched_track_ids and low_dets:
            remaining_tracks = [self.tracks[i] for i in unmatched_track_ids]
            cost_matrix_low = self._compute_iou_matrix(remaining_tracks, low_dets)
            row_idx_low, col_idx_low = linear_sum_assignment(cost_matrix_low)

            newly_matched = set()
            for r, c in zip(row_idx_low, col_idx_low):
                if cost_matrix_low[r, c] < (1.0 - self.match_thresh):
                    continue
                orig_idx = unmatched_track_ids[r]
                track = self.tracks[orig_idx]
                det = low_dets[c]
                self._update_kalman(track, np.array(det["bbox"], dtype=np.float64))
                track.bbox = np.array(det["bbox"], dtype=np.float64)
                track.confidence = det.get("confidence", track.confidence)
                track.class_id = det.get("class_id", track.class_id)
                track.last_seen = timestamp
                track._lost_count = 0
                newly_matched.add(orig_idx)

            unmatched_track_ids = [
                i for i in unmatched_track_ids if i not in newly_matched
            ]

        # 5. Create new tracks for unmatched high detections
        for idx in unmatched_det_ids:
            det = high_dets[idx]
            bbox = np.array(det["bbox"], dtype=np.float64)
            kf = self._init_kalman(bbox)
            new_track = TrackedObject(
                track_id=self._next_id,
                bbox=bbox,
                class_id=det.get("class_id", 0),
                confidence=det.get("confidence", 0.0),
                first_seen=timestamp,
                last_seen=timestamp,
                dwell_time=0.0,
                kalman_state=kf.x.flatten().copy() if kf else bbox.copy(),
                is_fod_candidate=False,
                _lost_count=0,
                _kalman_filter=kf,
            )
            self.tracks.append(new_track)
            self._next_id += 1

        # 6. Mark unmatched tracks as lost
        for idx in unmatched_track_ids:
            self.tracks[idx]._lost_count += 1

        # 7. Remove tracks lost for > max_lost frames
        self.tracks = [
            t for t in self.tracks if t._lost_count <= self.max_lost
        ]

        # 8-9. Update dwell_time and FOD candidacy
        for track in self.tracks:
            track.dwell_time = track.last_seen - track.first_seen
            if track.dwell_time > self.dwell_threshold_sec:
                track.is_fod_candidate = True

        return list(self.tracks)

    def get_fod_candidates(self) -> List[TrackedObject]:
        """Return only tracks whose dwell_time exceeds the threshold."""
        return [
            t
            for t in self.tracks
            if t.dwell_time > self.dwell_threshold_sec
        ]

    # -- Kalman helpers ------------------------------------------------------

    def _init_kalman(self, bbox: np.ndarray) -> object:
        """Create a Kalman filter for a new track.

        State: [x_center, y_center, width, height, vx, vy, vw, vh]
        Measurement: [x_center, y_center, width, height]
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1

        if FilterPyKalmanFilter is not None:
            kf = FilterPyKalmanFilter(dim_x=8, dim_z=4)
            # State transition (constant velocity)
            kf.F = np.eye(8)
            kf.F[0, 4] = 1.0
            kf.F[1, 5] = 1.0
            kf.F[2, 6] = 1.0
            kf.F[3, 7] = 1.0
            # Measurement matrix
            kf.H = np.zeros((4, 8))
            kf.H[0, 0] = 1.0
            kf.H[1, 1] = 1.0
            kf.H[2, 2] = 1.0
            kf.H[3, 3] = 1.0
            # Covariances
            kf.P *= 10.0
            kf.P[4:, 4:] *= 100.0
            kf.R *= 1.0
            kf.Q = np.eye(8) * 0.01
            kf.Q[4:, 4:] *= 0.1
            # Initial state
            kf.x[:4] = np.array([[cx], [cy], [w], [h]])
            kf.x[4:] = 0.0
            return kf

        # Fallback: return None when filterpy is unavailable
        logger.warning("filterpy not installed – Kalman filtering disabled")
        return None

    def _predict_kalman(self, track: TrackedObject) -> np.ndarray:
        """Predict the next bounding box for *track*."""
        kf = track._kalman_filter
        if kf is not None:
            kf.predict()
            cx, cy, w, h = kf.x[:4].flatten()
            predicted = np.array(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                dtype=np.float64,
            )
            track.kalman_state = kf.x.flatten().copy()
            return predicted
        return track.bbox.copy()

    def _update_kalman(
        self, track: TrackedObject, detection_bbox: np.ndarray
    ) -> None:
        """Update the Kalman filter with a matched detection."""
        kf = track._kalman_filter
        if kf is not None:
            x1, y1, x2, y2 = detection_bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = x2 - x1, y2 - y1
            kf.update(np.array([[cx], [cy], [w], [h]]))
            track.kalman_state = kf.x.flatten().copy()

    # -- IoU helpers ---------------------------------------------------------

    def _compute_iou_matrix(
        self, tracks: List[TrackedObject], detections: List[Dict]
    ) -> np.ndarray:
        """Compute an IoU-based cost matrix (1 - IoU) for Hungarian matching."""
        n_tracks = len(tracks)
        n_dets = len(detections)
        cost = np.ones((n_tracks, n_dets), dtype=np.float64)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                det_bbox = np.array(det["bbox"], dtype=np.float64)
                iou_val = self._iou(track.bbox, det_bbox)
                cost[i, j] = 1.0 - iou_val
        return cost

    @staticmethod
    def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection-over-Union of two bboxes (x1,y1,x2,y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return inter / union


# ---------------------------------------------------------------------------
# DwellTimeFilter
# ---------------------------------------------------------------------------
class DwellTimeFilter:
    """Thin wrapper that filters tracked objects by dwell time."""

    def __init__(self, threshold_seconds: float = 30.0):
        self.threshold_seconds = threshold_seconds

    def filter(
        self, tracked_objects: List[TrackedObject]
    ) -> List[TrackedObject]:
        """Return only objects whose dwell_time exceeds the threshold."""
        return [
            obj
            for obj in tracked_objects
            if obj.dwell_time > self.threshold_seconds
        ]
