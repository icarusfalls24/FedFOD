"""
FedFOD Client Module
====================

Exports core client-side components for federated FOD detection:
- RunwayMosaicIngestion: Multi-camera frame ingestion and preprocessing
- ByteTracker: Multi-object tracking with Kalman filter + Hungarian matching
- RTDETRDetector: RT-DETR-L inference engine
- FalseAlarmFilterMLP: Weather-aware false alarm filtering
- CLIPOpenWorldDetector: CLIP-based open-world FOD detection
- FedFODClient: Flower NumPyClient for federated learning
- ScaffoldTrainer: SCAFFOLD local training loop
"""

__all__ = [
    "RunwayMosaicIngestion",
    "ByteTracker",
    "RTDETRDetector",
    "FalseAlarmFilterMLP",
    "CLIPOpenWorldDetector",
    "FedFODClient",
    "ScaffoldTrainer",
]

try:
    from src.client.ingestion import RunwayMosaicIngestion, ByteTracker
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import ingestion components (RunwayMosaicIngestion, ByteTracker): {e}. "
        f"Install required dependencies: opencv-python, filterpy, scipy."
    )
    RunwayMosaicIngestion = None
    ByteTracker = None

try:
    from src.client.inference import RTDETRDetector, FalseAlarmFilterMLP
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import inference components (RTDETRDetector, FalseAlarmFilterMLP): {e}. "
        f"Install required dependencies: ultralytics, torch."
    )
    RTDETRDetector = None
    FalseAlarmFilterMLP = None

try:
    from src.client.open_world import CLIPOpenWorldDetector
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import open_world components (CLIPOpenWorldDetector): {e}. "
        f"Install required dependencies: clip, torch, Pillow."
    )
    CLIPOpenWorldDetector = None

try:
    from src.client.local_trainer import FedFODClient, ScaffoldTrainer
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import local_trainer components (FedFODClient, ScaffoldTrainer): {e}. "
        f"Install required dependencies: flwr, torch, ultralytics."
    )
    FedFODClient = None
    ScaffoldTrainer = None
