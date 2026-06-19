#!/usr/bin/env python3
"""Local RT-DETR fine-tuning script for FedFOD.

Performs per-airport pre-training of RT-DETR-L on partitioned
FOD detection data before federated learning rounds.
"""

import argparse
import os
import sys
import yaml
import logging
import pathlib
import torch

try:
    from ultralytics import RTDETR
except ImportError:
    RTDETR = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FOD_CLASS_NAMES = [
    "bolt", "nut", "washer", "rivet", "wildlife", "wire",
    "tool", "fabric", "aircraft_part", "debris", "vehicle",
    "plastic", "rubber", "paper", "metal_sheet",
]

AUGMENT_PROFILES = {
    "weather_heavy": "rain:0.30,fog:0.20,glare:0.30",
    "weather_light": "rain:0.10,fog:0.05,glare:0.15",
    "none": None,
}


# ---------------------------------------------------------------------------
# Data YAML creation
# ---------------------------------------------------------------------------

def create_data_yaml(data_dir, num_classes: int, class_names: list, output_path):
    """Write a YOLO-format data.yaml for Ultralytics training.

    Args:
        data_dir: Root directory containing train/val/test splits.
        num_classes: Total number of object classes.
        class_names: Ordered list of class name strings.
        output_path: Where to write the yaml file.

    Returns:
        pathlib.Path to the written yaml file.
    """
    data_dir = pathlib.Path(data_dir)
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_config = {
        "path": str(data_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": num_classes,
        "names": list(class_names),
    }

    with open(output_path, "w") as f:
        yaml.safe_dump(data_config, f, default_flow_style=False, sort_keys=False)

    logger.info("Created data YAML at %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------

def pretrain(
    client_id: str,
    data_dir: str,
    epochs: int = 80,
    batch_size: int = 8,
    imgsz: int = 640,
    device: str = "cuda:0",
    lr0: float = 1e-4,
    warmup_epochs: int = 5,
    augment_profile: str = "none",
    project_dir: str = "runs/pretrain",
):
    """Run RT-DETR-L fine-tuning for a single federated client.

    Args:
        client_id: Unique identifier for this airport client.
        data_dir: Directory containing train/val/test image & label folders.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        imgsz: Input image size (square).
        device: Compute device (cuda:N or cpu).
        lr0: Initial learning rate.
        warmup_epochs: Number of linear-warmup epochs.
        augment_profile: Weather augmentation profile key.
        project_dir: Root directory for training runs.

    Returns:
        dict with client_id, best_checkpoint path, and results summary.
    """
    if RTDETR is None:
        logger.error(
            "ultralytics is not installed. Install with: pip install ultralytics"
        )
        sys.exit(1)

    data_dir = pathlib.Path(data_dir)
    project_dir = pathlib.Path(project_dir)

    # 1. Create data.yaml
    yaml_path = create_data_yaml(
        data_dir,
        num_classes=len(FOD_CLASS_NAMES),
        class_names=FOD_CLASS_NAMES,
        output_path=data_dir / "data.yaml",
    )

    # 2. Device fallback
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available – falling back to CPU")
        device = "cpu"

    # 3. Load model
    logger.info("Loading RT-DETR-L model…")
    model = RTDETR("rtdetr-l.pt")

    # 4. Log configuration
    logger.info("=" * 60)
    logger.info("RT-DETR Pre-training Configuration")
    logger.info("  Client:          %s", client_id)
    logger.info("  Data directory:  %s", data_dir)
    logger.info("  Epochs:          %d", epochs)
    logger.info("  Batch size:      %d", batch_size)
    logger.info("  Image size:      %d", imgsz)
    logger.info("  Device:          %s", device)
    logger.info("  Learning rate:   %s", lr0)
    logger.info("  Warmup epochs:   %d", warmup_epochs)
    logger.info("  Augment profile: %s", augment_profile)
    logger.info("  Project dir:     %s", project_dir)
    logger.info("=" * 60)

    # 5. Weather augmentation note
    if augment_profile and augment_profile != "none":
        profile_str = AUGMENT_PROFILES.get(augment_profile)
        if profile_str:
            logger.info(
                "Weather augmentation profile '%s' active (%s). "
                "Ensure augmented data was generated via augment.py.",
                augment_profile,
                profile_str,
            )
        else:
            logger.warning("Unknown augmentation profile '%s' – ignored", augment_profile)

    # 6. Train
    logger.info("Starting training…")
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        lr0=lr0,
        warmup_epochs=warmup_epochs,
        amp=False,
        project=str(project_dir),
        name=client_id,
        exist_ok=True,
        verbose=True,
        patience=20,
        save=True,
        save_period=10,
        workers=4,
        seed=42,
    )

    # 7. Locate best checkpoint
    best_path = project_dir / client_id / "weights" / "best.pt"
    last_path = project_dir / client_id / "weights" / "last.pt"

    if best_path.exists():
        logger.info("Best checkpoint saved at %s", best_path)
    elif last_path.exists():
        logger.info("Last checkpoint saved at %s (best not found)", last_path)
        best_path = last_path
    else:
        logger.warning("No checkpoint found under %s", project_dir / client_id / "weights")

    logger.info("Training complete for client '%s'", client_id)

    return {
        "client_id": client_id,
        "best_checkpoint": str(best_path),
        "results": str(results),
        "epochs": epochs,
        "device": device,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FedFOD – RT-DETR local pre-training")
    parser.add_argument("--client", type=str, required=True, help="Client identifier (e.g. airport_A)")
    parser.add_argument("--data", type=str, required=True, help="Path to client data directory")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs (default: 80)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: cuda:0, falls back to cpu)")
    parser.add_argument("--lr0", type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs (default: 5)")
    parser.add_argument(
        "--augment",
        type=str,
        choices=["weather_heavy", "weather_light", "none"],
        default="none",
        help="Weather augmentation profile (default: none)",
    )
    parser.add_argument("--project", type=str, default="runs/pretrain", help="Project directory for runs")
    args = parser.parse_args()

    result = pretrain(
        client_id=args.client,
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        lr0=args.lr0,
        warmup_epochs=args.warmup_epochs,
        augment_profile=args.augment,
        project_dir=args.project,
    )

    logger.info("Final result: %s", result)


if __name__ == "__main__":
    main()
