#!/usr/bin/env python3
"""False Alarm Filter MLP training for FedFOD.

Trains a lightweight MLP classifier to distinguish real FOD detections
from false alarms based on detection metadata and environmental context.
"""

import argparse
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
import pathlib
from torch.utils.data import Dataset, DataLoader, random_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "confidence", "bbox_area", "aspect_ratio",
    "x_center", "y_center",
    "rain_prob", "fog_prob", "glare_prob",
    "hour_sin", "hour_cos",
    "luminance_mean", "luminance_std",
]
LABEL_COLUMN = "is_real_fod"


class FalseAlarmDataset(Dataset):
    """Detection-level dataset for the false-alarm filter MLP.

    If the CSV file does not exist, a synthetic dataset is generated
    automatically so that the training pipeline can run end-to-end
    without real inference outputs.
    """

    def __init__(self, csv_path: str):
        csv_path = pathlib.Path(csv_path)

        if csv_path.exists():
            logger.info("Loading FA filter data from %s", csv_path)
            self.data = pd.read_csv(csv_path)
        else:
            logger.warning("CSV not found at %s – generating synthetic training data", csv_path)
            self.data = self._generate_synthetic(n_samples=2000)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.data.to_csv(csv_path, index=False)
            logger.info("Saved synthetic FA data to %s", csv_path)

        self.features = torch.tensor(
            self.data[FEATURE_COLUMNS].values, dtype=torch.float32
        )
        self.labels = torch.tensor(
            self.data[LABEL_COLUMN].values, dtype=torch.float32
        ).unsqueeze(1)

    @staticmethod
    def _generate_synthetic(n_samples: int = 2000) -> pd.DataFrame:
        """Generate synthetic false-alarm vs real-FOD samples."""
        rng = np.random.default_rng(42)
        confidence = rng.uniform(0.10, 0.99, n_samples)
        bbox_area = rng.uniform(0.001, 0.30, n_samples)
        aspect_ratio = rng.uniform(0.2, 5.0, n_samples)
        x_center = rng.uniform(0.0, 1.0, n_samples)
        y_center = rng.uniform(0.0, 1.0, n_samples)
        rain_prob = rng.uniform(0.0, 0.50, n_samples)
        fog_prob = rng.uniform(0.0, 0.50, n_samples)
        glare_prob = rng.uniform(0.0, 0.50, n_samples)
        hour_angle = rng.uniform(0, 2 * math.pi, n_samples)
        hour_sin = np.sin(hour_angle)
        hour_cos = np.cos(hour_angle)
        luminance_mean = rng.uniform(50, 200, n_samples)
        luminance_std = rng.uniform(10, 60, n_samples)

        # Label: likely real FOD if high confidence and reasonable bbox area
        base_label = ((confidence > 0.5) & (bbox_area > 0.01)).astype(np.float64)
        # Add ~10 % label noise
        flip_mask = rng.random(n_samples) < 0.10
        noisy_label = np.where(flip_mask, 1.0 - base_label, base_label)

        return pd.DataFrame({
            "confidence": confidence,
            "bbox_area": bbox_area,
            "aspect_ratio": aspect_ratio,
            "x_center": x_center,
            "y_center": y_center,
            "rain_prob": rain_prob,
            "fog_prob": fog_prob,
            "glare_prob": glare_prob,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "luminance_mean": luminance_mean,
            "luminance_std": luminance_std,
            "is_real_fod": noisy_label.astype(int),
        })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_fa_mlp(input_dim: int = 12) -> nn.Sequential:
    """Build a small MLP for false-alarm classification."""
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_fa_filter(data_path: str, epochs: int = 50, batch_size: int = 64,
                    lr: float = 0.001, device: str = "cuda:0",
                    output_path: str = "checkpoints/fa_filter.pt") -> dict:
    """Train the false-alarm filter MLP.

    Returns a dict with best validation loss, accuracy, and model path.
    """
    # Device fallback
    if "cuda" in device and not torch.cuda.is_available():
        logger.warning("CUDA not available – falling back to CPU")
        device = "cpu"
    device = torch.device(device)

    # Data
    dataset = FalseAlarmDataset(data_path)
    n_val = int(0.20 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model / loss / optimiser
    model = build_fa_mlp(input_dim=len(FEATURE_COLUMNS)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(feats)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * feats.size(0)
            predicted = (preds >= 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.numel()

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                preds = model(feats)
                loss = criterion(preds, labels)

                val_loss += loss.item() * feats.size(0)
                predicted = (preds >= 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.numel()

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(output_path))

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d  train_loss=%.4f  train_acc=%.4f  "
                "val_loss=%.4f  val_acc=%.4f",
                epoch, epochs, train_loss, train_acc, val_loss, val_acc,
            )

    logger.info("Best val loss: %.4f  |  Best val acc: %.4f", best_val_loss, best_val_acc)
    logger.info("Model saved to %s", output_path)

    return {
        "best_val_loss": round(best_val_loss, 5),
        "best_val_acc": round(best_val_acc, 5),
        "model_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FedFOD – False Alarm Filter MLP training")
    parser.add_argument("--data", type=str, default="data/fa_filter_data.csv",
                        help="Path to CSV (auto-generated if missing)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (falls back to cpu if CUDA unavailable)")
    parser.add_argument("--output", type=str, default="checkpoints/fa_filter.pt",
                        help="Output path for best model checkpoint")
    args = parser.parse_args()

    result = train_fa_filter(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        output_path=args.output,
    )
    logger.info("Training result: %s", result)


if __name__ == "__main__":
    main()
