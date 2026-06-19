#!/usr/bin/env python3
"""Train the FalseAlarmFilterMLP model on synthetic weather-FOD interaction data."""
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Insert project root into path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.client.inference import FalseAlarmFilterMLP


def generate_synthetic_data(num_samples: int = 6000, seed: int = 42):
    """Generate synthetic 12-d feature vectors and labels for training.
    
    Features (12-d):
      [0] confidence
      [1] area_norm
      [2] aspect_ratio
      [3] x_center_norm
      [4] y_center_norm
      [5] rain_prob
      [6] fog_prob
      [7] glare_prob
      [8] hour_sin
      [9] hour_cos
      [10] luminance_mean
      [11] luminance_std
    
    Labels:
      1.0 -> Real FOD (should be kept)
      0.0 -> Weather-induced False Alarm (should be filtered out)
    """
    np.random.seed(seed)
    features = []
    labels = []
    
    for _ in range(num_samples):
        # 50% chance of being a real FOD vs weather false positive
        is_real = np.random.choice([True, False])
        
        # Base parameters
        rain_prob = np.random.uniform(0.0, 1.0)
        fog_prob = np.random.uniform(0.0, 1.0)
        glare_prob = np.random.uniform(0.0, 1.0)
        hour = np.random.uniform(0.0, 24.0)
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        
        # Spatial placement (uniformly scattered across runway frame)
        x_center = np.random.uniform(0.1, 0.9)
        y_center = np.random.uniform(0.3, 0.8) # runway is usually bottom/middle
        
        if is_real:
            # --- TRUE FOD (Label 1) ---
            # Real debris has clean, distinct features and higher confidence
            confidence = np.random.uniform(0.45, 0.95)
            # Typically small relative size
            area_norm = np.random.uniform(0.0001, 0.015)
            # Aspect ratio close to 1.0 (screws, chunks, gaskets)
            aspect_ratio = np.random.uniform(0.4, 2.5)
            # Luminance depends on weather but is generally distinct from background
            if fog_prob > 0.6:
                lum_mean = np.random.uniform(0.3, 0.6)
                lum_std = np.random.uniform(0.04, 0.12)
            else:
                lum_mean = np.random.uniform(0.2, 0.7)
                lum_std = np.random.uniform(0.08, 0.30)
                
            features.append([
                confidence, area_norm, aspect_ratio, x_center, y_center,
                rain_prob, fog_prob, glare_prob, hour_sin, hour_cos,
                lum_mean, lum_std
            ])
            labels.append(1.0)
        else:
            # --- FALSE ALARM (Label 0) ---
            # False alarms are usually triggered by environmental reflections/noise
            # We model three primary false alarm causes:
            fa_type = np.random.choice(["reflection", "fog_noise", "sun_glare"])
            
            if fa_type == "reflection":
                # Puddles/wet tarmac reflections (high rain_prob, high local lum, high std)
                rain_prob = np.random.uniform(0.6, 1.0)
                confidence = np.random.uniform(0.15, 0.55) # Lower/medium confidence
                area_norm = np.random.uniform(0.01, 0.08)   # Larger reflections
                aspect_ratio = np.random.uniform(1.5, 6.0) # Elongated reflections
                lum_mean = np.random.uniform(0.65, 0.9)    # Bright reflection highlight
                lum_std = np.random.uniform(0.18, 0.35)
                
            elif fa_type == "fog_noise":
                # Fog low-contrast hallucination (high fog_prob, extremely low std)
                fog_prob = np.random.uniform(0.7, 1.0)
                confidence = np.random.uniform(0.15, 0.45)
                area_norm = np.random.uniform(0.005, 0.04)
                aspect_ratio = np.random.uniform(0.2, 5.0)
                lum_mean = np.random.uniform(0.4, 0.55)    # Gray/foggy
                lum_std = np.random.uniform(0.01, 0.06)    # Low contrast/std
                
            else: # sun_glare
                # Sun glare reflections (high glare, daytime, bright)
                glare_prob = np.random.uniform(0.6, 1.0)
                # Noon sun glare
                hour = np.random.uniform(10.0, 16.0)
                hour_sin = math.sin(2 * math.pi * hour / 24.0)
                hour_cos = math.cos(2 * math.pi * hour / 24.0)
                confidence = np.random.uniform(0.15, 0.60)
                area_norm = np.random.uniform(0.005, 0.06)
                aspect_ratio = np.random.uniform(2.0, 8.0) # Glare streaks
                lum_mean = np.random.uniform(0.75, 0.98)   # Bleached highlights
                lum_std = np.random.uniform(0.05, 0.20)
                
            features.append([
                confidence, area_norm, aspect_ratio, x_center, y_center,
                rain_prob, fog_prob, glare_prob, hour_sin, hour_cos,
                lum_mean, lum_std
            ])
            labels.append(0.0)
            
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32).unsqueeze(1)


def train_mlp():
    print("[*] Generating synthetic weather-aware FOD detection features...")
    X, y = generate_synthetic_data(num_samples=7000)
    
    # Split into train/validation
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    print(f"[+] Total samples: {len(X)} | Train: {len(X_train)} | Val: {len(X_val)}")
    
    # Initialize MLP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing FalseAlarmFilterMLP model on device: {device}")
    model = FalseAlarmFilterMLP(input_dim=12).to(device)
    
    # Loss & Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    epochs = 80
    batch_size = 128
    
    print("[*] Starting training loop...")
    best_val_loss = float("inf")
    best_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices].to(device), y_train[indices].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(indices)
            
        epoch_loss /= X_train.size(0)
        
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device))
            val_loss = criterion(val_outputs, y_val.to(device)).item()
            
            # Compute accuracy
            preds = (val_outputs >= 0.5).float()
            correct = (preds == y_val.to(device)).float().sum()
            accuracy = correct / y_val.size(0)
            
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy*100:.2f}%")
            
    # Load best weights
    model.load_state_dict(best_weights)
    
    # Save the model
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "mlp_filter.pt"
    
    torch.save(model.state_dict(), model_path)
    print(f"\n[+] Training completed! Saved best model weights to: {model_path}")
    
    # Print final validation metrics
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val.to(device))
        preds = (val_outputs >= 0.5).float().cpu().numpy()
        targets = y_val.numpy()
        
        tp = ((preds == 1.0) & (targets == 1.0)).sum()
        fp = ((preds == 1.0) & (targets == 0.0)).sum()
        fn = ((preds == 0.0) & (targets == 1.0)).sum()
        tn = ((preds == 0.0) & (targets == 0.0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"[*] Final Evaluation Results:")
        print(f"    - Accuracy:  {(tp + tn) / len(targets) * 100:.2f}%")
        print(f"    - Precision: {precision * 100:.2f}% (keep rate on real FODs)")
        print(f"    - Recall:    {recall * 100:.2f}%")
        print(f"    - F1-Score:  {f1:.4f}")
        print(f"    - False Alarm Rejection Rate: {tn / (tn + fp) * 100:.2f}% (weather reflections filtered)")


if __name__ == "__main__":
    train_mlp()
