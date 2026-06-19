<p align="center">
  <h1 align="center">FedFOD — Federated Foreign Object Debris Detection</h1>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://flower.ai/"><img src="https://img.shields.io/badge/Flower-FL-green.svg" alt="Flower FL"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

## Overview

**FedFOD** is a privacy-preserving **Federated Learning (FL)** system for **Foreign Object Debris (FOD)** detection across multi-airport IoT camera networks. The system is designed to meet the stringent safety, latency, and privacy requirements of modern aviation operations and targets publication at **IEEE Transactions on Aerospace and Electronic Systems (T-AES)**.

FOD on runways and taxiways poses critical safety risks, causing an estimated US $13 billion in annual damages to the global aviation industry. FedFOD enables collaborative model training across geographically distributed airport camera networks **without sharing raw imagery**, preserving operational confidentiality while achieving state-of-the-art detection performance.

---

## Architecture

FedFOD integrates four key pillars into a unified federated detection pipeline:

### 1. RT-DETR-L Backbone (Real-Time Detection Transformer)
- Transformer-based object detector optimized for real-time inference on edge hardware.
- Operates at 640×640 input resolution with 15 FOD class categories.
- Achieves ≥ 0.79 mAP@50 on known classes with end-to-end alert latency < 45 seconds.

### 2. SCAFFOLD Aggregation (Stochastic Controlled Averaging)
- Addresses client drift in heterogeneous (non-IID) airport environments via control variates.
- Supports asynchronous participation with staleness-penalized aggregation for satellite-linked airports.
- Federated Averaging with weighted SCAFFOLD correction as the core aggregation strategy.

### 3. CLIP-Based Open-World Detection
- Integrates OpenAI CLIP (ViT-B/32) for zero-shot recognition of novel/unknown FOD categories.
- Maintains per-class prototypes with periodic federated updates.
- Enables ≥ 0.51 mAP@50 on novel, previously unseen debris classes.

### 4. Privacy-Preserving Mechanisms
- **Differential Privacy (DP):** Per-sample gradient clipping + Gaussian noise (ε=4.0, δ=1e-6) via Opacus.
- **Secure Multi-Party Computation (SMPC):** Shamir's Secret Sharing (3-of-2 threshold) for aggregation across ICAO, Consortium, and Neutral Party nodes.
- **Communication Efficiency:** Top-k sparsification (5%) + 8-bit quantization, ≤ 2 MB payload per round.

---

## Key Features

| Feature | Specification |
|---|---|
| Detection Model | RT-DETR-L (timm backbone) |
| FL Framework | Flower (flwr) with SCAFFOLD |
| FL Rounds | 90 rounds, 3 clients, min 2/round |
| Privacy | DP (ε=4.0) + SMPC (Shamir 3-of-2) |
| Open-World | CLIP ViT-B/32, cosine sim ≥ 0.70 |
| Communication | gRPC, ≤ 2 MB/round, 8-bit quant |
| mAP@50 (Known) | ≥ 0.79 |
| mAP@50 (Novel) | ≥ 0.51 |
| Alert Latency | < 45 seconds end-to-end |
| False Alarms | ≤ 2 per hour |

---

## FOD Classes

The system detects **15 categories** of Foreign Object Debris:

| ID | Class Name | Description |
|----|---|---|
| 0 | `metal_fastener` | Bolts, nuts, screws, rivets |
| 1 | `tyre_fragment` | Rubber tyre pieces from blowouts |
| 2 | `rubber_strip` | Rubber seals, gaskets, strips |
| 3 | `composite_debris` | Carbon-fibre / composite material fragments |
| 4 | `wildlife_bird` | Birds on or near the runway |
| 5 | `wildlife_mammal` | Mammals (rodents, hares, etc.) |
| 6 | `maintenance_tool` | Wrenches, screwdrivers, safety cones |
| 7 | `luggage_fragment` | Broken luggage, loose straps |
| 8 | `aircraft_panel_fragment` | Detached fuselage/wing panels |
| 9 | `pavement_fragment` | Concrete/asphalt chunks |
| 10 | `ground_equipment_part` | Parts from GSE vehicles |
| 11 | `volcanic_ash_deposit` | Ash accumulation from volcanic events |
| 12 | `ice_chunk` | Ice formed on aircraft or taxiway |
| 13 | `cargo_debris` | Loose cargo netting, pallet pieces |
| 14 | `unknown_critical` | Unrecognized but safety-critical objects |

---

## Quick Start

### 1. Prerequisites

- Python 3.9 or later
- CUDA 11.8+ (optional — CPU fallback is supported)
- Git

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/fedfod-research/fedfod.git
cd fedfod

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -e .
```

### 3. Download Data

Prepare your FOD dataset in COCO format and place it under the `data/` directory:

```
data/
├── airport_A/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── annotations/
│       ├── train.json
│       └── val.json
├── airport_B/
│   └── ...
└── airport_N/
    └── ...
```

### 4. Run Federated Simulation

```bash
# Launch the FL simulation (3 airports, 90 rounds)
python -m fedfod.simulation \
    --config config/global_config.yaml \
    --airports config/airport_configs/ \
    --num-rounds 90 \
    --device auto
```

### 5. Evaluate

```bash
# Evaluate the global model on all airport validation sets
python -m fedfod.evaluate \
    --config config/global_config.yaml \
    --checkpoint checkpoints/global_round_090.pt \
    --device auto
```

### 6. Experiment Tracking (Optional)

FedFOD supports optional [Weights & Biases](https://wandb.ai/) integration. If `wandb` is installed and configured, metrics are logged automatically. Otherwise, tracking is gracefully skipped.

```bash
wandb login  # one-time setup
```

---

## Project Structure

```
FedFOD/
├── config/
│   ├── global_config.yaml          # Global FL + model configuration
│   └── airport_configs/            # Per-airport configurations
│       ├── airport_A.yaml
│       ├── airport_B.yaml
│       └── airport_N.yaml
├── fedfod/
│   ├── __init__.py
│   ├── models/                     # RT-DETR, CLIP integration
│   ├── fl/                         # SCAFFOLD, aggregation strategies
│   ├── privacy/                    # DP (Opacus), SMPC (Shamir)
│   ├── communication/              # gRPC, sparsification, quantization
│   ├── data/                       # Dataset loaders, augmentations
│   ├── tracking/                   # Kalman filter-based object tracking
│   ├── evaluation/                 # mAP, latency, false-alarm metrics
│   └── utils/                      # Logging, config helpers
├── tests/                          # Unit and integration tests
├── scripts/                        # Utility scripts
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

---

## Citation

If you use FedFOD in your research, please cite:

```bibtex
@article{fedfod2025,
  title     = {{FedFOD}: Privacy-Preserving Federated Learning for Foreign Object
               Debris Detection across Multi-Airport IoT Camera Networks},
  author    = {{FedFOD Research Team}},
  journal   = {IEEE Transactions on Aerospace and Electronic Systems},
  year      = {2025},
  note      = {Under review}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO/RT-DETR framework
- [Flower](https://flower.ai/) for the federated learning infrastructure
- [OpenAI CLIP](https://github.com/openai/CLIP) for vision-language embeddings
- [Opacus](https://opacus.ai/) for differential privacy in PyTorch
