#!/usr/bin/env python3
"""FedFOD Single-Image Runway Predictor
========================================
Runs Foreign Object Debris (FOD) detection on a single airport runway image.
Loads the model, detects objects, applies false alarm filtering, and runs
open-world classification.

Usage:
    python predict.py --image data/airport_A/train/images/runway_001.jpg --model rtdetr-l.pt
    python predict.py --image test_runway.png --model yolov8n.pt --conf 0.25
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch

# ---- Project imports ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.client.inference import RTDETRDetector, FalseAlarmFilterMLP
from src.client.open_world import CLIPOpenWorldDetector

def main():
    parser = argparse.ArgumentParser(description="FedFOD Runway FOD Detection")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input airport runway image")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt",
                        help="Path to model weights (e.g. yolov8n.pt or rtdetr-l.pt)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold for detections")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (cuda | cpu)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Bypass the weather-aware False Alarm Filter (MLP)")
    parser.add_argument("--mlp-weights", type=str, default="checkpoints/mlp_filter.pt",
                        help="Path to pre-trained False Alarm Filter (MLP) weights")

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: Image file not found at '{args.image}'")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading model backbone '{args.model}' on {device}...")
    
    try:
        detector = RTDETRDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            device=device
        )
    except Exception as exc:
        print(f"Error loading detector model: {exc}")
        sys.exit(1)

    print(f"[*] Reading input image: {args.image}")
    frame = cv2.imread(args.image)
    if frame is None:
        print("Error: Could not read image using OpenCV.")
        sys.exit(1)

    # 1. Run detection (RT-DETR or YOLO)
    print("[*] Running core object detection...")
    detections = detector.detect(frame)
    print(f"[+] Detected {len(detections)} potential FOD items.")

    if not detections:
        print("No FOD objects detected on the runway.")
        return

    # 2. Weather-aware False Alarm Filtering (MLP)
    if args.no_filter:
        print("[*] Bypassing weather-aware False Alarm Filter (MLP)...")
        filtered_detections = detections
    else:
        print("[*] Applying weather-aware False Alarm Filter (MLP)...")
        # Simulate a typical dry, daytime context
        weather_ctx = {
            "rain_prob": 0.0,
            "fog_prob": 0.0,
            "glare_prob": 0.05,
            "hour": 14.0,
            "luminance_mean": 0.65,
            "luminance_std": 0.12
        }
        
        mlp = FalseAlarmFilterMLP(input_dim=12)
        if args.mlp_weights and os.path.isfile(args.mlp_weights):
            print(f"[*] Loading MLP weights from {args.mlp_weights}...")
            mlp.load_model(args.mlp_weights)
        else:
            print("[!] No trained MLP weights loaded; using initialized weights.")
            
        # Filter detections (keeping those scored > 0.5 as real FOD)
        filtered_detections = mlp.filter_detections(detections, weather_ctx, frame)
        print(f"[+] validated {len(filtered_detections)} / {len(detections)} detections after filtering.")

    if not filtered_detections:
        print("All detections filtered out as weather-induced false alarms.")
        return

    # 3. Open-World CLIP classification
    print("[*] Loading CLIP model (ViT-B/32) for open-world category validation...")
    try:
        clip_detector = CLIPOpenWorldDetector(device=device)
    except Exception as exc:
        print(f"Error loading CLIP: {exc}")
        sys.exit(1)

    # We will build a dummy prototype bank containing class representative embeddings
    # (In production, the client downloads the real prototype bank from the server)
    dummy_proto_bank = {}

    print("\n" + "="*80)
    print(f"{'Detected FOD Item':<20} | {'Confidence':<10} | {'Location (BBox)':<25} | {'CLIP Classification'}")
    print("="*80)
    
    for idx, det in enumerate(filtered_detections):
        x1, y1, x2, y2 = map(int, det.bbox)
        # Extract crop for CLIP open-world validation
        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        
        classification_str = det.class_name
        if crop.size > 0:
            try:
                # Compute CLIP embedding
                emb = clip_detector.compute_fod_embedding(crop)
                # Classify against known templates
                class_id, class_name, sim = clip_detector.classify_known(emb)
                
                # Check novelty
                is_novel, nearest_class, distance = clip_detector.detect_novelty(emb, dummy_proto_bank)
                
                if is_novel and len(dummy_proto_bank) > 0:
                    classification_str = f"Novel Class (Nearest: {nearest_class})"
                else:
                    classification_str = f"{class_name} (CLIP sim: {sim:.2f})"
            except Exception as e:
                classification_str = f"{det.class_name} (CLIP error: {e})"
        
        bbox_str = f"[{x1}, {y1}, {x2}, {y2}]"
        print(f"#{idx+1:<18} | {det.confidence:<10.2f} | {bbox_str:<25} | {classification_str}")
        
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
