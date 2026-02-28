# train.py
"""
YOLO11 Training Script for Fire Detection
==========================================
Train a YOLO11n model on custom fire detection dataset.

Usage:
    python train.py

Requirements:
    - ultralytics >= 8.1.0
    - NVIDIA GPU with CUDA support
    - Dataset in YOLO format (see forestfire.yaml)

Notes:
    - Train for at least 100 epochs for reliable detection
    - 10 epochs is NOT enough (mAP50≈0.54, too low for quantization)
    - Adjust batch size based on your GPU memory
"""
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # Load pretrained YOLO11n model
    model = YOLO("yolo11n.pt")

    # Start training
    model.train(
        data="forestfire.yaml",     # Path to dataset config
        cache=False,
        imgsz=640,                  # Input size (must match quantization config)
        epochs=100,                 # ★ At least 100 epochs!
        batch=60,                   # Adjust based on GPU memory
        close_mosaic=10,
        workers=1,
        device='0',
        optimizer='SGD',
        patience=0,                 # No early stopping
        project='results',
        name='fire-detect',
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("Best weights: results/fire-detect/weights/best.pt")
    print("=" * 60)
