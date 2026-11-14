"""Matched weak-augmentation transfer run; use an isolated Ultralytics environment."""
import argparse
from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--model',choices=['yolo26x','yolo11x'],required=True)
    a=p.parse_args(); root=Path(a.root)
    model=YOLO(str(root/(a.model+'.pt')))
    model.train(data=str(root/'data640/dataset.yaml'), epochs=50, patience=0,
        imgsz=640, batch=32, device=[0,1], workers=6, seed=42,
        deterministic=True, optimizer='AdamW', lr0=1e-4, lrf=0.1,
        cos_lr=True, weight_decay=1e-4, nbs=32, momentum=0.9,
        warmup_epochs=3, warmup_bias_lr=0.0, warmup_momentum=0.9,
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0,
        translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, close_mosaic=0, multi_scale=False,
        rect=False, cache=False, amp=True, plots=False,
        project=str(root/'runs'), name=a.model, exist_ok=False,
        save=True, val=True)
