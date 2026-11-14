"""YOLO26-x transfer baseline: shared weak or native Ultralytics recipe."""
import argparse
from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--recipe', choices=['shared', 'native'], required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    kwargs = dict(data=str(root / 'data640/dataset.yaml'), epochs=50, patience=0,
        imgsz=640, batch=16, device=[0, 1], workers=4, seed=42,
        deterministic=True, amp=True, plots=False, save=True, val=True,
        project=str(root / 'runs'), name='yolo26x_' + a.recipe, exist_ok=False)
    if a.recipe == 'shared':
        kwargs.update(optimizer='AdamW', lr0=1e-4, lrf=0.1, cos_lr=True,
            weight_decay=1e-4, nbs=16, momentum=0.9,
            warmup_epochs=3, warmup_bias_lr=0., warmup_momentum=0.9,
            mosaic=0., mixup=0., cutmix=0., copy_paste=0.,
            hsv_h=0., hsv_s=0., hsv_v=0., degrees=0., translate=0., scale=0.,
            shear=0., perspective=0., flipud=0., fliplr=0.5, close_mosaic=0,
            multi_scale=False, rect=False, cache=False)
    else:
        kwargs.update(optimizer='auto')
    YOLO(a.checkpoint).train(**kwargs)
