"""Secondary baseline: standard Ultralytics augmentation and automatic optimizer."""
import argparse
from pathlib import Path
from ultralytics import YOLO

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--model',choices=['yolo26x','yolo11x'],required=True)
    a=p.parse_args();root=Path(a.root)
    YOLO(str(root/(a.model+'.pt'))).train(
        data=str(root/'data640/dataset.yaml'),epochs=50,patience=0,
        imgsz=640,batch=32,device=[0,1],workers=6,seed=42,
        deterministic=True,optimizer='auto',amp=True,plots=False,
        project=str(root/'runs'),name=a.model+'_native',exist_ok=False,
        save=True,val=True)
