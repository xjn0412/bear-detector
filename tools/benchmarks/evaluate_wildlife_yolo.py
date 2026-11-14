"""Evaluate an Ultralytics checkpoint with the same COCO evaluator as Bear."""
import argparse,json,hashlib
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO
from faster_coco_eval import COCO,COCOeval_faster

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for key in ['checkpoint','annotations','images','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='0');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)
    gt=COCO(a.annotations);model=YOLO(a.checkpoint);preds=[]
    parameters_before_fusion=sum(p.numel() for p in model.model.parameters())
    ims=sorted(gt.dataset['images'],key=lambda x:x['id'])
    for start in range(0,len(ims),16):
        batch=ims[start:start+16]
        results=model.predict([str(Path(a.images)/im['file_name']) for im in batch],
            imgsz=640,conf=1e-5,iou=0.7,max_det=300,device=a.device,
            half=False,augment=False,verbose=False,rect=False,batch=16,nms=None)
        for im,result in zip(batch,results):
            for box,score,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()):
                x1,y1,x2,y2=box
                preds.append(dict(image_id=im['id'],category_id=int(cls),bbox=[x1,y1,x2-x1,y2-y1],score=score))
    (out/'predictions.json').write_text(json.dumps(preds))
    assert preds,'No predictions'
    evaluator=COCOeval_faster(gt,gt.loadRes(preds),'bbox');evaluator.params.maxDets=[1,10,100]
    evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    per_class={}
    for k,cat in enumerate(evaluator.params.catIds):
        v=evaluator.eval['precision'][:,:,k,0,-1];v=v[v>-1]
        per_class[gt.cats[cat]['name']]=float(v.mean()*100) if v.size else None
    import ultralytics,torch
    report=dict(checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=sha(a.checkpoint),annotations_sha256=sha(a.annotations),
        AP=float(evaluator.stats[0]*100),AP50=float(evaluator.stats[1]*100),AP75=float(evaluator.stats[2]*100),AP_small=float(evaluator.stats[3]*100),AP_medium=float(evaluator.stats[4]*100),AP_large=float(evaluator.stats[5]*100),per_class_AP=per_class,
        evaluator='faster_coco_eval.COCOeval_faster bbox maxDets [1,10,100]',num_images=len(ims),num_predictions=len(preds),
        parameters_before_fusion=parameters_before_fusion,settings=dict(imgsz=640,conf=1e-5,iou=0.7,max_det=300,half=False,augment=False,nms=None,end2end=getattr(model.model,'end2end',None)),ultralytics_version=ultralytics.__version__,torch_version=torch.__version__)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
