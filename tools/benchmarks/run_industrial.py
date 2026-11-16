"""One isolated benchmark model/dataset job. Shared images, native model recipes.

100 epochs maximum, patience20, seed42, batch8, 640 inputs.
Test is evaluated only after validation selects the checkpoint.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import yaml


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--dataset',required=True)
    ap.add_argument('--model',choices=['bear_l','yolo26x'],required=True)
    ap.add_argument('--resume',action='store_true',help='Resume Bear from its full last checkpoint')
    a=ap.parse_args()
    if a.resume and a.model!='bear_l':raise ValueError('Resume currently supports Bear only')
    base=a.base.resolve();repo=base/'repo';data=base/'datasets'/a.dataset
    out=base/'runs'/a.dataset/a.model;out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    py=sys.executable
    protocol=dict(dataset=a.dataset,model=a.model,seed=42,input=640,batch=8,epoch_limit=100,patience=20,
                  recipe='Model-specific fine-tuning recipes, shared image geometry/splits/budget; not an architecture-only ablation',
                  checkpoint_selection='Best validation AP50:95; test excluded from training and selection',
                  test_metric='COCO bbox AP50:95 maxDets100 with faster-coco-eval',
                  annotations={s:sha(data/(s+'.json')) for s in ['train','val','test']},
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                  gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),started=time.time())
    def run(cmd,name):
        print('RUN',a.dataset,a.model,name,flush=True)
        with (out/(name+'.log')).open('a' if a.resume else 'w') as log:
            subprocess.run(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    if a.model=='bear_l':
        ck=base/'downloads/bear-l-coco-mixed.pt';prepared=out/'prepared'
        protocol.update(initialization_sha256=sha(ck),initialization='Released Bear-L COCO mixed checkpoint; FP32 parameters and decoder; fresh class heads/optimizer/EMA')
        if not (prepared/'initial.pth').exists():
            run([py,'tools/prepare_finetune.py','--config','configs/L.yml','--checkpoint',str(ck),
                 '--train-json',str(data/'train.json'),'--val-json',str(data/'val.json'),
                 '--train-images',str(data/'images/train'),'--val-images',str(data/'images/val'),
                 '--output',str(prepared),'--epochs','100','--batch','8','--lr','0.0001'],'prepare')
        cfgpath=prepared/'config.yml';cfg=yaml.safe_load(cfgpath.read_text())
        for split in ['train','val']:
            cfg[split+'_dataloader'].update(drop_last=False,num_workers=2,total_batch_size=8)
        cfg.update(checkpoint_freq=100,flat_epoch=3,no_aug_epoch=0,lr_gamma=0.1,
                   warmup_iter=3*math.ceil(len(json.loads((data/'train.json').read_text())['images'])/8))
        cfg['optimizer']['weight_decay']=1e-4
        cfg['train_dataloader']['dataset']['transforms']['ops']=[op for op in cfg['train_dataloader']['dataset']['transforms']['ops'] if op['type'] not in ['RandomPhotometricDistort','SanitizeBoundingBoxes']]
        cfgpath.write_text(yaml.safe_dump(cfg,sort_keys=False))
        if a.resume:
            prior=json.loads((out/'protocol.json').read_text())
            protocol['started']=prior['started']
            protocol['recovery']=prior.get('recovery',[])+[dict(time=time.time(),from_epoch=json.loads((prepared/"training/log.txt").read_text().splitlines()[-1])["epoch"]+1,
                reason='Reproduced FP16 overflow in P5 projection convolution before GroupNorm',
                precision='P5 projection FP32; remaining backbone/neck FP16 AMP; decoder/loss FP32',
                preserves='Full model/optimizer/scaler/EMA, validation history and patience')]
        protocol['settings']=cfg
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        train_cmd=[py,'tools/benchmarks/train_industrial_bear.py','--run',str(prepared)]
        if a.resume:
            last=prepared/'training/last.pth'
            if not last.exists():raise FileNotFoundError(last)
            train_cmd+=['--resume',str(last)]
        run(train_cmd,'train')
        rows=[json.loads(l) for l in (prepared/'training/log.txt').read_text().splitlines()]
        curves=[dict(epoch=r['epoch']+1,val_AP=r['test_coco_eval_bbox'][0]*100,val_AP50=r['test_coco_eval_bbox'][1]*100) for r in rows]
        selected=max(curves,key=lambda r:r['val_AP'])['epoch']
        ck=prepared/'training/best_stg1.pth'
        assert ck.exists()
        # Same FP32 evaluator precision as the YOLO final test pass.
        run([py,'tools/validate_deployment.py','--backend','torch-fp32','--config',str(cfgpath),
             '--checkpoint',str(ck),'--annotations',str(data/'test.json'),'--images',str(data/'images/test'),
             '--output',str(out/'test-report.json'),'--batch-size','4','--workers','2'],'evaluate')
    else:
        os.environ['YOLO_CONFIG_DIR']=str(out/'ultra-settings')
        os.environ['YOLO_LONG_BUDGET_ROOT']=str(out)
        from ultralytics import YOLO
        from yolo_long_budget_trainer import LongBudgetTrainer
        import ultralytics
        ck=base/'downloads/yolo26x.pt'
        model=YOLO(str(ck))
        protocol.update(initialization_sha256=sha(ck),initialization='Official YOLO26x COCO checkpoint',ultralytics_version=ultralytics.__version__)
        kwargs=dict(data=str(data/'dataset.yaml'),epochs=100,patience=20,imgsz=640,batch=8,device=0,
                    workers=2,seed=42,deterministic=True,project=str(out),name='training',exist_ok=False,
                    optimizer='AdamW',lr0=0.001,momentum=0.9,weight_decay=0.0005,amp=True,
                    plots=False,save=True,save_period=20,cache=False,rect=False)
        kwargs['save_dir']=str(out/'training')
        protocol['settings']=kwargs
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        model.train(trainer=LongBudgetTrainer,**kwargs)
        rows=list(csv.DictReader((out/'training/results.csv').open()))
        curves=[dict(epoch=int(r['epoch']),val_AP=float(r['metrics/mAP50-95(B)'])*100,val_AP50=float(r['metrics/mAP50(B)'])*100) for r in rows]
        selected=max(curves,key=lambda r:r['val_AP'])['epoch']
        ck=out/'training/weights/best.pt'
        run([py,'tools/benchmarks/evaluate_wildlife_yolo.py','--checkpoint',str(ck),
             '--annotations',str(data/'test.json'),'--images',str(data/'images/test'),
             '--output',str(out/'evaluation')],'evaluate')
        shutil.copyfile(out/'evaluation/report.json',out/'test-report.json')
    (out/'curves.json').write_text(json.dumps(curves,indent=2))
    complete=dict(dataset=a.dataset,model=a.model,completed_epochs=len(curves),selected_epoch=selected,
                  checkpoint=str(ck),checkpoint_sha256=sha(ck),finished=time.time(),
                  report=json.loads((out/'test-report.json').read_text()))
    (out/'COMPLETE.json').write_text(json.dumps(complete,indent=2))
    print('COMPLETE',json.dumps(complete),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();sys.exit(1)
