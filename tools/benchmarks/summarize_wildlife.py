"""Common COCO metrics and paired image bootstrap for the transfer experiment."""
import argparse,contextlib,io,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from faster_coco_eval import COCO,COCOeval_faster


def evaluate(dataset,predictions):
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=dataset;gt.createIndex()
        ev=COCOeval_faster(gt,gt.loadRes(predictions),'bbox')
        ev.params.maxDets=[1,10,100]
        ev.evaluate();ev.accumulate();ev.summarize()
    per_class={}
    for k,c in enumerate(ev.params.catIds):
        v=ev.eval['precision'][:,:,k,0,-1];v=v[v>-1]
        per_class[gt.cats[c]['name']]=float(v.mean()*100) if v.size else None
    keys=['AP','AP50','AP75','AP_small','AP_medium','AP_large']
    return dict(zip(keys,map(lambda x:float(x*100) if x>=0 else None,ev.stats[:6]))),per_class


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--bootstrap',type=int,default=200);a=p.parse_args();root=Path(a.root)
    dataset=json.loads((root/'data640/test.json').read_text())
    files={'Bear-B':root/'evaluation/bear_b.predictions.json',
           'YOLO26-x':root/'evaluation/yolo26x/predictions.json',
           'YOLO11-x':root/'evaluation/yolo11x/predictions.json',
           'YOLO26-x native':root/'evaluation/yolo26x_native/predictions.json',
           'YOLO11-x native':root/'evaluation/yolo11x_native/predictions.json'}
    preds={n:json.loads(f.read_text()) for n,f in files.items()}
    summary={'metric':'COCO bbox AP50:95, 101 recall points, maxDets100; all values in percentage points',
             'split':'held-out test','images':len(dataset['images']),'models':{},
             'undefined_metric_note':'Null means no ground-truth objects in that size stratum.'}
    for n,ps in preds.items():
        metrics,pc=evaluate(dataset,ps)
        summary['models'][n]=dict(**metrics,per_class_AP=pc)
    anns=defaultdict(list)
    for x in dataset['annotations']:anns[x['image_id']].append(x)
    indexed={n:defaultdict(list) for n in preds}
    for n,ps in preds.items():
        for x in ps:indexed[n][x['image_id']].append(x)
    rng=np.random.default_rng(42);samples={n:[] for n in preds}
    for b in range(a.bootstrap):
        selected=rng.integers(0,len(dataset['images']),len(dataset['images']))
        boot=dict(info={},licenses=[],categories=dataset['categories'],images=[],annotations=[])
        bp={n:[] for n in preds};aid=0
        for new_id,k in enumerate(selected,1):
            im=dataset['images'][int(k)];old=im['id'];boot['images'].append(dict(im,id=new_id))
            for ann in anns[old]:
                aid+=1;boot['annotations'].append(dict(ann,id=aid,image_id=new_id))
            for n in preds:bp[n].extend(dict(x,image_id=new_id) for x in indexed[n][old])
        for n in preds:samples[n].append(evaluate(boot,bp[n])[0]['AP'])
        if (b+1)%25==0:print('bootstrap',b+1,'/',a.bootstrap,flush=True)
    if a.bootstrap:
        summary['bootstrap']=dict(replicates=a.bootstrap,seed=42,method='Paired image resampling with replacement; percentile 95% CI. Conditional on these trained checkpoints, not training-seed variability.',AP_95CI={n:np.percentile(v,[2.5,97.5]).tolist() for n,v in samples.items()},Bear_minus_YOLO_95CI={n:np.percentile(np.array(samples['Bear-B'])-np.array(samples[n]),[2.5,97.5]).tolist() for n in preds if n!='Bear-B'})
    out=root/'evaluation/summary.json';out.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
