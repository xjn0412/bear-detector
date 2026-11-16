"""Rescore saved benchmark predictions after positive COCO annotation-ID repair."""
import argparse,json,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from coco_metrics import ground_truth
from faster_coco_eval import COCOeval_faster

def rescore(annotations,predictions,report_path):
    report_path=Path(report_path)
    old=json.loads(report_path.read_text())
    if old.get('annotation_ids_reindexed'):return old
    backup=report_path.with_name(report_path.stem+'.annotation-id0.json')
    if not backup.exists():shutil.copy2(report_path,backup)
    gt=ground_truth(annotations);pred=json.loads(Path(predictions).read_text())
    ev=COCOeval_faster(gt,gt.loadRes(pred),'bbox');ev.params.maxDets=[1,10,100]
    ev.evaluate();ev.accumulate();ev.summarize()
    for name,value in zip(['AP','AP50','AP75','AP_small','AP_medium','AP_large'],ev.stats[:6]):
        old[name]=float(value*100)
    old['per_class_AP']={}
    for k,cat in enumerate(ev.params.catIds):
        values=ev.eval['precision'][:,:,k,0,-1];values=values[values>-1]
        old['per_class_AP'][gt.cats[cat]['name']]=float(values.mean()*100) if values.size else None
    old.update(annotation_ids_reindexed=True,evaluation_correction='Ground-truth annotation IDs replaced in memory by1..N; predictions/geometry/checkpoint unchanged')
    report_path.write_text(json.dumps(old,indent=2))
    return old

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    b=a.base
    for ds in ['cr7','gc10','neu-det','ksdd2']:
        for model in ['bear_l','yolo26x']:
            out=b/'runs'/ds/model
            if not (out/'COMPLETE.json').exists():continue
            predictions=out/('test-report.predictions.json' if model=='bear_l' else 'evaluation/predictions.json')
            report=rescore(b/'datasets'/ds/'test.json',predictions,out/'test-report.json')
            if model=='yolo26x':(out/'evaluation/report.json').write_text(json.dumps(report,indent=2))
            complete=json.loads((out/'COMPLETE.json').read_text());complete['report']=report
            complete['evaluation_correction']='positive COCO annotation IDs; original report retained locally'
            (out/'COMPLETE.json').write_text(json.dumps(complete,indent=2))
            print('RESCORED',ds,model,report['AP'],flush=True)
    out=b/'deployment-validation-20261116/cr7'
    if (out/'fp32-val.json').exists():
        rescore(b/'datasets/cr7/val.json',out/'fp32-val.predictions.json',out/'fp32-val.json')

if __name__=='__main__':main()
