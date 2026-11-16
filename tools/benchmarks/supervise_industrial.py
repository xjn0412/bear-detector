"""Launch eight isolated jobs and maintain a publishable evidence snapshot."""
from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import sys
import time


def snapshot(base,jobs):
    repo=base/'repo';dest=repo/'validation/benchmarks/industrial-20261116'
    dest.mkdir(parents=True,exist_ok=True)
    summary={}
    for ds in ['cr7','gc10','neu-det','ksdd2']:
        d=dest/ds;d.mkdir(exist_ok=True)
        data=base/'datasets'/ds
        for name in ['dataset-manifest.json','split-membership.json']:
            if (data/name).exists():(d/name).write_bytes((data/name).read_bytes())
        summary[ds]={}
        for model in ['bear_l','yolo26x']:
            out=base/'runs'/ds/model
            result={'status':jobs.get(ds+'/'+model,{}).get('status','waiting')}
            rows=[]
            if model=='bear_l':
                p=out/'prepared/training/log.txt'
                if p.exists():
                    for line in p.read_text().splitlines():
                        try:
                            r=json.loads(line);rows.append(dict(epoch=r['epoch']+1,val_AP=r['test_coco_eval_bbox'][0]*100,val_AP50=r['test_coco_eval_bbox'][1]*100))
                        except (ValueError,KeyError):pass
            else:
                p=out/'training/results.csv'
                if p.exists():
                    for r in csv.DictReader(p.open()):
                        rows.append(dict(epoch=int(r['epoch']),val_AP=float(r['metrics/mAP50-95(B)'])*100,val_AP50=float(r['metrics/mAP50(B)'])*100))
            if rows:
                (d/(model+'-curves.json')).write_text(json.dumps(rows,indent=2))
                result.update(completed_epochs=len(rows),best_val_AP=max(x['val_AP'] for x in rows))
            for source,target in [('protocol.json',model+'-protocol.json'),('test-report.json',model+'-test.json'),('COMPLETE.json',model+'-complete.json')]:
                if (out/source).exists():(d/target).write_bytes((out/source).read_bytes())
            if (out/'COMPLETE.json').exists():result.update(json.loads((out/'COMPLETE.json').read_text()))
            summary[ds][model]=result
    status=dict(updated=time.time(),jobs=jobs,results=summary)
    (dest/'status.json').write_text(json.dumps(status,indent=2))
    (base/'status.json').write_text(json.dumps(status,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    b=a.base.resolve();repo=b/'repo';py=str(b/'venv/bin/python')
    jobs={};processes={};preparing={}
    # One GPU per run avoids DDP overhead for these small datasets.
    while True:
        for dsindex,ds in enumerate(['cr7','gc10','neu-det','ksdd2']):
            ready=(b/'datasets'/ds/'dataset-manifest.json').exists()
            raw=b/'raw'/ds
            if not ready and ds=='ksdd2' and ds not in preparing and raw.exists():
                # A finalized archive plus extraction completion is required.
                archive=b/'downloads'/('neu-det.zip' if ds=='neu-det' else ds+'.zip')
                expected=3335*2 if ds=='ksdd2' else None
                extracted=sum(1 for x in raw.rglob('*') if x.is_file())
                if archive.exists() and (expected is None or extracted>=expected):
                    log=(b/('supervisor_prepare_'+ds+'.log')).open('w')
                    preparing[ds]=subprocess.Popen([py,str(repo/'tools/benchmarks/prepare_industrial.py'),'--base',str(b),'--dataset',ds],stdout=log,stderr=subprocess.STDOUT)
            for mi,model in enumerate(['bear_l','yolo26x']):
                key=ds+'/'+model;gpu=dsindex*2+mi
                if key not in jobs and ready:
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                             PYTHONUNBUFFERED='1',XDG_CACHE_HOME=str(b/'cache'),HF_HOME=str(b/'cache/hf'),
                             YOLO_CONFIG_DIR=str(b/('cache/yolo-'+str(gpu))),MPLCONFIGDIR=str(b/('cache/mpl-'+str(gpu))))
                    out=b/'runs'/ds/model;out.mkdir(parents=True,exist_ok=True)
                    cmd=[py,str(repo/'tools/benchmarks/run_industrial.py'),'--base',str(b),'--dataset',ds,'--model',model]
                    log=(out/'job.log').open('w')
                    proc=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
                    processes[key]=proc;jobs[key]=dict(pid=proc.pid,gpu=gpu,status='running',started=time.time(),command=cmd)
                    print('START',key,proc.pid,'GPU',gpu,flush=True)
                if key in processes and processes[key].poll() is not None and jobs[key]['status']=='running':
                    rc=processes[key].returncode
                    jobs[key].update(status='complete' if rc==0 else 'failed',returncode=rc,finished=time.time())
                    print('EXIT',key,rc,flush=True)
        snapshot(b,jobs)
        if len(jobs)==8 and all(x['status']!='running' for x in jobs.values()):break
        time.sleep(30)


if __name__=='__main__':main()
