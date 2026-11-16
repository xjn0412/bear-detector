"""Audit published industrial metrics against raw epoch logs and saved predictions."""
import argparse,csv,hashlib,json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from coco_metrics import ground_truth
from faster_coco_eval import COCOeval_faster

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def blob(path):
    raw=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--published-tree',type=Path,required=True);p.add_argument('--published-commit',required=True);p.add_argument('--published-root',type=Path);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    b=a.base;r=b/'repo';published=a.published_root or r;root=published/'validation/benchmarks/industrial-20261116'
    checks=[];errors=[];runs=[];publication=[];table_curves=[];findings=[]
    def check(ok,label,**details):
        entry=dict(check=label,passed=bool(ok),**details);checks.append(entry)
        if not ok:errors.append(entry)
    for entry in json.loads(a.published_tree.read_text()):
        path=published/entry['path']
        same=path.is_file() and blob(path)==entry['sha']
        publication.append(dict(path=entry['path'],published_blob=entry['sha'],local_blob=blob(path) if path.is_file() else None,matched=same))
    check(all(x['matched'] for x in publication),'published files equal audited local bytes',files=len(publication))
    exported=[]
    for ds in ['cr7','gc10','neu-det','ksdd2']:
        data=b/'datasets'/ds
        for model in ['bear_l','yolo26x']:
            out=b/'runs'/ds/model;directory=root/ds
            if model=='bear_l':
                source=out/'prepared/training/log.txt';raw=[json.loads(line) for line in source.read_text().splitlines()]
                rows=[dict(epoch=x['epoch']+1,val_AP=x['test_coco_eval_bbox'][0]*100,val_AP50=x['test_coco_eval_bbox'][1]*100) for x in raw]
                predictions=out/'test-report.predictions.json'
            else:
                source=out/'training/results.csv';raw=list(csv.DictReader(source.open()))
                rows=[dict(epoch=int(x['epoch']),val_AP=float(x['metrics/mAP50-95(B)'])*100,val_AP50=float(x['metrics/mAP50(B)'])*100) for x in raw]
                predictions=out/'evaluation/predictions.json'
            curve=json.loads((directory/(model+'-curves.json')).read_text());table_curves.append(curve)
            check([x['epoch'] for x in rows]==list(range(1,len(rows)+1)),f'{ds}/{model}: contiguous completed epochs')
            check(len(curve)==len(rows) and all(p['epoch']==q['epoch'] and abs(p['val_AP']-q['val_AP'])<1e-10 and abs(p['val_AP50']-q['val_AP50'])<1e-10 for p,q in zip(curve,rows)),f'{ds}/{model}: every published epoch matches raw log',epochs=len(rows),metric_values=len(rows)*2)
            complete=json.loads((out/'COMPLETE.json').read_text());report=json.loads((out/'test-report.json').read_text())
            best=max(rows,key=lambda x:x['val_AP'])
            check(complete['selected_epoch']==best['epoch'] and complete['completed_epochs']==len(rows),f'{ds}/{model}: selected epoch and run length')
            check(json.loads((directory/(model+'-complete.json')).read_text())==complete and json.loads((directory/(model+'-test.json')).read_text())==report,f'{ds}/{model}: published final records match source reports')
            check(sha(complete['checkpoint'])==complete['checkpoint_sha256']==report['checkpoint_sha256'],f'{ds}/{model}: checkpoint SHA256')
            check(sha(data/'test.json')==report['annotations_sha256'],f'{ds}/{model}: annotation SHA256')
            if 'predictions_sha256' in report:
                check(sha(predictions)==report['predictions_sha256'],f'{ds}/{model}: predictions SHA256')
            protocol=json.loads((out/'protocol.json').read_text())
            check(all(sha(data/(split+'.json'))==protocol['annotations'][split] for split in ['train','val','test']),f'{ds}/{model}: split annotation hashes')
            check(report.get('annotation_ids_reindexed') is True,f'{ds}/{model}: corrected COCO annotation IDs disclosed')
            gt=ground_truth(data/'test.json')
            predicted=json.loads(predictions.read_text())
            check(all(math.isfinite(x['score']) and all(math.isfinite(v) for v in x['bbox']) for x in predicted),f'{ds}/{model}: saved predictions finite')
            evaluator=COCOeval_faster(gt,gt.loadRes(predicted),'bbox');evaluator.params.maxDets=[1,10,100]
            evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
            recalculated=dict(zip(['AP','AP50','AP75','AP_small','AP_medium','AP_large'],[float(x*100) for x in evaluator.stats[:6]]))
            check(all(abs(report[key]-value)<1e-9 for key,value in recalculated.items()),f'{ds}/{model}: independently rescored test metrics',recalculated=recalculated)
            runs.append(dict(dataset=ds,model=model,epochs=len(rows),best_epoch=best['epoch'],best_native_val_AP=best['val_AP'],native_curve_source_sha256=sha(source),test_AP=recalculated['AP'],test_AP50=recalculated['AP50'],selected_checkpoint_sha256=complete['checkpoint_sha256'],saved_predictions_sha256=sha(predictions)))
            for row in rows:exported.append(dict(dataset=ds,model=model,epoch=row['epoch'],val_mAP50_95_pct=row['val_AP'],val_mAP50_pct=row['val_AP50'],selected_epoch=row['epoch']==best['epoch'],metric_source='native Bear evaluator (historical annotation-ID0)' if model=='bear_l' else 'native Ultralytics evaluator'))
            print('AUDITED',ds,model,len(rows),recalculated['AP'],flush=True)
    table_lines=[line for line in (root/'epoch-progress.md').read_text().splitlines() if line.startswith('|') and line.split('|')[1].isdigit()]
    table_ok=len(table_lines)==max(len(curve) for curve in table_curves)
    for i,line in enumerate(table_lines):
        expected=[str(i+1)]+[f"{curve[i]['val_AP']:.2f}" if i<len(curve) else '—' for curve in table_curves]
        table_ok=table_ok and line.split('|')[1:-1]==expected
    check(table_ok,'combined per-epoch table matches all eight curves')
    for directory in root.glob('*'):
        compressed=directory/'split-membership.json.gz'
        if compressed.exists():
            import gzip
            actual=json.loads(gzip.decompress(compressed.read_bytes()))
            expected=json.loads((b/'datasets'/directory.name/'split-membership.json').read_text())
            check(actual==expected,f'{directory.name}: published compressed split membership matches source')
    archive=root/'gc10-invalid-12class'
    for model in ['bear_l','yolo26x']:
        old=json.loads((archive/(model+'-curves.json')).read_text())
        local=json.loads((r/'validation/benchmarks/industrial-20261116/gc10-invalid-12class'/(model+'-curves.json')).read_text())
        if len(old)!=len(local):findings.append(dict(scope='invalid GC10 diagnostic archive',model=model,published_epochs=len(old),full_archived_epochs=len(local),active_results_affected=False,repair='Refresh the explicitly invalid archive with its complete recorded curve'))
    a.report.parent.mkdir(parents=True,exist_ok=True)
    csv_path=a.report.parent/'all-epochs.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)
    result=dict(audited_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),published_commit=a.published_commit,passed=not errors,completed_runs=len(runs),epoch_rows=len(exported),epoch_metric_values=len(exported)*2,checks=checks,errors=errors,publication_file_checks=publication,runs=runs,findings=findings,limitations=['Historical Bear native validation curves and checkpoint selection retain the annotation-ID0 evaluator limitation; only final test scores were corrected.','Single seed, different pretrained systems and model-specific recipes; not an architecture-only comparison.','Original invalid12-class GC10 runs are diagnostic archives only.'])
    a.report.write_text(json.dumps(result,indent=2))
    print('AUDIT_RESULT',json.dumps(dict(passed=result['passed'],runs=len(runs),epoch_rows=len(exported),checks=len(checks),errors=errors)),flush=True)
    if errors:sys.exit(1)

if __name__=='__main__':main()
