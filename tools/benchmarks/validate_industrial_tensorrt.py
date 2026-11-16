"""Rebuild and validate a selected Bear checkpoint; precision selection uses val only."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback,shutil
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--dataset',default='cr7');p.add_argument('--export-pid',type=int);p.add_argument('--gpu',default='0');p.add_argument('--precision',choices=['auto','mixed','fp32-backbone','fp32'],default='auto');a=p.parse_args()
b=a.base.resolve();r=b/'repo';out=b/'deployment-validation-20261116'/(a.dataset if a.precision=='auto' else a.dataset+'-'+a.precision);out.mkdir(parents=True,exist_ok=True)
py=str(b/'export-venv/bin/python');cfg=b/'runs'/a.dataset/'bear_l/prepared/config.yml';checkpoint=b/'runs'/a.dataset/'bear_l/prepared/training/best_stg1.pth';data=b/'datasets'/a.dataset
env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
state=dict(dataset=a.dataset,pid=os.getpid(),status='running',started=time.time(),checkpoint=str(checkpoint),gpu=a.gpu,AP_tolerance_points=0.2,precision_selection_split='val')
def save(stage,**details):
    state.update(stage=stage,updated=time.time(),**details)
    t=out/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(out/'status.json')
    print('DEPLOYMENT',json.dumps(state),flush=True)
def run(cmd,name,timeout=3600):
    save(name,command=cmd)
    with (out/(name+'.log')).open('w') as log:
        subprocess.run([py]+cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=True)
def validate(backend,split,destination,engine=None):
    cmd=['tools/validate_deployment.py','--backend',backend,'--config',str(cfg),'--checkpoint',str(checkpoint),'--annotations',str(data/(split+'.json')),'--images',str(data/'images'/split),'--output',str(destination),'--batch-size','1','--workers','2']
    if engine:cmd+=['--engine',str(engine)]
    run(cmd,destination.stem)
    return json.loads(destination.read_text())
try:
    if a.export_pid:
        save('waiting_existing_export',export_pid=a.export_pid)
        deadline=time.time()+7200
        while time.time()<deadline:
            proc=Path('/proc')/str(a.export_pid)
            try:
                raw=(proc/'stat').read_text()
                running=raw[raw.rfind(')')+2:].split()[0]!='Z'
            except OSError:running=False
            if not running:break
            time.sleep(5)
        else:raise TimeoutError('Existing export exceeded two-hour deadline')
        if not (out/'mixed/model.engine').exists():
            raise RuntimeError('Initial mixed export failed; inspect export-mixed.log')
    val_file=out/'fp32-val.json'
    if not val_file.exists() or not json.loads(val_file.read_text()).get('annotation_ids_reindexed'):
        baseline=validate('torch-fp32','val',val_file)
    else:baseline=json.loads(val_file.read_text())
    attempts=[];selected=None
    for mode in (['mixed','fp32-backbone','fp32'] if a.precision=='auto' else [a.precision]):
        directory=out/mode
        directory.mkdir(exist_ok=True)
        if not (directory/'model.engine').exists():
            if mode!='mixed':
                for f in ['model.onnx','export.json','classes.json']:
                    if (out/'mixed'/f).exists():shutil.copy2(out/'mixed'/f,directory/f)
            cmd=['tools/export_engine.py','--config',str(cfg),'--checkpoint',str(checkpoint),'--output',str(directory),'--build','--precision',mode]
            if (directory/'export.json').exists():cmd+=['--build-only']
            run(cmd,'build-'+mode,7200)
        result_path=out/('val-'+mode+'.json')
        try:
            result=validate('tensorrt','val',result_path,directory/'model.engine')
            delta=result['AP']-baseline['AP']
            accepted=result['finite_outputs'] and abs(delta)<=0.2
            attempts.append(dict(precision=mode,AP=result['AP'],AP_delta_points=delta,accepted=accepted))
        except subprocess.CalledProcessError:
            attempts.append(dict(precision=mode,accepted=False,error='validation command failed; see log'))
            accepted=False
        (out/'precision-attempts.json').write_text(json.dumps(attempts,indent=2))
        if accepted:selected=mode;break
    if selected is None:raise RuntimeError('No engine met finite-output/AP validation tolerance')
    save('precision_locked',selected=selected,attempts=attempts)
    engine=out/selected/'model.engine'
    test_fp32=validate('torch-fp32','test',out/'test-fp32.json')
    test_trt=validate('tensorrt','test',out/'test-tensorrt.json',engine)
    # The captured training inputs are stress inputs only, never AP/tuning data.
    capture=b/'debug/ksdd2-finite/failure.pt'
    stress=dict(performed=False)
    if capture.exists():
        stress_code="""import sys,json,torch
from pathlib import Path
sys.path[:0]=['.','tools','source']
from deployment_backend import load_backend
forward,nc,extra=load_backend('tensorrt',sys.argv[1],sys.argv[2],sys.argv[3])
d=torch.load(sys.argv[4],map_location='cpu',weights_only=False)
samples=d['samples'];count=0
with torch.inference_mode():
 for image in samples:
  x=image[None].cuda();sz=torch.tensor([[640,640]],device='cuda',dtype=torch.int64)
  result=forward(x,sz)
  assert all(torch.isfinite(t).all() for t in result)
  count+=1
Path(sys.argv[5]).write_text(json.dumps(dict(performed=True,images=count,finite_outputs=True,meaning='Captured inputs only; no annotation/AP inference')))
"""
        run(['-c',stress_code,str(cfg),str(checkpoint),str(engine),str(capture),str(out/'stress.json')],'captured-input-stress')
        stress=json.loads((out/'stress.json').read_text())
    im=json.loads((data/'test.json').read_text())['images'][0]['file_name']
    for backend in ['torch-fp32','tensorrt']:
        cmd=['tools/benchmark_latency.py','--backend',backend,'--config',str(cfg),'--checkpoint',str(checkpoint),'--image',str(data/'images/test'/im),'--output',str(out/('latency-'+backend+'.json')),'--warmup','50','--iterations','100','--repeats','3']
        if backend=='tensorrt':cmd+=['--engine',str(engine)]
        run(cmd,'latency-'+backend)
    metadata=json.loads((out/selected/'export.json').read_text())
    test_delta=test_trt['AP']-test_fp32['AP']
    accepted=test_trt['finite_outputs'] and abs(test_delta)<=0.2
    metadata.update(accuracy_validated=accepted,validation_performed=True,accuracy_tolerance_met=accepted,validation=dict(dataset=a.dataset,split='val',images=baseline['images'],AP_tolerance_points=0.2,selected_precision=selected),test_validation=dict(images=test_trt['images'],FP32_AP=test_fp32['AP'],TensorRT_AP=test_trt['AP'],AP_delta_points=test_delta,finite_outputs=test_trt['finite_outputs']),captured_input_stress=stress)
    (out/selected/'export.json').write_text(json.dumps(metadata,indent=2))
    result=dict(dataset=a.dataset,selected_precision=selected,checkpoint_sha256=metadata['checkpoint_sha256'],engine_sha256=metadata['engine_sha256'],accuracy_validated=accepted,validation_performed=True,accuracy_tolerance_met=accepted,val_attempts=attempts,test_fp32=test_fp32,test_tensorrt=test_trt,test_AP_delta_points=test_delta,stress=stress,latency={backend:json.loads((out/('latency-'+backend+'.json')).read_text()) for backend in ['torch-fp32','tensorrt']})
    (out/'COMPLETE.json').write_text(json.dumps(result,indent=2))
    save('complete',status='complete',accuracy_validated=accepted,result=str(out/'COMPLETE.json'))
except Exception as error:
    save('failed',status='failed',error=str(error))
    traceback.print_exc()
    sys.exit(1)
