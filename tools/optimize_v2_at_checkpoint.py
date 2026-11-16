"""One-shot, guarded optimization handoff after a complete epoch checkpoint."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
import yaml


def wait_until_exited(processes, timeout):
    """Poll identity-aware process state; do not depend on container pidfd support."""
    deadline = time.monotonic() + timeout
    while True:
        alive = []
        for process in processes:
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    alive.append(process)
            except psutil.NoSuchProcess:
                pass
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(.2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir',required=True)
    p.add_argument('--target-epoch',type=int,default=1)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1];work=Path(a.workdir).resolve()
    output=work/'optimization';output.mkdir(exist_ok=True)
    def status(phase,**details):
        row=dict(phase=phase,**details)
        (output/'status.json').write_text(json.dumps(row,indent=2))
        print('OPTIMIZATION',json.dumps(row),flush=True)
    def launch_queue():
        with (work/'sequence.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-u',str(root/'tools/run_v2_coco_sequence.py'),
                '--workdir',str(work),'--epochs','100','--patience','20','--warmup','60','--steps','20'],
                cwd=root,stdout=log,stderr=log,start_new_session=True)
        return process.pid
    def run(command,label,env,timeout=600):
        if (output/'CANCEL').exists():raise RuntimeError('Optimization cancelled')
        with (output/(label+'.log')).open('w') as log:
            process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=log,start_new_session=True)
            (output/'active-experiment.pid').write_text(str(process.pid))
            try:
                code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                return False
        return code==0
    status('waiting_for_complete_epoch_checkpoint',target_epoch=a.target_epoch)
    deadline=time.monotonic()+6*3600
    while True:
        if (output/'CANCEL').exists():
            status('cancelled_without_stopping_training');return
        progress=work/'runs/L/status.json'
        if progress.exists():
            try: row=json.loads(progress.read_text())
            except json.JSONDecodeError: row={}
            if row.get('epoch',0)>=a.target_epoch and (work/'runs/L/last.pth').exists():break
        supervisor=int((work/'sequence.pid').read_text())
        if not psutil.pid_exists(supervisor):
            raise RuntimeError('Original supervisor exited before a complete checkpoint; preserving files')
        if time.monotonic()>deadline:
            raise TimeoutError('No complete checkpoint; original training left running')
        time.sleep(2)
    # status.json is emitted after last/best checkpoint writes have completed.
    import torch
    checkpoint=work/'runs/L/last.pth'
    metadata=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert metadata['last_epoch']+1>=a.target_epoch
    assert {'optimizer','scaler','ema','model'} <= set(metadata)
    preserved_epoch=metadata['last_epoch']+1
    del metadata
    supervisor=int((work/'sequence.pid').read_text())
    owner=psutil.Process(supervisor)
    if owner.uids().real!=os.getuid() or 'run_v2_coco_sequence.py' not in ' '.join(owner.cmdline()) or os.getpgid(supervisor)!=supervisor:
        raise RuntimeError('Supervisor identity/group validation failed; no process stopped')
    status('checkpoint_preserved',epoch=preserved_epoch,checkpoint=str(checkpoint))
    jobs=[owner,*owner.children(recursive=True)]
    resumed=False
    try:
        # Recovery covers termination and waiting too, not only experiments.
        # Poll status because pidfd_open can fail with EINVAL in containers.
        os.killpg(supervisor,signal.SIGTERM)
        alive=wait_until_exited(jobs,20)
        for process in alive:
            try:
                if process.is_running() and os.getpgid(process.pid)==supervisor:process.kill()
            except (ProcessLookupError,psutil.NoSuchProcess):pass
        if wait_until_exited(alive,10):
            raise RuntimeError('Owned queue did not exit; abort experiments')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',
                 TORCHINDUCTOR_COMPILE_THREADS='1',BEAR_FAST_RUNTIME='0',BEAR_COMPILE_NECK='0')
        parity_path=output/'execution-parity.json'
        status('validating_gpu_math',epoch=preserved_epoch)
        run([sys.executable,str(root/'tools/verify_execution_parity.py'),'--config',str(work/'configs/L-g8-full.yml'),
             '--checkpoint',str(checkpoint),'--output',str(parity_path),'--compile'],'parity',env,600)
        parity=json.loads(parity_path.read_text()) if parity_path.exists() else {}
        eligible=parity.get('passed',False)
        modes=[('baseline',{'BEAR_FAST_RUNTIME':'0','BEAR_BACKBONE_CHECKPOINT':'1','BEAR_COMPILE_NECK':'0'})]
        if eligible:
            modes.append(('fast',{'BEAR_FAST_RUNTIME':'1','BEAR_BACKBONE_CHECKPOINT':'1','BEAR_COMPILE_NECK':'0'}))
            if parity.get('tests',{}).get('backbone_checkpoint_bitwise'):
                modes.append(('fast_no_recompute',{'BEAR_FAST_RUNTIME':'1','BEAR_BACKBONE_CHECKPOINT':'0','BEAR_COMPILE_NECK':'0'}))
            if parity.get('compile_deployable'):
                modes.append(('fast_compiled',{'BEAR_FAST_RUNTIME':'1','BEAR_BACKBONE_CHECKPOINT':'1','BEAR_COMPILE_NECK':'1'}))
        results=[]
        for label,settings in modes:
            status('benchmarking_execution',candidate=label,epoch=preserved_epoch)
            report_path=output/(label+'.json')
            process_env=dict(env,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',**settings)
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                str(root/'tools/benchmark_v2_ddp.py'),'--config',str(work/'configs/L-g8.yml'),
                '--checkpoint',str(checkpoint),'--output',str(report_path),'--accumulate','1',
                '--warmup','40','--steps','30','--restore-training-state','--skip-validation']
            succeeded=run(command,label,process_env,600)
            if succeeded and report_path.exists():
                result=json.loads(report_path.read_text())
                if result.get('passed'):
                    results.append((result['seconds_per_optimizer_window'],label,settings))
        # Compilation may change rounding even after a limited parity test.
        # Keep it experimental until a separate AP acceptance gate is supplied.
        deployable=[row for row in results if row[1]!='fast_compiled']
        baseline=next((row for row in deployable if row[1]=='baseline'),None)
        if baseline is None:raise RuntimeError('Baseline benchmark failed; resume original configuration')
        best=min(deployable)
        if best[0]>baseline[0]*.97:best=baseline
        profile=dict(environment=best[2],selected=best[1],baseline_seconds_per_window=baseline[0],
                     selected_seconds_per_window=best[0],speedup=baseline[0]/best[0],
                     preserved_epoch=preserved_epoch,global_batch=8,micro_batch_per_gpu=1,
                     note='Same architecture/recipe/batch. GPU math gate plus measured throughput; no final AP claim.')
        (work/'execution-profile.json').write_text(json.dumps(profile,indent=2))
        # A capacity test is recorded separately; changing micro-batch changes
        # BatchNorm behavior and is deliberately ineligible for automatic use.
        capacity=yaml.safe_load((work/'configs/L-g4.yml').read_text())
        capacity['train_dataloader']['total_batch_size']=8
        capacity_path=output/'capacity-batch2.yml'
        capacity_path.write_text(yaml.safe_dump(capacity,sort_keys=False))
        status('measuring_capacity_only',selected=best[1])
        run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
             str(root/'tools/benchmark_v2_ddp.py'),'--config',str(capacity_path),
             '--checkpoint',str(checkpoint),'--output',str(output/'capacity-batch2.json'),
             '--accumulate','1','--warmup','20','--steps','20','--restore-training-state','--skip-validation'],
             'capacity-batch2',dict(env,CUDA_VISIBLE_DEVICES='0,1,2,3',**best[2]),300)
        pid=launch_queue();resumed=True
        status('resumed_optimized_training',supervisor_pid=pid,profile=profile)
    finally:
        if not resumed and not (output/'CANCEL').exists():
            # Never leave a failed optimization experiment holding the queue.
            (work/'execution-profile.json').write_text(json.dumps(dict(environment={
                'BEAR_FAST_RUNTIME':'0','BEAR_BACKBONE_CHECKPOINT':'1','BEAR_COMPILE_NECK':'0'},selected='baseline_recovery')))
            pid=launch_queue()
            status('resumed_baseline_after_failed_experiment',supervisor_pid=pid)


if __name__=='__main__':main()
