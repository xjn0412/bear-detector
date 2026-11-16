"""Persistent preparation/4-vs-8 benchmark/training queue in L,B,S,T order."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir', required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--warmup', type=int, default=60)
    p.add_argument('--steps', type=int, default=20)
    a = p.parse_args()
    work = Path(a.workdir).resolve()
    root = Path(__file__).resolve().parents[1]
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'sequence.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (work / 'sequence.pid').write_text(str(os.getpid()))
    for folder in ['logs', 'configs', 'benchmarks', 'runs']:
        (work / folder).mkdir(exist_ok=True)
    status_path = work / 'sequence-status.json'
    def status(phase, **details):
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(phase=phase, order=['L','B','S','T'], **details), indent=2))
        temporary.replace(status_path)
        print('V2_SEQUENCE', phase, details, flush=True)
    def run(command, logfile, gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, range(gpus))),
                   OMP_NUM_THREADS='4', CHANNELS_LAST='1', BEAR_PRECISION='auto',
                   NCCL_ASYNC_ERROR_HANDLING='1', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        profile = work/'execution-profile.json'
        if profile.exists():
            setting=json.loads(profile.read_text())
            env.update({key:str(value) for key,value in setting['environment'].items()})
        with logfile.open('a') as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=log)
            (work / 'active-child.pid').write_text(str(child.pid))
            code = child.wait()
        if code:
            status('failed', returncode=code, logfile=str(logfile))
            raise RuntimeError(f'Child exited {code}; inspect {logfile}')
    status('waiting_for_assets')
    while not (work / 'benchmark-assets-ready.json').exists() and not (work / 'assets-ready.json').exists():
        time.sleep(10)
    assets = json.loads((work / ('benchmark-assets-ready.json' if (work / 'benchmark-assets-ready.json').exists() else 'assets-ready.json')).read_text())
    configs = {}
    for size in ['L', 'B', 'S', 'T']:
        base = yaml.safe_load((root / 'configs' / f'V2-{size}.yml').read_text())
        base.update(epoches=a.epochs, checkpoint_freq=5)
        for split in ['train','val']:
            base[split+'_dataloader']['dataset'].update(img_folder=assets[split]['directory'], ann_file=assets[split]['annotations'])
        for gpus in [4, 8]:
            cfg = copy.deepcopy(base)
            cfg['train_dataloader']['total_batch_size'] = gpus  # batch1 per GPU
            cfg['val_dataloader']['total_batch_size'] = gpus
            cfg['output_dir'] = str(work / 'runs' / size)
            config_path = work / 'configs' / f'{size}-g{gpus}.yml'
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            configs[size,gpus] = config_path
            full = copy.deepcopy(cfg)
            for split in ['train','val']:
                full[split+'_dataloader']['dataset'].update(img_folder=str(work/'coco'/f'{split}2017'), ann_file=str(work/'coco/annotations'/f'instances_{split}2017.json'))
            (work/'configs'/f'{size}-g{gpus}-full.yml').write_text(yaml.safe_dump(full,sort_keys=False))
    selected = {}
    for size in ['L', 'B', 'S', 'T']:
        rows = []
        for gpus in [4, 8]:
            status('benchmarking', model=size, gpus=gpus, selected=selected)
            output = work / 'benchmarks' / f'{size}-g{gpus}.json'
            if not output.exists():
                run([sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={gpus}',
                     str(root / 'tools/benchmark_v2_ddp.py'), '--config', str(configs[size,gpus]),
                     '--checkpoint', str(work / 'weights' / f'v2-{size.lower()}-initial.pt'),
                     '--output', str(output), '--accumulate', str(8//gpus),
                     '--warmup', str(a.warmup), '--steps', str(a.steps)],
                    work / 'logs' / f'benchmark-{size}-g{gpus}.log', gpus)
            result = json.loads(output.read_text())
            if not result['passed']:
                status('benchmark_rejected', model=size, gpus=gpus, reason=result.get('reason'))
                raise RuntimeError(f'Unreliable benchmark: {output}; inspect before training')
            rows.append(result)
        winner = min(rows, key=lambda row: row['estimated_epoch_seconds_excluding_checkpoint'])
        selected[size] = winner['gpus']
        (work / 'gpu-selection.json').write_text(json.dumps(dict(effective_global_batch=8,
            selection='Minimum measured estimated epoch wall time, fixed effective batch8, sequential L/B/S/T; checkpoint time common to both counts',
            selected=selected), indent=2))
    status('waiting_for_full_coco', selected=selected)
    while not (work / 'assets-ready.json').exists():
        time.sleep(10)
    for size in ['L', 'B', 'S', 'T']:
        gpus = selected[size]
        out = work / 'runs' / size
        done = out / 'COMPLETE.json'
        if done.exists():
            continue
        status('training', model=size, gpus=gpus, selected=selected)
        resume = out / 'last.pth'
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={gpus}',
                   str(root / 'tools/train_v2.py'), '--config', str(work/'configs'/f'{size}-g{gpus}-full.yml'),
                   '--resume' if resume.exists() else '--checkpoint',
                   str(resume if resume.exists() else work / 'weights' / f'v2-{size.lower()}-initial.pt'),
                   '--amp', 'auto', '--accumulate', str(8//gpus), '--patience', str(a.patience)]
        run(command, work / 'logs' / f'train-{size}.log', gpus)
        if not (out / 'best.pt').exists():
            raise RuntimeError('Training exited without a validated best EMA checkpoint')
        done.write_text(json.dumps(dict(completed=True, model=size, gpus=gpus)))
    status('complete', selected=selected)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('V2_SEQUENCE_FAILED', repr(error), flush=True)
        raise
