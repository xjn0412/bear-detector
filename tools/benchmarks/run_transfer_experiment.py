"""Run a prepared Bear-B/YOLO26-x experiment and preserve measured evidence.

Requires the Bear training environment and a separate PYTHONPATH containing the
recorded Ultralytics release. Runs Bear and YOLO on disjoint pairs of GPUs.
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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'repo', 'bear-checkpoint', 'yolo-checkpoint', 'ultra-path']:
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--bear-gpus', required=True)
    ap.add_argument('--yolo-gpus', required=True)
    a = ap.parse_args()
    root, repo = Path(a.root).resolve(), Path(a.repo).resolve()
    py = sys.executable
    manifest = json.loads((root / 'dataset-manifest.json').read_text())
    base = os.environ.copy()
    base.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    bear_env = dict(base, CUDA_VISIBLE_DEVICES=a.bear_gpus)
    ultra_env = dict(base, CUDA_VISIBLE_DEVICES=a.yolo_gpus,
                     PYTHONPATH=a.ultra_path, YOLO_CONFIG_DIR=str(root / 'ultra-settings'))

    def run(cmd, name, env=base, cwd=repo):
        print('RUN', name, flush=True)
        with (root / (name + '.log')).open('w') as f:
            subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)

    if not (root / 'run_bear/config.yml').exists():
        run([py, 'tools/prepare_finetune.py', '--config', 'configs/B.yml',
         '--checkpoint', a.bear_checkpoint, '--train-json', str(root / 'data640/train.json'),
         '--val-json', str(root / 'data640/val.json'), '--train-images', str(root / 'data640/images/train'),
         '--val-images', str(root / 'data640/images/val'), '--output', str(root / 'run_bear'),
         '--epochs', '50', '--batch', '16', '--lr', '0.0001'], 'prepare_bear')
    cf = root / 'run_bear/config.yml'
    d = yaml.safe_load(cf.read_text())
    for split in ['train', 'val']:
        d[split + '_dataloader'].update(drop_last=False, num_workers=4, total_batch_size=16)
    d['train_dataloader']['dataset']['transforms']['ops'] = [
        op for op in d['train_dataloader']['dataset']['transforms']['ops']
        if op['type'] not in ['RandomPhotometricDistort', 'SanitizeBoundingBoxes']]
    # Preparation already clips/validates boxes. Flip + fixed resize need no
    # additional filter, which would remove positive subpixel Aquarium boxes.
    d.update(checkpoint_freq=50, warmup_iter=3 * math.ceil(manifest['counts']['train']['images'] / 16),
             flat_epoch=3, no_aug_epoch=0, lr_gamma=0.1)
    d['optimizer']['weight_decay'] = 1e-4
    cf.write_text(yaml.safe_dump(d, sort_keys=False))
    jobs_file = root / 'training-jobs.json'
    jobs = json.loads(jobs_file.read_text()) if jobs_file.exists() else {}
    class Existing:
        def __init__(self, pid, name):
            self.pid, self.name = pid, name
        def wait(self):
            while True:
                try:
                    if Path('/proc/' + str(self.pid) + '/stat').read_text().split()[2] == 'Z':
                        break
                except FileNotFoundError:
                    break
                time.sleep(5)
            return 0 if finished(self.name) else 1
    def finished(name):
        f = root / 'run_bear/training/log.txt' if name == 'bear_b' else root / 'runs' / name / 'results.csv'
        if not f.exists():
            return False
        if name == 'bear_b':
            return len(f.read_text().splitlines()) == 50
        return len(list(csv.DictReader(f.open()))) == 50
    def start(name, cmd, env):
        if name in jobs:
            pid = jobs[name]['pid']
            try:
                alive = Path('/proc/' + str(pid) + '/stat').read_text().split()[2] != 'Z'
            except FileNotFoundError:
                alive = False
            if alive or finished(name):
                print('ATTACH', name, pid, flush=True)
                return Existing(pid, name)
        log = (root / (name + '.log')).open('w')
        proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        jobs[name] = dict(pid=proc.pid, command=cmd, started=time.time(),
                          gpus=env['CUDA_VISIBLE_DEVICES'], log=str(root / (name + '.log')))
        (root / 'training-jobs.json').write_text(json.dumps(jobs, indent=2))
        return proc
    bear = start('bear_b', [py, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', '--',
                            'tools/train_custom.py', '--run', str(root / 'run_bear')], bear_env)
    for recipe in ['shared', 'native']:
        cmd = [py, 'tools/benchmarks/train_transfer_yolo.py', '--root', str(root),
               '--checkpoint', a.yolo_checkpoint, '--recipe', recipe]
        child = start('yolo26x_' + recipe, cmd, ultra_env)
        rc = child.wait()
        jobs['yolo26x_' + recipe]['returncode'] = rc
        (root / 'training-jobs.json').write_text(json.dumps(jobs, indent=2))
        if rc:
            raise RuntimeError('YOLO training failed: ' + recipe)
    rc = bear.wait()
    jobs['bear_b']['returncode'] = rc
    (root / 'training-jobs.json').write_text(json.dumps(jobs, indent=2))
    if rc:
        raise RuntimeError('Bear training failed')
    curves = {}
    rows = [json.loads(line) for line in (root / 'run_bear/training/log.txt').read_text().splitlines()]
    if len(rows) != 50:
        raise ValueError(('Incomplete Bear epochs', len(rows)))
    curves['Bear-B'] = [dict(epoch=r['epoch'] + 1, native_val_AP=r['test_coco_eval_bbox'][0] * 100) for r in rows]
    for recipe in ['shared', 'native']:
        rows = list(csv.DictReader((root / 'runs' / ('yolo26x_' + recipe) / 'results.csv').open()))
        if len(rows) != 50:
            raise ValueError(('Incomplete YOLO epochs', recipe, len(rows)))
        curves['YOLO26-x ' + recipe] = [dict(epoch=int(r['epoch']), native_val_AP=float(r['metrics/mAP50-95(B)']) * 100) for r in rows]
    (root / 'curves.json').write_text(json.dumps(curves, indent=2))
    evaluation = root / 'evaluation'
    evaluation.mkdir(exist_ok=True)
    best = root / 'run_bear/training/best_stg1.pth'
    if not best.exists():
        raise FileNotFoundError(best)
    run([py, 'tools/validate_deployment.py', '--backend', 'torch-amp', '--config', str(cf),
         '--checkpoint', str(best), '--annotations', str(root / 'data640/test.json'),
         '--images', str(root / 'data640/images/test'), '--output', str(evaluation / 'bear_b.json'),
         '--batch-size', '8', '--workers', '4'], 'evaluate_bear', bear_env)
    for recipe in ['shared', 'native']:
        run([py, 'tools/benchmarks/evaluate_wildlife_yolo.py', '--checkpoint',
             str(root / 'runs' / ('yolo26x_' + recipe) / 'weights/best.pt'),
             '--annotations', str(root / 'data640/test.json'), '--images', str(root / 'data640/images/test'),
             '--output', str(evaluation / ('yolo26x_' + recipe))], 'evaluate_yolo_' + recipe, ultra_env)
    bootstrap = '200' if len(json.loads((root / 'data640/test.json').read_text())['images']) < 1000 else '0'
    run([py, 'tools/benchmarks/summarize_transfer.py', '--root', str(root), '--bootstrap', bootstrap], 'summarize', ultra_env)
    import torch
    release = root / 'release'
    release.mkdir(exist_ok=True)
    ck = torch.load(best, map_location='cpu', weights_only=False)
    state = ck['ema']['module']
    metadata = dict(model='Bear-B', dataset=manifest['dataset'], num_classes=d['num_classes'],
                    best_epoch=ck['last_epoch'] + 1, storage='original FP32 EMA', input_size=640,
                    config=d, classes=json.loads((root / 'run_bear/classes.json').read_text()))
    target = release / ('bear-b-' + manifest['dataset'] + '-ema.pt')
    torch.save(dict(model=state, last_epoch=ck['last_epoch'], metadata=metadata), target)
    saved = torch.load(target, map_location='cpu', weights_only=False)['model']
    assert set(saved) == set(state)
    assert all(torch.equal(saved[k], state[k]) and saved[k].dtype == state[k].dtype for k in state)
    sys.path.insert(0, str(repo / 'tools'))
    from common import load_model
    _, verified_model = load_model(cf, target)
    assert all(not t.is_floating_point() or t.dtype == torch.float32 for t in saved.values())
    del verified_model, saved
    for recipe in ['shared', 'native']:
        shutil.copyfile(root / 'runs' / ('yolo26x_' + recipe) / 'weights/best.pt',
                        release / ('yolo26x-' + manifest['dataset'] + '-' + recipe + '.pt'))
    for src, dest in [(cf, 'bear-b-' + manifest['dataset'] + '.yml'),
                      (root / 'run_bear/classes.json', 'classes.json'),
                      (root / 'dataset-manifest.json', 'dataset-manifest.json'),
                      (root / 'curves.json', 'curves.json'),
                      (evaluation / 'summary.json', 'summary.json')]:
        if src == cf:
            (release / dest).write_text(src.read_text().replace(str(root), '/work/' + manifest['dataset']))
        else:
            shutil.copyfile(src, release / dest)
    files = {f.name: dict(bytes=f.stat().st_size, sha256=sha(f)) for f in release.iterdir() if f.is_file()}
    (release / 'weights-manifest.json').write_text(json.dumps(dict(files=files, bear_ema_tensors_exact=True), indent=2))
    (release / 'SHA256SUMS').write_text(''.join(v['sha256'] + '  ' + n + '\n' for n, v in sorted(files.items())))
    (root / 'COMPLETE.json').write_text(json.dumps(dict(time=time.time(), status='measured and exported', summary=str(evaluation / 'summary.json'))))
    print((evaluation / 'summary.json').read_text(), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
