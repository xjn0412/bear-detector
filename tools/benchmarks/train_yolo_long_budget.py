"""Fresh 100-epoch default-augmentation run with the prior resolved optimizer.

Completed 50-epoch checkpoints were stripped and cannot be exact resumptions.
This restarts the same upstream model/seed with a new 100-epoch schedule,
patience20, and otherwise the recorded default-run settings. Original runs stay
immutable. Explicit AdamW avoids auto changing optimizer when budget increases.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True)
    a = p.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    root, repo = Path(spec['root']), Path(spec['repo'])
    baseline = Path(spec['baseline'])
    root.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_LONG_BUDGET_ROOT'] = str(root.resolve())
    import torch
    import yaml
    from ultralytics import YOLO
    from yolo_long_budget_trainer import LongBudgetTrainer
    old = torch.load(spec['previous_last'], map_location='cpu', weights_only=False)
    recorded = old['train_args']
    assert recorded['optimizer'] == 'AdamW', recorded['optimizer']
    kwargs = yaml.safe_load(Path(spec['previous_args']).read_text())
    kwargs.update(model=spec['initial_checkpoint'], data=str(baseline / 'data640/dataset.yaml'),
        epochs=100, patience=20, optimizer='AdamW', lr0=recorded['lr0'],
        # auto built AdamW with beta1=.9; train_args retains the unused .937.
        momentum=0.9, warmup_bias_lr=recorded['warmup_bias_lr'],
        device=[0, 1], project=str(root / 'runs'), name='native100',
        save_dir=str(root / 'runs/native100'),
        exist_ok=False, resume=False, save=True, save_period=10,
        plots=False, seed=42, deterministic=True)
    protocol = dict(model=spec['model'], dataset=spec['dataset'],
        initialization='Fresh upstream COCO checkpoint; new 100-epoch LR/augmentation schedule',
        initialization_sha256=digest(spec['initial_checkpoint']),
        baseline_last_sha256=digest(spec['previous_last']),
        old_checkpoint_optimizer_available=old.get('optimizer') is not None,
        resolved_optimizer='AdamW', resolved_lr0=recorded['lr0'], resolved_beta1=0.9,
        reason_optimizer_pinned='Preserve actual 50-epoch auto choice; longer auto budget may switch optimizer',
        epoch_limit=100, patience=20, seed=42, batch=kwargs['batch'], imgsz=kwargs['imgsz'],
        old_completed_epochs=50, preserve_original_50_epoch_run=True,
        test_annotations_sha256=digest(baseline / 'data640/test.json'), settings=kwargs)
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (root / 'resolved-settings.yml').write_text(yaml.safe_dump(kwargs, sort_keys=False))
    del old
    YOLO(spec['initial_checkpoint']).train(trainer=LongBudgetTrainer, **kwargs)
    run = root / 'runs/native100'
    rows = list(csv.DictReader((run / 'results.csv').open()))
    assert 1 <= len(rows) <= 100
    outcome = json.loads((run / 'training-outcome.json').read_text())
    assert outcome['completed_epochs'] == len(rows)
    full = torch.load(run / 'weights/last-resume.pt', map_location='cpu', weights_only=False)
    assert full['optimizer'] is not None and full['ema'] is not None
    assert full['epoch'] + 1 == len(rows)
    del full
    full_best = torch.load(run / 'weights/best-resume.pt', map_location='cpu', weights_only=False)
    selected_epoch = full_best['epoch'] + 1
    del full_best
    curves = [dict(epoch=int(r['epoch']), native_val_AP=float(r['metrics/mAP50-95(B)']) * 100,
                    native_val_AP50=float(r['metrics/mAP50(B)']) * 100) for r in rows]
    (root / 'curves.json').write_text(json.dumps(curves, indent=2))
    evaluation = root / 'evaluation'
    evaluation.mkdir(exist_ok=True)
    with (root / 'evaluation.log').open('w') as log:
        subprocess.run([sys.executable, str(repo / 'tools/benchmarks/evaluate_wildlife_yolo.py'),
            '--checkpoint', str(run / 'weights/best.pt'), '--annotations', str(baseline / 'data640/test.json'),
            '--images', str(baseline / 'data640/images/test'), '--output', str(evaluation)],
            cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    release = root / 'release'
    release.mkdir(exist_ok=True)
    target = release / (spec['model'] + '-' + spec['dataset'] + '-native100.pt')
    shutil.copyfile(run / 'weights/best.pt', target)
    manifest = dict(file=target.name, bytes=target.stat().st_size, sha256=digest(target),
                    epoch_limit=100, patience=20, completed_epochs=len(rows),
                    selected_epoch=selected_epoch)
    (release / 'weights-manifest.json').write_text(json.dumps(manifest, indent=2))
    (root / 'COMPLETE.json').write_text(json.dumps(dict(time=time.time(), **manifest), indent=2))
    print('COMPLETE', spec['model'], spec['dataset'], len(rows), flush=True)


if __name__ == '__main__':
    main()
