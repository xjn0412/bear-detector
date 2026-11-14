"""Fine-tune Bear-L with an existing Bear-B experiment's exact data and recipe.

The baseline remains immutable. Classification heads are reset, training starts
from the supplied COCO detector, and validation selects the final EMA checkpoint.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import yaml


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'root', 'repo', 'checkpoint', 'gpus', 'dataset']:
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    root, baseline, repo = Path(a.root).resolve(), Path(a.baseline).resolve(), Path(a.repo).resolve()
    root.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    old = yaml.safe_load((baseline / 'run_bear/config.yml').read_text())
    batch = old['train_dataloader']['total_batch_size']
    epochs = old['epoches']
    assert epochs == 50 and batch % len(a.gpus.split(',')) == 0
    run = root / 'run_bear_l'

    def execute(command, name):
        print('RUN', name, flush=True)
        with (root / (name + '.log')).open('w') as log:
            subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    execute([py, 'tools/prepare_finetune.py', '--config', 'configs/L.yml',
             '--checkpoint', a.checkpoint, '--train-json', str(baseline / 'data640/train.json'),
             '--val-json', str(baseline / 'data640/val.json'),
             '--train-images', str(baseline / 'data640/images/train'),
             '--val-images', str(baseline / 'data640/images/val'), '--output', str(run),
             '--epochs', str(epochs), '--batch', str(batch), '--lr', str(old['optimizer']['lr'])], 'prepare_l')
    # B and L differ only in their ConvNeXt variant after the recipe is fixed.
    fresh = yaml.safe_load((run / 'config.yml').read_text())
    d = copy.deepcopy(old)
    d['DeepStageHybrid'] = fresh['DeepStageHybrid']
    assert d['DeepStageHybrid']['variant'] == 'convnext_large'
    d['output_dir'] = str(run / 'training')
    for split in ['train', 'val']:
        d[split + '_dataloader']['dataset']['ann_file'] = str(run / (split + '.json'))
        d[split + '_dataloader']['dataset']['img_folder'] = str(baseline / 'data640/images' / split)
    (run / 'config.yml').write_text(yaml.safe_dump(d, sort_keys=False))
    protocol = dict(model='Bear-L', dataset=a.dataset, seed=42, epochs=epochs, batch=batch,
                    gpus=a.gpus, baseline=str(baseline),
                    baseline_config_sha256=digest(baseline / 'run_bear/config.yml'),
                    initialization_sha256=digest(a.checkpoint),
                    class_heads_reset=True, recipe='Exact baseline Bear-B configuration except L backbone and output paths',
                    input='RGB, direct bilinear 640, ImageNet normalization',
                    annotations={s: digest(baseline / 'data640' / (s + '.json')) for s in ['train', 'val', 'test']})
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    command = [py, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=' + str(len(a.gpus.split(','))), '--',
               'tools/train_custom.py', '--run', str(run)]
    execute(command, 'train_l')
    rows = [json.loads(line) for line in (run / 'training/log.txt').read_text().splitlines()]
    assert len(rows) == epochs, ('Incomplete epochs', len(rows), epochs)
    curves = [dict(epoch=r['epoch'] + 1, native_val_AP=r['test_coco_eval_bbox'][0] * 100,
                   native_val_AP50=r['test_coco_eval_bbox'][1] * 100) for r in rows]
    (root / 'curves.json').write_text(json.dumps(curves, indent=2))
    best = run / 'training/best_stg1.pth'
    assert best.is_file()
    evaluation = root / 'evaluation'
    evaluation.mkdir(exist_ok=True)
    execute([py, 'tools/validate_deployment.py', '--backend', 'torch-amp', '--config', str(run / 'config.yml'),
             '--checkpoint', str(best), '--annotations', str(baseline / 'data640/test.json'),
             '--images', str(baseline / 'data640/images/test'), '--output', str(evaluation / 'bear_l.json'),
             '--batch-size', '8', '--workers', '4'], 'evaluate_l')
    import torch
    ck = torch.load(best, map_location='cpu', weights_only=False)
    state = ck['ema']['module']
    release = root / 'release'
    release.mkdir(exist_ok=True)
    target = release / ('bear-l-' + a.dataset + '-ema.pt')
    torch.save(dict(model=state, last_epoch=ck['last_epoch'], metadata=dict(
        model='Bear-L', dataset=a.dataset, num_classes=d['num_classes'], config=d,
        best_epoch=ck['last_epoch'] + 1, storage='Original FP32 EMA', protocol=protocol)), target)
    saved = torch.load(target, map_location='cpu', weights_only=False)['model']
    assert set(saved) == set(state)
    assert all(torch.equal(saved[k], state[k]) and saved[k].dtype == state[k].dtype for k in state)
    assert all(not t.is_floating_point() or t.dtype == torch.float32 for t in saved.values())
    sys.path.insert(0, str(repo / 'tools'))
    from common import load_model
    _, model = load_model(run / 'config.yml', target)
    parameters = sum(p.numel() for p in model.parameters())
    del model, saved
    shutil.copyfile(run / 'classes.json', release / 'classes.json')
    portable = (run / 'config.yml').read_text().replace(str(root), '/work/' + a.dataset + '/large').replace(str(baseline), '/work/' + a.dataset + '/baseline')
    (release / ('bear-l-' + a.dataset + '.yml')).write_text(portable)
    manifest = dict(model='Bear-L', dataset=a.dataset, parameters=parameters,
                    selected_epoch=ck['last_epoch'] + 1, ema_tensors_bitwise_exact=True, strict_load=True,
                    files={f.name: dict(bytes=f.stat().st_size, sha256=digest(f)) for f in release.iterdir() if f.is_file()})
    (release / 'weights-manifest.json').write_text(json.dumps(manifest, indent=2))
    (release / 'SHA256SUMS').write_text(''.join(v['sha256'] + '  ' + n + '\n' for n, v in sorted(manifest['files'].items())))
    (root / 'COMPLETE.json').write_text(json.dumps(dict(time=time.time(), dataset=a.dataset,
        selected_epoch=ck['last_epoch'] + 1, evaluation=str(evaluation / 'bear_l.json'), checkpoint=str(target)), indent=2))
    print('COMPLETE', a.dataset, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
