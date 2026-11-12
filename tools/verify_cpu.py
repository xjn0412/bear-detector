"""Run the documented offline pipeline in a temporary directory, without GPU use."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from common import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx', action='store_true', help='Also export/check ONNX; requires requirements-export.txt')
    p.add_argument('--output', default='results/cpu-verification.json')
    a = p.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    records = []
    with tempfile.TemporaryDirectory(prefix='ccd-verify-') as tmp:
        t = Path(tmp)
        commands = [
            ['tests/test_classes.py'], ['tests/test_sampling.py'],
            ['tools/initialize.py', '--config', 'configs/T.yml', '--random-init', '--output', str(t/'initial.pt')],
            ['tools/make_smoke_data.py', '--output', str(t/'data')],
            ['tools/prepare_finetune.py', '--config', 'configs/T.yml', '--checkpoint', str(t/'initial.pt'),
             '--train-json', str(t/'data/train.json'), '--val-json', str(t/'data/val.json'),
             '--train-images', str(t/'data/images'), '--val-images', str(t/'data/images'),
             '--output', str(t/'run'), '--epochs', '2', '--batch', '2'],
            ['tools/predict.py', '--config', str(t/'run/config.yml'), '--checkpoint', str(t/'run/initial.pth'),
             '--classes', str(t/'run/classes.json'), '--image', str(t/'data/images/4.jpg'),
             '--device', 'cpu', '--output', str(t/'predictions.json')],
        ]
        if a.onnx:
            commands.append(['tools/export_engine.py', '--config', str(t/'run/config.yml'),
                             '--checkpoint', str(t/'run/initial.pth'), '--output', str(t/'export')])
        for cmd in commands:
            result = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
            records.append({'tool': cmd[0], 'exit_code': result.returncode})
            if result.returncode:
                print(result.stdout[-4000:] + result.stderr[-4000:], file=sys.stderr)
                raise RuntimeError('Verification failed: ' + cmd[0])
            print('PASS', cmd[0], flush=True)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'passed': True, 'scope': 'CPU offline wiring; no training convergence or GPU speed claim', 'commands': records}, indent=2))


if __name__ == '__main__':
    main()
