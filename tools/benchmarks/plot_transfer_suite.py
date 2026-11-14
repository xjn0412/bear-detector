"""Plot only measured common-test metrics and native validation histories."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evidence', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
evidence, output = Path(a.evidence), Path(a.output)
output.parent.mkdir(parents=True, exist_ok=True)
names = ['Bear-B', 'YOLO26-x shared', 'YOLO26-x native']
labels = ['Bear-B', 'YOLO26-x · shared', 'YOLO26-x · default']
colors = ['#2765a4', '#208568', '#c57a28']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
for col, (dataset, title) in enumerate([('aquarium', 'Aquarium'), ('voc2007', 'VOC2007 · all boxes')]):
    root = evidence / dataset
    summary = json.loads((root / 'summary.json').read_text())
    curves = json.loads((root / 'curves.json').read_text())
    ax = axes[0, col]
    values = [summary['models'][n]['AP'] for n in names]
    errors = None
    intervals = summary.get('bootstrap', {}).get('AP_95CI')
    if intervals:
        errors = [[max(0, v - intervals[n][0]) for n, v in zip(names, values)],
                  [max(0, intervals[n][1] - v) for n, v in zip(names, values)]]
    y = np.arange(len(names))
    ax.barh(y, values, color=colors, xerr=errors, error_kw=dict(capsize=3, lw=1))
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 103)
    ax.set_xlabel('Held-out test COCO AP50:95 (%)')
    ax.set_title(f'{title} · {summary["images"]:,} test images', loc='left', weight='bold')
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    for i, v in enumerate(values):
        right = intervals[names[i]][1] if intervals else v
        ax.text(right + 1, i, f'{v:.2f}', va='center', weight='bold')
    ax = axes[1, col]
    for n, label, color in zip(names, labels, colors):
        rows = curves[n]
        x, val = [r['epoch'] for r in rows], [r['native_val_AP'] for r in rows]
        ax.plot(x, val, label=label, color=color, lw=2, ls='--' if 'native' in n else '-')
        k = int(np.argmax(val))
        ax.scatter([x[k]], [val[k]], color=color, s=25, zorder=3)
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Fine-tuning epoch')
    ax.set_ylabel('Native validation AP50:95 (%)')
    ax.grid(alpha=.15)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
fig.suptitle('Bear-B transfer comparison with YOLO26-x', fontsize=18, weight='bold')
fig.supxlabel('50 epochs · seed 42 · global batch 16 · fixed 640. Checkpoints selected using validation only.\n'
              'Common test evaluator; Aquarium bars include paired image-bootstrap 95% CI (200 samples).\n'
              'Different model sizes, pretraining and native optimization; one training seed.', fontsize=9)
fig.savefig(output.with_suffix('.svg'))
fig.savefig(output.with_suffix('.png'), dpi=160)
