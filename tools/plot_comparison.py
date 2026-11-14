"""Render sourced comparison figures; requires matplotlib (documentation only)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'validation/benchmarks/comparison-20261114/comparison.json'
OUT = ROOT / 'assets/benchmarks'
COLORS = {'Bear': '#d34b37', 'YOLO26': '#4f60b5', 'YOLO11': '#a3a7bd',
          'DEIMv2': '#008c83', 'D-FINE': '#c6a467', 'D-FINE O365': '#9b6bb3',
          'RT-DETRv2': '#608aa5', 'YOLO27 preview': '#26374b'}


def style(ax):
    ax.set_facecolor('#f7f9fc')
    ax.grid(color='#dfe4ec', linewidth=.7, alpha=.8)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#bbc4d1')
    ax.tick_params(colors='#48576b')


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (name + '.png'), dpi=180, facecolor='white')
    fig.savefig(OUT / (name + '.svg'), facecolor='white')
    plt.close(fig)


def main():
    data = json.loads(DATA.read_text())
    rows = data['models'] + data.get('preview_models', [])
    bears = [r for r in rows if r['family'] == 'Bear']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6))
    fig.subplots_adjust(left=.065, right=.97, top=.79, bottom=.20, wspace=.22)
    for ax, field, xlabel in zip(axes, ['params_m', 'gflops'],
                                ['Parameters (millions; log scale)', 'Compute (GFLOPs; log scale)']):
        style(ax)
        for family, color in COLORS.items():
            points = [r for r in rows if r['family'] == family]
            points.sort(key=lambda r: r[field])
            ax.plot([r[field] for r in points], [r['AP'] for r in points],
                    color=color, marker='*' if family == 'Bear' else 'o',
                    markersize=11 if family == 'Bear' else 5, linewidth=2.3 if family == 'Bear' else 1.3,
                    label=family, zorder=5 if family == 'Bear' else 2,
                    linestyle='--' if family in ['D-FINE O365', 'YOLO27 preview'] else '-',
                    markerfacecolor='white' if family == 'YOLO27 preview' else color)
            if family == 'Bear':
                for r in points:
                    ax.annotate(r['model'].replace('Bear-', ''), (r[field], r['AP']),
                                xytext=(4, 7), textcoords='offset points', color=color,
                                weight='bold', fontsize=10)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylabel('COCO val2017 box AP50:95')
        ax.set_ylim(38, 62)
    axes[0].set_xlim(2, 280); axes[0].set_xticks([3, 10, 30, 100, 250])
    axes[1].set_xlim(5, 800); axes[1].set_xticks([5, 20, 100, 300, 700])
    fig.suptitle('Bear Detector | Accuracy and model cost', x=.065, ha='left',
                 y=.965, fontsize=21, weight='bold', color='#172b43')
    fig.text(.065, .906, '640 input • Bear measured; other families author-reported • Snapshot 2026-11-14',
             color='#526176', fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, .875),
               ncol=4, frameon=False, fontsize=9, columnspacing=1.0)
    fig.text(.065, .085, 'Parameter conventions and FLOPs profilers differ; these curves are descriptive, not controlled architecture ablations.',
             color='#526176', fontsize=9)
    fig.text(.065, .047, 'YOLO26: NMS-head AP. O365: extra detection pretraining. YOLO27: unreleased preliminary figures. No external speed comparison.',
             color='#526176', fontsize=9)
    save(fig, 'comparison-efficiency')

    entries = [dict(model=r['model'], AP=r['AP'], own=True) for r in bears]
    entries += [dict(model=r['model'], AP=r['AP'], own=False) for r in data['leaderboard_selection'] if r['leaderboard']]
    entries += [dict(model='DEIMv2-X (640)', AP=57.8, own=False),
                dict(model='YOLO26-x (640, NMS)', AP=57.5, own=False)]
    entries.sort(key=lambda r: r['AP'])
    fig, ax = plt.subplots(figsize=(12, 7.0))
    fig.subplots_adjust(left=.30, right=.93, top=.80, bottom=.19)
    style(ax)
    ax.barh(range(len(entries)), [r['AP'] for r in entries], height=.60,
            color=[COLORS['Bear'] if r['own'] else '#8294b1' for r in entries])
    ax.set_yticks(range(len(entries)), [r['model'] for r in entries])
    ax.set_xlim(0, 70)
    ax.set_xlabel('COCO val2017 box AP50:95 (0–100 points)')
    for i, r in enumerate(entries):
        ax.text(r['AP']+.5, i, f"{r['AP']:.2f}", va='center', fontsize=10, weight='bold')
    fig.suptitle('Where Bear sits among published results', x=.055, ha='left', y=.96,
                 fontsize=21, weight='bold', color='#172b43')
    fig.text(.055, .905, 'Selected SOTA2 entries verified against papers + 640 deployment baselines; no official Bear rank.',
             fontsize=10, color='#526176')
    fig.text(.055, .085, 'Red: our 640 single-scale measurements. Gray: published results; pretraining, resolution and TTA differ.',
             color='#526176', fontsize=9)
    fig.text(.055, .047, 'Focal-Stable-DINO: no TTA; DINO: TTA; Grounding DINO: enlarged input, COCO fine-tuned. Snapshot 2026-11-14.',
             color='#526176', fontsize=9)
    save(fig, 'comparison-leaderboard')

    fig, ax = plt.subplots(figsize=(11, 6.3))
    fig.subplots_adjust(left=.09, right=.96, top=.80, bottom=.20)
    style(ax)
    for field, metric, label, color in [('latency_trt_ms', 'AP_TRT', 'TensorRT 10.13.3.9', '#d34b37'),
                                        ('latency_torch_ms', 'AP', 'PyTorch AMP', '#4f60b5')]:
        ax.scatter([r[field] for r in bears], [r[metric] for r in bears], s=85, color=color, label=label)
        for r in bears:
            dx, dy = (6, 6)
            if field == 'latency_torch_ms' and r['model'] == 'Bear-B': dx, dy = (7, -15)
            ax.annotate(r['model'].replace('Bear-', ''), (r[field], r[metric]),
                        xytext=(dx, dy), textcoords='offset points', color=color, weight='bold')
    ax.set_xlim(0, 18.3); ax.set_ylim(55.8, 61)
    ax.set_xlabel('Measured GPU inference latency (ms, batch 1, RTX 5090)')
    ax.set_ylabel('COCO val2017 box AP50:95')
    ax.legend(loc='lower right', frameon=False)
    fig.suptitle('Bear family | Accuracy versus measured latency', x=.09, ha='left', y=.96,
                 fontsize=20, weight='bold', color='#172b43')
    fig.text(.09, .905, 'Same GPU and 640 input • Backend-specific AP • Full FP32 decoder preserved', color='#526176')
    fig.text(.09, .085, 'Latency: batch 1, eager, no CUDA Graph; excludes preprocessing and host transfers. TRT includes device-input copies.',
             color='#526176', fontsize=9)
    fig.text(.09, .047, 'Accuracy: full 5,000 images; PyTorch batch 8, TRT batch 1. External models were not timed on this hardware.',
             color='#526176', fontsize=9)
    save(fig, 'comparison-latency')


if __name__ == '__main__':
    main()
