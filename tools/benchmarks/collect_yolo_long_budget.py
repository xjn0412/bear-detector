"""Report fresh max100/patience20 YOLO runs separately from fixed50 baselines."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--repo', required=True)
    a = p.parse_args()
    root, repo = Path(a.root).resolve(), Path(a.repo).resolve()
    evidence = repo / 'validation/benchmarks/yolo-long-budget-20261114'
    tasks = ['wildlife_yolo26x', 'wildlife_yolo11x', 'aquarium_yolo26x', 'voc2007_yolo26x']
    summaries, histories = {}, {}
    for task in tasks:
        run = root / task
        assert (run / 'COMPLETE.json').exists(), task
        spec = json.loads((root / (task + '.json')).read_text())
        base = Path(spec['baseline'])
        model = spec['model']
        old_key = 'YOLO26-x native' if model == 'yolo26x' else 'YOLO11-x native'
        old_report = json.loads((base / 'evaluation' / (model + '_native') / 'report.json').read_text())
        new_report = json.loads((run / 'evaluation/report.json').read_text())
        assert old_report['annotations_sha256'] == new_report['annotations_sha256']
        old_curve_file = base / 'curves.json'
        if not old_curve_file.exists():
            old_curve_file = base / 'evidence/curves.json'
        old_curve = json.loads(old_curve_file.read_text())[old_key]
        new_curve = json.loads((run / 'curves.json').read_text())
        outcome = json.loads((run / 'runs/native100/training-outcome.json').read_text())
        manifest = json.loads((run / 'release/weights-manifest.json').read_text())
        large_report_path = Path('/var/lib/sentinelx/bear-l-transfer-20261114') / spec['dataset'] / 'evaluation/bear_l.json'
        large = json.loads(large_report_path.read_text())
        bear = json.loads((base / 'evaluation/bear_b.json').read_text())
        assert bear['annotations_sha256'] == large['annotations_sha256'] == new_report['annotations_sha256']
        summary = dict(dataset=spec['dataset'], model=model,
            original50=dict(**{k: old_report[k] for k in ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'per_class_AP']},
                selected_epoch=max(old_curve, key=lambda r: r['native_val_AP'])['epoch'], completed_epochs=50),
            max100=dict(**{k: new_report[k] for k in ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'per_class_AP']},
                selected_epoch=manifest['selected_epoch'], completed_epochs=manifest['completed_epochs'],
                limit=100, patience=20),
            Bear_B50={k: bear[k] for k in ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large']},
            Bear_L50={k: large[k] for k in ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large']},
            max100_minus_original50_AP=new_report['AP'] - old_report['AP'],
            annotations_sha256=new_report['annotations_sha256'], outcome=outcome)
        summaries[task], histories[task] = summary, (old_curve, new_curve)
        out = evidence / task
        out.mkdir(parents=True, exist_ok=True)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        (out / 'curves.json').write_text(json.dumps(dict(original50=old_curve, max100=new_curve), indent=2))
        for src, name in [(run / 'protocol.json', 'protocol.json'),
                          (run / 'resolved-settings.yml', 'settings.yml'),
                          (run / 'runs/native100/training-outcome.json', 'training-outcome.json'),
                          (run / 'evaluation/report.json', 'test-report.json'),
                          (run / 'release/weights-manifest.json', 'weights-manifest.json')]:
            (out / name).write_text(src.read_text().replace(str(root), '/work/yolo-long-budget').replace(str(base), '/work/baseline/' + spec['dataset']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, (task, curves) in zip(axes.flat, histories.items()):
        for curve, label, color in zip(curves, ['Original 50 epochs', 'Max100 · patience20'], ['#748296', '#26876b']):
            ax.plot([r['epoch'] for r in curve], [r['native_val_AP'] for r in curve], label=label, color=color, lw=2)
            best = max(curve, key=lambda r: r['native_val_AP'])
            ax.scatter([best['epoch']], [best['native_val_AP']], color=color, s=25)
        ax.set_title(task.replace('_', ' · '), loc='left', weight='bold')
        ax.set_xlim(1, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Native validation AP50:95 (%)')
        ax.grid(alpha=.15)
        ax.legend(loc='lower right', frameon=False)
    fig.suptitle('YOLO default augmentation: longer training budget', fontsize=18, weight='bold')
    fig.supxlabel('Fresh runs from the same upstream checkpoint/seed; 100-epoch schedule, not an exact resume.\n'
                  'Early stopping can finish before100. Test results use one common held-out COCO evaluator.', fontsize=10)
    asset = repo / 'assets/benchmarks/yolo-long-budget'
    asset.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(asset.with_suffix('.svg'))
    fig.savefig(asset.with_suffix('.png'), dpi=160)
    table = ['| Dataset / model | Original50 AP | Max100 AP | Actual epochs | Selected epoch | AP difference |',
             '|---|---:|---:|---:|---:|---:|']
    comparisons = ['| Dataset | Bear-B50 | Bear-L50 | YOLO26x default50 | YOLO26x max100 |',
                   '|---|---:|---:|---:|---:|']
    for task, s in summaries.items():
        table.append(f'| {s["dataset"]} / {s["model"]} | {s["original50"]["AP"]:.2f} | {s["max100"]["AP"]:.2f} | '
                     f'{s["max100"]["completed_epochs"]} | {s["max100"]["selected_epoch"]} | {s["max100_minus_original50_AP"]:+.2f} |')
        if s['model'] == 'yolo26x':
            comparisons.append(f'| {s["dataset"]} | {s["Bear_B50"]["AP"]:.2f} | {s["Bear_L50"]["AP"]:.2f} | '
                               f'{s["original50"]["AP"]:.2f} | {s["max100"]["AP"]:.2f} |')
    doc = repo / 'docs/benchmarks/YOLO_LONG_BUDGET.md'
    text = doc.read_text().split('\n\n## Measured results')[0]
    text = text.replace('Status: fresh longer-budget runs are being prepared/trained; results are pending.',
                        'Status: all four longer-budget runs and common held-out test evaluations completed.')
    text += '\n\n## Measured results\n\n![Validation learning curves](../../assets/benchmarks/yolo-long-budget.svg)\n\n'
    text += '\n'.join(table) + '\n\nAP values are test percentages; differences are AP points. '
    text += 'Actual completed epochs are reported explicitly, including early-stopped runs. '
    text += 'The100-epoch schedule and late mosaic-closing point differ from the original50-epoch schedule.\n\n'
    text += '\n'.join(comparisons)
    text += '\n\nThis is a longer-budget comparison with Bear fixed at50 epochs. One seed, unequal model sizes and pretraining '
    text += 'remain limitations. No test labels selected an epoch or optimizer. VOC difficult boxes are ordinary positives '
    text += 'and use COCO AP rather than official VOC AP.\n\n'
    text += '[Full per-class/size metrics, epoch histories, resolved settings and hashes](../../validation/benchmarks/yolo-long-budget-20261114/).\n'
    doc.write_text(text)
    zh = '# YOLO 默认增强配方：更长训练预算\n\n四组实验完成。上限100个epoch，验证集20个epoch无改善则提前停止。'
    zh += '表中保留实际训练轮数；从同一原始COCO权重和种子重新训练，使用新的100轮调度。\n\n'
    zh += '\n'.join(table) + '\n\n' + '\n'.join(comparisons)
    zh += '\n\n以上为统一测试集COCO AP百分数，差值为百分点。Bear保持50轮，长预算YOLO单独列出。'
    zh += '只使用一个种子，参数量、预训练和优化实现不同；测试集不参与选轮次或超参数。'
    zh += 'VOC difficult作为普通正样本，指标不是官方VOC AP。\n\n[完整协议、曲线和统计](YOLO_LONG_BUDGET.md)\n'
    (doc.parent / 'YOLO_LONG_BUDGET.zh-CN.md').write_text(zh)
    (root / 'comparison-summary.json').write_text(json.dumps(summaries, indent=2))
    print('LONG-BUDGET COMPARISON COMPLETE', flush=True)


if __name__ == '__main__':
    main()
