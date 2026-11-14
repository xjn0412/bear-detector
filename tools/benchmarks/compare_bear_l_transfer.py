"""Compare finished Bear-L runs with the existing B/YOLO26-x test predictions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import numpy as np
from summarize_wildlife import evaluate
from run_bear_variant_transfer import digest


def compare(root, baseline, dataset):
    assert (root / 'COMPLETE.json').exists(), 'Bear-L incomplete: ' + dataset
    gt = json.loads((baseline / 'data640/test.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    assert digest(baseline / 'data640/test.json') == protocol['annotations']['test'], 'Test annotations changed'
    baseline_report = json.loads((baseline / 'evaluation/bear_b.json').read_text())
    assert baseline_report['annotations_sha256'] == protocol['annotations']['test'], 'B/L test data differ'
    prior = json.loads((baseline / 'evaluation/summary.json').read_text())
    curves_path = baseline / 'curves.json'
    if not curves_path.exists():
        curves_path = baseline / 'evidence/curves.json'
    curves = json.loads(curves_path.read_text())
    large_curve = json.loads((root / 'curves.json').read_text())
    names = {'Bear-B': 'Bear-B', 'YOLO26-x shared': 'YOLO26-x' if dataset == 'wildlife' else 'YOLO26-x shared',
             'YOLO26-x native': 'YOLO26-x native'}
    predictions = {'Bear-L': root / 'evaluation/bear_l.predictions.json',
                   'Bear-B': baseline / 'evaluation/bear_b.predictions.json',
                   'YOLO26-x shared': baseline / 'evaluation' / ('yolo26x' if dataset == 'wildlife' else 'yolo26x_shared') / 'predictions.json',
                   'YOLO26-x native': baseline / 'evaluation/yolo26x_native/predictions.json'}
    metrics, per_class = evaluate(gt, json.loads(predictions['Bear-L'].read_text()))
    best = max(large_curve, key=lambda r: r['native_val_AP'])
    summary = dict(dataset=dataset, images=len(gt['images']), boxes=len(gt['annotations']),
                   metric='Common held-out COCO bbox AP50:95; 101 recall points; maxDets100; top300',
                   models={'Bear-L': dict(**metrics, per_class_AP=per_class, selected_epoch=best['epoch'],
                                          best_native_val_AP=best['native_val_AP'])})
    all_curves = {'Bear-L': large_curve}
    for name, key in names.items():
        old = prior['models'][key]
        curve = curves[key]
        selected = max(curve, key=lambda r: r['native_val_AP'])
        summary['models'][name] = {k: old[k] for k in ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'per_class_AP']}
        summary['models'][name].update(selected_epoch=selected['epoch'], best_native_val_AP=selected['native_val_AP'])
        all_curves[name] = curve
    selected_yolo = max(['YOLO26-x shared', 'YOLO26-x native'], key=lambda n: summary['models'][n]['best_native_val_AP'])
    summary['yolo_recipe_selected_on_validation'] = selected_yolo
    summary['L_minus_B_AP'] = metrics['AP'] - summary['models']['Bear-B']['AP']
    summary['L_minus_selected_YOLO_AP'] = metrics['AP'] - summary['models'][selected_yolo]['AP']
    if len(gt['images']) < 1000:
        preds = {n: json.loads(f.read_text()) for n, f in predictions.items()}
        indexed = {n: defaultdict(list) for n in preds}
        anns = defaultdict(list)
        for ann in gt['annotations']:
            anns[ann['image_id']].append(ann)
        for name, rows in preds.items():
            for row in rows:
                indexed[name][row['image_id']].append(row)
        rng = np.random.default_rng(42)
        samples = {n: [] for n in preds}
        for b in range(200):
            boot = dict(info={}, licenses=[], categories=gt['categories'], images=[], annotations=[])
            bp = {n: [] for n in preds}
            aid = 0
            for new_id, k in enumerate(rng.integers(0, len(gt['images']), len(gt['images'])), 1):
                im = gt['images'][int(k)]
                boot['images'].append(dict(im, id=new_id))
                for ann in anns[im['id']]:
                    aid += 1
                    boot['annotations'].append(dict(ann, id=aid, image_id=new_id))
                for name in preds:
                    bp[name].extend(dict(pred, image_id=new_id) for pred in indexed[name][im['id']])
            for name in preds:
                samples[name].append(evaluate(boot, bp[name])[0]['AP'])
            if (b + 1) % 25 == 0:
                print(dataset, 'bootstrap', b + 1, flush=True)
        summary['bootstrap'] = dict(replicates=200, seed=42,
            AP_95CI={n: np.percentile(v, [2.5, 97.5]).tolist() for n, v in samples.items()},
            L_minus_baseline_95CI={n: np.percentile(np.array(samples['Bear-L']) - np.array(samples[n]), [2.5, 97.5]).tolist() for n in preds if n != 'Bear-L'},
            scope='Paired image sampling, conditional on checkpoints, not seed variability')
    return summary, all_curves


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'repo', 'wildlife-baseline', 'additional-baseline']:
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    root, repo = Path(a.root).resolve(), Path(a.repo).resolve()
    evidence = repo / 'validation/benchmarks/bear-l-transfer-20261114'
    summaries, histories = {}, {}
    for dataset in ['wildlife', 'aquarium', 'voc2007']:
        base = Path(a.wildlife_baseline) if dataset == 'wildlife' else Path(a.additional_baseline) / dataset
        summary, curves = compare(root / dataset, base, dataset)
        summaries[dataset], histories[dataset] = summary, curves
        out = evidence / dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        (out / 'curves.json').write_text(json.dumps(curves, indent=2))
        for src, name in [(root / dataset / 'protocol.json', 'protocol.json'),
                          (root / dataset / 'run_bear_l/config.yml', 'large-config.yml'),
                          (root / dataset / 'run_bear_l/initialization.json', 'initialization.json'),
                          (root / dataset / 'release/weights-manifest.json', 'weights-manifest.json'),
                          (base / 'dataset-manifest.json', 'dataset-manifest.json')]:
            (out / name).write_text(src.read_text().replace(str(root), '/work/large-transfer').replace(str(base), '/work/' + dataset + '/baseline'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    names = ['Bear-L', 'Bear-B', 'YOLO26-x shared', 'YOLO26-x native']
    colors = ['#724aa1', '#2765a4', '#208568', '#c57a28']
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout='constrained')
    titles = {'wildlife': 'African Wildlife', 'aquarium': 'Aquarium', 'voc2007': 'VOC2007 · all boxes'}
    for col, (dataset, summary) in enumerate(summaries.items()):
        ax = axes[0, col]
        vals = [summary['models'][n]['AP'] for n in names]
        ax.barh(range(4), vals, color=colors)
        ax.set_yticks(range(4), names)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_xlabel('Held-out COCO AP50:95 (%)')
        ax.set_title(titles[dataset] + f' · {summary["images"]:,} test images', weight='bold', loc='left')
        for i, v in enumerate(vals):
            ax.text(v + 1, i, f'{v:.2f}', va='center')
        ax = axes[1, col]
        for n, color in zip(names, colors):
            curve = histories[dataset][n]
            ax.plot([r['epoch'] for r in curve], [r['native_val_AP'] for r in curve], color=color, label=n,
                    ls='--' if 'native' in n else '-', lw=1.8)
        ax.set_xlim(1, 50)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Fine-tuning epoch')
        ax.set_ylabel('Native validation AP50:95 (%)')
        ax.grid(alpha=.15)
        ax.legend(fontsize=8, frameon=False, loc='lower right')
    fig.suptitle('Bear-L transfer comparison with Bear-B and YOLO26-x', fontsize=18, weight='bold')
    fig.supxlabel('50 epochs · seed 42 · identical audited splits and task-specific baseline recipe.\n'
                  'Wildlife global batch 32; Aquarium/VOC batch 16. One seed; unequal parameters and pretraining.', fontsize=10)
    asset = repo / 'assets/benchmarks/bear-l-transfer'
    asset.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(asset.with_suffix('.svg'))
    fig.savefig(asset.with_suffix('.png'), dpi=160)
    rows = ['| Dataset | Bear-L | Bear-B | YOLO26-x shared | YOLO26-x default | L minus validation-selected YOLO |',
            '|---|---:|---:|---:|---:|---:|']
    details = ['| Dataset | Model | Selected epoch | AP50:95 | AP50 | AP75 | Small AP | Medium AP | Large AP |',
               '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda v: '—' if v is None else f'{v:.2f}'
    for ds, summary in summaries.items():
        vals = ' | '.join(fmt(summary['models'][n]['AP']) for n in names)
        rows.append(f'| {titles[ds]} | {vals} | {summary["L_minus_selected_YOLO_AP"]:+.2f} |')
        for n in names:
            model = summary['models'][n]
            details.append(f'| {titles[ds]} | {n} | {model["selected_epoch"]} | ' + ' | '.join(fmt(model[k]) for k in ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large']) + ' |')
    notes = ('All values are percentages; differences are AP points. Validation selects each checkpoint and the YOLO recipe. '
             'Test scores do not select recipes or epochs. Bear-L starts from its original COCO FP32 EMA, with fresh class heads. '
             'It uses the prior Bear-B task recipe, including global batch32 for Wildlife and16 for Aquarium/VOC, 50 epochs and seed42. '
             'The baseline data and predictions are reused unchanged. Both Bear sizes retain FP32 decoder/losses with AMP bulk. '
             'Bear-L is substantially larger than Bear-B/YOLO26-x; DINOv3 representation pretraining versus YOLO26 Objects365→COCO '
             'and model-specific optimization remain confounders. No architecture-only or efficiency claim follows. '
             'VOC difficult boxes are ordinary positives, so this is all-box COCO AP, not official VOC AP. '
             'Paired image-bootstrap intervals for Wildlife/Aquarium are in the evidence; they do not measure training-seed uncertainty. '
             'An interval crossing zero does not establish a stable AP advantage.')
    text = '# Bear-L transfer comparison\n\nStatus: all three Bear-L fine-tuning runs and common test evaluations complete.\n\n'
    text += '![Test results and learning curves](../../assets/benchmarks/bear-l-transfer.svg)\n\n'
    text += '\n'.join(rows) + '\n\n' + notes + '\n\n' + '\n'.join(details)
    text += '\n\n[Recorded metrics, per-class AP, intervals, configurations and hashes](../../validation/benchmarks/bear-l-transfer-20261114/). '
    text += 'Plain original FP32 `.pt` weights and full optimizer checkpoints are retained on the training host.\n'
    (repo / 'docs/benchmarks/BEAR_L_TRANSFER.md').write_text(text)
    zh = '# Bear-L 微调对比\n\n三组 Bear-L 均完成 50 个 epoch，并按验证集选择最佳 EMA，统一测试集计算 COCO AP。\n\n'
    zh += '![实测结果与学习曲线](../../assets/benchmarks/bear-l-transfer.svg)\n\n' + '\n'.join(rows)
    zh += '\n\n以上 AP 为百分数，差值为百分点。YOLO 配方仅按验证集选择。沿用原先三组数据划分和训练配方：'
    zh += 'African Wildlife 的全局 batch32，Aquarium/VOC 的 batch16；均为640输入、种子42、50个epoch。'
    zh += 'VOC difficult 均作为普通正样本保留，结果不是官方 VOC AP。Bear-L 从原始 COCO FP32权重重新微调，'
    zh += '分类头重新初始化。模型大小、预训练和优化实现不同，不能把差异完全归因于架构。'
    zh += '小数据集的配对bootstrap区间见统计文件；区间跨零时不能认定稳定领先。\n\n'
    zh += '[详细指标和实验协议](BEAR_L_TRANSFER.md) · [配置、曲线、每类AP及校验信息](../../validation/benchmarks/bear-l-transfer-20261114/)\n'
    (repo / 'docs/benchmarks/BEAR_L_TRANSFER.zh-CN.md').write_text(zh)
    (root / 'comparison-summary.json').write_text(json.dumps(summaries, indent=2))
    print('COMPARISON COMPLETE', flush=True)


if __name__ == '__main__':
    main()
