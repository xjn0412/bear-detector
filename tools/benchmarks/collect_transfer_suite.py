"""Collect completed additional-transfer evidence and generate a factual report."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--repo', required=True)
    a = p.parse_args()
    root, repo = Path(a.root).resolve(), Path(a.repo).resolve()
    out = repo / 'validation/benchmarks/additional-transfer-20261114'
    reports = {}
    for dataset in ['aquarium', 'voc2007']:
        src, dst = root / dataset, out / dataset
        if not (src / 'COMPLETE.json').exists():
            raise ValueError('Experiment incomplete: ' + dataset)
        dst.mkdir(parents=True, exist_ok=True)
        for source, target in [(src / 'evaluation/summary.json', 'summary.json'),
                               (src / 'curves.json', 'curves.json'),
                               (src / 'dataset-manifest.json', 'dataset-manifest.json'),
                               (src / 'run_bear/initialization.json', 'bear-initialization.json'),
                               (src / 'run_bear/config.yml', 'bear-config.yml'),
                               (src / 'run_bear/classes.json', 'classes.json'),
                               (src / 'release/weights-manifest.json', 'weights-manifest.json'),
                               (src / 'release/SHA256SUMS', 'SHA256SUMS'),
                               (src / 'evaluation/bear_b.json', 'bear-test-report.json'),
                               (src / 'evaluation/yolo26x_shared/report.json', 'yolo26x-shared-test-report.json'),
                               (src / 'evaluation/yolo26x_native/report.json', 'yolo26x-native-test-report.json'),
                               (src / 'runs/yolo26x_shared/args.yaml', 'yolo26x-shared-args.yml'),
                               (src / 'runs/yolo26x_native/args.yaml', 'yolo26x-native-args.yml')]:
            text = source.read_text().replace(str(src), '/work/' + dataset)
            (dst / target).write_text(text)
        reports[dataset] = json.loads((dst / 'summary.json').read_text())
    subprocess.run([sys.executable, str(repo / 'tools/benchmarks/plot_transfer_suite.py'),
                    '--evidence', str(out), '--output', str(repo / 'assets/benchmarks/additional-transfer')], check=True)
    original = repo / 'docs/benchmarks/MULTIDATA_TRANSFER.md'
    protocol = original.read_text()
    # Keep the protocol's source/scope material while replacing its status only.
    protocol = protocol.replace('Status: training in progress. Final held-out test scores have not been published yet.',
                                'Status: all six 50-epoch runs, held-out evaluations and strict checkpoint export completed.')
    rows = ['| Dataset | Model / recipe | Selected epoch | Test AP50:95 | Test AP50 | Test AP75 |',
            '|---|---|---:|---:|---:|---:|']
    primary = ['| Dataset | Bear-B | Validation-selected YOLO26-x | Bear minus YOLO |',
               '|---|---:|---:|---:|']
    analysis = []
    sizes = ['| Dataset | Model / recipe | Small AP | Medium AP | Large AP |',
             '|---|---|---:|---:|---:|']
    def fmt(v):
        return '—' if v is None else f'{v:.2f}'
    for ds, summary in reports.items():
        title = 'Aquarium' if ds == 'aquarium' else 'VOC2007 · all boxes'
        models = summary['models']
        for n, model in models.items():
            rows.append(f'| {title} | {n} | {model["selected_epoch"]} | {model["AP"]:.2f} | {model["AP50"]:.2f} | {model["AP75"]:.2f} |')
            sizes.append('| ' + title + ' | ' + n + ' | ' + ' | '.join(fmt(model[k]) for k in ['AP_small', 'AP_medium', 'AP_large']) + ' |')
        selected = summary['yolo_recipe_selected_on_validation']
        bear, yolo = models['Bear-B']['AP'], models[selected]['AP']
        primary.append(f'| {title} | {bear:.2f} | {yolo:.2f} ({selected.split()[-1]}) | {bear-yolo:+.2f} |')
        analysis.append(f'{title}: validation selects **{selected}** among the two YOLO recipes. '
                        f'Bear-B scores **{bear:.2f}**, versus **{yolo:.2f}** test AP50:95 '
                        f'({bear-yolo:+.2f} points).')
        if 'bootstrap' in summary:
            ci = summary['bootstrap']['Bear_minus_YOLO_95CI'][selected]
            analysis.append(f'The paired Aquarium image-bootstrap 95% interval for this difference is '
                            f'**{ci[0]:+.2f} to {ci[1]:+.2f} AP points**. '
                            'It reflects this small test set conditional on the trained checkpoints.')
    additions = '\n\n## Measured held-out test results\n\n'
    additions += '![Common test accuracy and validation learning curves](../../assets/benchmarks/additional-transfer.svg)\n\n'
    additions += '\n'.join(primary) + '\n\nAll values are percentages; differences are AP points. '
    additions += 'The YOLO recipe in this summary is selected on validation, before comparing test scores.\n\n'
    additions += '\n'.join(rows) + '\n\n' + '\n\n'.join(analysis)
    additions += '\n\n' + '\n'.join(sizes)
    additions += '\n\nThese results compare the released checkpoints and these recipes. They do not '
    additions += 'isolate architecture quality, fully tuned optima, compute efficiency or training-seed variability. '
    additions += 'No GPU latency advantage is claimed. Inspect convergence at the 50-epoch boundary in the curves.\n\n'
    additions += '[Full recorded evidence](../../validation/benchmarks/additional-transfer-20261114/) '
    additions += 'includes native training configurations, initialization hashes, per-epoch validation histories, '
    additions += 'common evaluator reports, per-class AP, archive hashes and checkpoint hashes. '
    additions += 'Raw test predictions and full optimizer checkpoints are retained on the training machine.\n'
    original.write_text(protocol.split('\n\n## Measured held-out test results')[0] + additions)
    zh = '# Aquarium 和 VOC2007 微调对比\n\n'
    zh += '六组实验均完成 50 个 epoch。固定输入 640、全局 batch 16、随机种子 42；'
    zh += '每个模型使用各自验证集 AP50:95 最佳的 EMA 权重，在独立测试集上统一计算 COCO bbox AP。\n\n'
    zh += '![测试精度与验证集学习曲线](../../assets/benchmarks/additional-transfer.svg)\n\n'
    zh += '\n'.join(primary) + '\n\n' + '\n'.join(rows) + '\n\n'
    zh += '以上 AP 为百分数，差值为百分点；YOLO 的 shared/native 配方只按验证集选择，不按测试集挑选。'
    zh += 'Aquarium 为 448/127/63 张训练/验证/测试图像；VOC2007 为 2500/2508/4952 张。'
    zh += '清除了 3 张 VOC 精确重复图片和 2 个 Aquarium 零面积标注；两模型共享相同图片和有效标注。\n\n'
    zh += '**VOC difficult 标注在两模型中均作为普通正样本保留，因此是全标注 COCO 风格 AP，'
    zh += '不是官方 VOC difficult 忽略规则下的 AP，也不是 VOC2007 的 11 点 AP。**\n\n'
    zh += 'Bear 使用 DINOv3 预训练骨干及 COCO 检测权重，YOLO26-x 使用官方 Objects365→COCO 权重。'
    zh += '两者参数量、分类头迁移、骨干学习率、损失和优化实现不同，不能据此把差异完全归因于架构。'
    zh += '只使用一个训练种子；Aquarium 的 200 次配对图像 bootstrap 区间不代表训练随机性的区间。'
    zh += '本实验只比较精度，没有测量推理延迟优势。\n\n'
    zh += '[完整英文协议与分析](MULTIDATA_TRANSFER.md) · '
    zh += '[原始统计与配置](../../validation/benchmarks/additional-transfer-20261114/)\n'
    (original.parent / 'MULTIDATA_TRANSFER.zh-CN.md').write_text(zh)
    for filename, text in [('README.md', '\nAdditional Aquarium and VOC2007 fine-tuning comparisons: '
                            '[measured results, learning curves and protocol](docs/benchmarks/MULTIDATA_TRANSFER.md).\n'),
                           ('README.zh-CN.md', '\n新增 Aquarium 和 VOC2007 微调对比：'
                            '[实测结果、学习曲线与实验协议](docs/benchmarks/MULTIDATA_TRANSFER.zh-CN.md)。\n')]:
        path = repo / filename
        content = path.read_text()
        if 'docs/benchmarks/MULTIDATA_TRANSFER.' not in content:
            marker = '\n## Models and downloads' if filename == 'README.md' else None
            if marker and marker in content:
                content = content.replace(marker, text + marker, 1)
            else:
                content += '\n' + text
            path.write_text(content)
    print('COLLECTED', out, flush=True)


if __name__ == '__main__':
    main()
