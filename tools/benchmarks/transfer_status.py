"""Show completed epochs and native validation AP for each experiment."""
import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', required=True)
a = p.parse_args()
result = {}
for root in [Path(a.root) / n for n in ['aquarium', 'voc2007']]:
    dataset = {}
    bear = root / 'run_bear/training/log.txt'
    if bear.exists():
        rows = [json.loads(line) for line in bear.read_text().splitlines() if line.strip()]
        if rows:
            best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
            dataset['Bear-B'] = dict(epochs=len(rows), latest_AP=rows[-1]['test_coco_eval_bbox'][0] * 100,
                                     best_AP=best['test_coco_eval_bbox'][0] * 100, best_epoch=best['epoch'] + 1)
    for recipe in ['shared', 'native']:
        f = root / 'runs' / ('yolo26x_' + recipe) / 'results.csv'
        if f.exists():
            rows = list(csv.DictReader(f.open()))
            if rows:
                best = max(rows, key=lambda r: float(r['metrics/mAP50-95(B)']))
                dataset['YOLO26-x ' + recipe] = dict(epochs=len(rows),
                    latest_AP=float(rows[-1]['metrics/mAP50-95(B)']) * 100,
                    best_AP=float(best['metrics/mAP50-95(B)']) * 100,
                    best_epoch=int(best['epoch']), elapsed_seconds=float(rows[-1]['time']))
    dataset['complete'] = (root / 'COMPLETE.json').exists()
    result[root.name] = dataset
print(json.dumps(result, indent=2))
