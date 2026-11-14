"""Common held-out COCO metrics and paired image-bootstrap for new transfer sets."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from summarize_wildlife import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--bootstrap', type=int, default=200)
    a = p.parse_args()
    root = Path(a.root)
    dataset = json.loads((root / 'data640/test.json').read_text())
    files = {'Bear-B': root / 'evaluation/bear_b.predictions.json',
             'YOLO26-x shared': root / 'evaluation/yolo26x_shared/predictions.json',
             'YOLO26-x native': root / 'evaluation/yolo26x_native/predictions.json'}
    preds = {n: json.loads(f.read_text()) for n, f in files.items()}
    curves = json.loads((root / 'curves.json').read_text())
    summary = dict(dataset=json.loads((root / 'dataset-manifest.json').read_text())['dataset'],
                   metric='Held-out COCO bbox AP50:95; 101 recall points; maxDets100; top300 predictions',
                   images=len(dataset['images']), boxes=len(dataset['annotations']), models={})
    for n, ps in preds.items():
        metrics, pc = evaluate(dataset, ps)
        best = max(curves[n], key=lambda x: x['native_val_AP'])
        summary['models'][n] = dict(**metrics, per_class_AP=pc, selected_epoch=best['epoch'],
                                     best_native_val_AP=best['native_val_AP'])
    summary['yolo_recipe_selected_on_validation'] = max(
        ['YOLO26-x shared', 'YOLO26-x native'], key=lambda n: summary['models'][n]['best_native_val_AP'])
    annotations = defaultdict(list)
    for ann in dataset['annotations']:
        annotations[ann['image_id']].append(ann)
    indexed = {n: defaultdict(list) for n in preds}
    for n, ps in preds.items():
        for pred in ps:
            indexed[n][pred['image_id']].append(pred)
    rng = np.random.default_rng(42)
    samples = {n: [] for n in preds}
    for b in range(a.bootstrap):
        selected = rng.integers(0, len(dataset['images']), len(dataset['images']))
        boot = dict(info={}, licenses=[], categories=dataset['categories'], images=[], annotations=[])
        bp = {n: [] for n in preds}
        aid = 0
        for new_id, k in enumerate(selected, 1):
            im = dataset['images'][int(k)]
            old = im['id']
            boot['images'].append(dict(im, id=new_id))
            for ann in annotations[old]:
                aid += 1
                boot['annotations'].append(dict(ann, id=aid, image_id=new_id))
            for n in preds:
                bp[n].extend(dict(pred, image_id=new_id) for pred in indexed[n][old])
        for n in preds:
            samples[n].append(evaluate(boot, bp[n])[0]['AP'])
        if (b + 1) % 25 == 0:
            print('bootstrap', b + 1, '/', a.bootstrap, flush=True)
    if a.bootstrap:
        summary['bootstrap'] = dict(replicates=a.bootstrap, seed=42,
            method='Paired image percentile 95% CI conditional on checkpoints, not seed variability',
            AP_95CI={n: np.percentile(v, [2.5, 97.5]).tolist() for n, v in samples.items()},
            Bear_minus_YOLO_95CI={n: np.percentile(np.array(samples['Bear-B']) - np.array(samples[n]), [2.5, 97.5]).tolist()
                                 for n in preds if n != 'Bear-B'})
    out = root / 'evaluation/summary.json'
    out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
