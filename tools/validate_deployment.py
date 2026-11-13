"""Full COCO validation of a selected deployment backend, with finite guards.

Accuracy only: separate from isolated latency measurement. Uses the compiled
faster-coco-eval COCO metric, maximum300 detections/image, original category IDs.
"""
import argparse
import json
import math
from pathlib import Path
import time
from common import categories, digest
from deployment_backend import load_backend, image_tensor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['torch-fp32', 'torch-amp', 'tensorrt'], required=True)
    for k in ['config', 'checkpoint', 'annotations', 'images', 'output']:
        p.add_argument('--' + k, required=True)
    p.add_argument('--engine')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-images', type=int, default=0)
    a = p.parse_args()
    if a.batch_size < 1 or a.workers < 0 or a.max_images < 0:
        p.error('Invalid batch/workers/image limit')
    import torch
    from torch.utils.data import Dataset, DataLoader
    from faster_coco_eval import COCO, COCOeval_faster
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    ann = json.loads(Path(a.annotations).read_text())
    catids = [c['id'] for c in categories(ann)]
    selected = ann['images'][:a.max_images] if a.max_images else ann['images']
    if not selected:
        raise ValueError('Empty image list')
    forward, nc, extra = load_backend(a.backend, a.config, a.checkpoint, a.engine)
    if nc != len(catids):
        raise ValueError('Class schema does not match detector')

    class Images(Dataset):
        def __len__(self):
            return len(selected)

        def __getitem__(self, index):
            im = selected[index]
            x, sz = image_tensor(Path(a.images) / im['file_name'])
            if sz.tolist() != [im['width'], im['height']]:
                raise ValueError('Image dimensions mismatch')
            return x, sz, im['id']

    loader = DataLoader(Images(), batch_size=a.batch_size, num_workers=a.workers,
                        shuffle=False, pin_memory=True)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    done = 0
    started = time.time()

    def consume(labels, boxes, scores, ids):
        labels, boxes, scores = labels.cpu(), boxes.float().cpu(), scores.float().cpu()
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
            raise FloatingPointError('Nonfinite predictions')
        if labels.min() < 0 or labels.max() >= nc:
            raise ValueError('Class index out of range')
        boxes[:, :, 2:] -= boxes[:, :, :2]
        for image_id, ls, bs, ss in zip(ids.tolist(), labels.tolist(), boxes.tolist(), scores.tolist()):
            rows.extend(dict(image_id=image_id, category_id=catids[int(l)], bbox=b, score=float(s))
                        for l, b, s in zip(ls, bs, ss))

    with torch.inference_mode():
        for x, sz, ids in loader:
            x, sz = x.cuda(non_blocking=True), sz.cuda(non_blocking=True)
            if a.backend == 'tensorrt':
                for i in range(len(ids)):
                    consume(*forward(x[i:i+1], sz[i:i+1]), ids[i:i+1])
            else:
                consume(*forward(x, sz), ids)
            done += len(ids)
            if done % 200 == 0 or done == len(selected):
                out.with_suffix('.progress.json').write_text(json.dumps(dict(
                    images=done, total=len(selected), elapsed=time.time()-started)))
                print('VALIDATE', a.backend, done, '/', len(selected), flush=True)
    pred = out.with_suffix('.predictions.json')
    pred.write_text(json.dumps(rows, separators=(',', ':')))
    gt = COCO(a.annotations)
    gt.dataset.setdefault('info', {})
    dt = gt.loadRes(rows)
    ev = COCOeval_faster(gt, dt, 'bbox')
    ev.params.imgIds = [im['id'] for im in selected]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    names = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large']
    metrics = {k: float(v * 100) for k, v in zip(names, ev.stats[:6])}
    if not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError('Nonfinite metrics')
    report = dict(backend=a.backend, images=len(selected), full_coco_val2017=len(selected)==5000,
        checkpoint_sha256=digest(a.checkpoint), annotations_sha256=digest(a.annotations),
        predictions_sha256=digest(pred), batch_size=a.batch_size if a.backend!='tensorrt' else 1,
        input=640, metric='faster-coco-eval box AP50:95, maxDets100 standard COCO summary; top300 predictions',
        finite_outputs=True, elapsed_seconds=time.time()-started, **metrics, **extra)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
