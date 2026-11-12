"""Compare original and compact checkpoints on identical COCO images and precision modes.

This measures accuracy only. Concurrent workloads make these runs unsuitable for speed claims.
"""
import argparse
import contextlib
import hashlib
import json
import math
import time
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'reference', 'candidate', 'annotations', 'images', 'output']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--max-images', type=int, default=0, help='0 means the complete annotation file')
    p.add_argument('--max-drop', type=float, default=.1, help='Maximum tolerated AP-point drop per precision mode')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--stop-marker', action='append', default=[])
    a = p.parse_args()
    if a.batch_size < 1 or a.workers < 0 or a.max_images < 0 or a.max_drop < 0:
        p.error('Invalid batch/worker/image/tolerance argument')
    import torch
    import numpy as np
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from common import load_model, Inference, categories
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def progress(data):
        tmp = output.with_suffix('.progress.tmp')
        tmp.write_text(json.dumps(dict(time=time.time(), **data), indent=2))
        tmp.replace(output.with_suffix('.progress.json'))
    def check_stop():
        if any(Path(m).exists() for m in a.stop_marker):
            raise RuntimeError('Manual stop marker respected')
    check_stop()
    ann = json.loads(Path(a.annotations).read_text())
    cats = categories(ann)
    catids = [c['id'] for c in cats]
    selected = ann['images'][:a.max_images] if a.max_images else ann['images']
    if not selected:
        raise ValueError('Empty validation set')
    class Images(Dataset):
        def __len__(self): return len(selected)
        def __getitem__(self, index):
            row = selected[index]
            img = Image.open(Path(a.images) / row['file_name']).convert('RGB')
            w, h = img.size
            if (w, h) != (row['width'], row['height']):
                raise ValueError('Image dimensions differ from annotation')
            x = torch.from_numpy(np.array(img.resize((640, 640), Image.Resampling.BILINEAR), copy=True)).permute(2, 0, 1).float() / 255
            x = (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
            return x, torch.tensor([w, h], dtype=torch.int64), row['id']
    loader = DataLoader(Images(), batch_size=a.batch_size, num_workers=a.workers, pin_memory=a.device == 'cuda', shuffle=False)
    models = {}
    for name, checkpoint in [('reference', a.reference), ('candidate', a.candidate)]:
        cfg, core = load_model(a.config, checkpoint, deploy=True)
        if cfg.yaml_cfg['num_classes'] != len(catids):
            raise ValueError('Annotation/model class-count mismatch')
        assert all(q.dtype == torch.float32 for q in core.parameters() if q.is_floating_point())
        modes = ['fp32', 'amp'] if a.device == 'cuda' else ['fp32']
        for mode in modes:
            models[name + '-' + mode] = Inference.create(core, cfg.postprocessor, amp=mode == 'amp').to(a.device).eval()
    predictions = {k: [] for k in models}
    started = time.time()
    done = 0
    progress(dict(stage='inference',images_completed=0,images_total=len(selected)))
    with torch.inference_mode():
        for x, sizes, ids in loader:
            check_stop()
            x = x.to(a.device, non_blocking=True)
            sizes = sizes.to(a.device, non_blocking=True)
            for key, model in models.items():
                labels, boxes, scores = model(x, sizes)
                labels, boxes, scores = labels.cpu(), boxes.float().cpu(), scores.float().cpu()
                if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
                    raise FloatingPointError('Nonfinite output in ' + key)
                if labels.min() < 0 or labels.max() >= len(catids):
                    raise ValueError('Class index out of range')
                boxes[:, :, 2:] -= boxes[:, :, :2]
                for image_id, ls, bs, ss in zip(ids.tolist(), labels.tolist(), boxes.tolist(), scores.tolist()):
                    predictions[key].extend(dict(image_id=image_id,category_id=catids[int(label)],bbox=box,score=float(score)) for label,box,score in zip(ls,bs,ss))
            done += len(ids)
            if done % 50 == 0 or done == len(selected):
                progress(dict(stage='inference',images_completed=done,images_total=len(selected),elapsed_seconds=time.time()-started))
                print('INFERENCE',done,'/',len(selected),flush=True)
    # Free validation GPU allocations while COCO matching/accumulation runs on CPU.
    models.clear()
    del model, core, cfg, x, sizes, labels, boxes, scores
    if a.device == 'cuda': torch.cuda.empty_cache()
    gt = COCO(a.annotations)
    gt.dataset.setdefault('info', {})
    results = {}
    names = ['AP','AP50','AP75','AP_small','AP_medium','AP_large']
    for key in list(predictions):
        check_stop()
        progress(dict(stage='coco_evaluation',case=key,images_completed=done,images_total=len(selected)))
        rows = predictions.pop(key)
        predfile = output.with_name(output.stem + '-' + key + '.predictions.json')
        with open(predfile,'w') as f:json.dump(rows,f,separators=(',',':'))
        dt = gt.loadRes(rows)
        ev = COCOeval(gt,dt,'bbox')
        ev.params.imgIds = [row['id'] for row in selected]
        ev.evaluate();ev.accumulate();ev.summarize()
        results[key] = {name:float(value*100) for name,value in zip(names,ev.stats[:6])}
        assert all(math.isfinite(v) for v in results[key].values())
        del rows,dt,ev
    deltas = {mode:{metric:results['candidate-'+mode][metric]-results['reference-'+mode][metric] for metric in names} for mode in modes}
    passed = all(d['AP'] >= -a.max_drop for d in deltas.values())
    full = len(selected) == len(ann['images']) == 5000
    report = dict(passed=passed,gate='all outputs finite and AP drop <= threshold in each measured precision mode',max_allowed_AP_drop=a.max_drop,
                  full_COCO_val2017=full,images=len(selected),input_size=640,batch_size=a.batch_size,precision_modes=modes,
                  annotations_sha256=sha256(a.annotations),reference_sha256=sha256(a.reference),candidate_sha256=sha256(a.candidate),
                  results=results,deltas=deltas,device=a.device,torch=torch.__version__,timing_claim=False,
                  TensorRT_validated=False,scope='These checkpoints and COCO only; compact weight rounding requires separate validation on custom tasks')
    output.write_text(json.dumps(report,indent=2))
    progress(dict(stage='complete',passed=passed,images_completed=done,images_total=len(selected)))
    print(json.dumps(report),flush=True)
    if not passed:raise SystemExit(2)


if __name__ == '__main__':main()
