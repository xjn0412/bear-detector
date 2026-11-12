"""Single-image PyTorch inference with the same preprocessing used for evaluation."""
import argparse
import json
from pathlib import Path
from common import load_model, Inference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['config', 'checkpoint', 'image', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--classes', help='Prepared run classes.json, for names/original IDs')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--threshold', type=float, default=0.3)
    a = p.parse_args()
    if not 0 <= a.threshold <= 1:
        p.error('Threshold must be in [0,1]')
    if a.amp and a.device != 'cuda':
        p.error('--amp is supported on CUDA only')
    import torch
    import numpy as np
    from PIL import Image
    torch.set_num_threads(4)
    cfg, core = load_model(a.config, a.checkpoint, deploy=True)
    model = Inference.create(core, cfg.postprocessor, amp=a.amp).to(a.device)
    image = Image.open(a.image).convert('RGB')
    w, h = image.size
    arr = torch.from_numpy(np.array(image.resize((640, 640), Image.Resampling.BILINEAR), copy=True)).permute(2, 0, 1).float()/255
    x = ((arr-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]).unsqueeze(0).to(a.device)
    with torch.inference_mode():
        labels, boxes, scores = model(x, torch.tensor([[w,h]], dtype=torch.int64, device=a.device))
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise FloatingPointError('Nonfinite predictions')
    names = None
    if a.classes:
        schema = json.loads(Path(a.classes).read_text())
        if schema['num_classes'] != cfg.yaml_cfg['num_classes']:
            raise ValueError('classes.json does not match model')
        names = schema['classes']
    rows = []
    for label, box, score in zip(labels[0].tolist(), boxes[0].tolist(), scores[0].tolist()):
        if score >= a.threshold:
            d = {'class_index': label, 'xyxy': box, 'score': score}
            if names:
                d.update(name=names[label]['name'], category_id=names[label]['category_id'])
            rows.append(d)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'image': a.image, 'width': w, 'height': h, 'detections': rows}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
