"""Audit Aquarium (ODinW) or VOC2007 and create identical fixed640 COCO/YOLO data.

VOC difficult boxes are retained as ordinary positives for both models. This is
an all-box COCO transfer experiment, not the official VOC difficult/11-point AP.
Original train/val/test splits are kept, except exact RGB duplicates (priority
test > val > train). No test images select checkpoints or training recipes.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import tarfile

from PIL import Image
import yaml

VOC_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
             'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def extract(archive, raw):
    if not zipfile.is_zipfile(archive):
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if not (raw / member.name).resolve().is_relative_to(raw.resolve()) or member.issym() or member.islnk():
                    raise ValueError('Unsafe archive member')
            tar.extractall(raw, filter='data')
        return
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (raw / name).resolve().is_relative_to(raw.resolve()):
                raise ValueError('Unsafe archive path')
        if z.testzip() is not None:
            raise ValueError('Corrupt archive')
        z.extractall(raw)


def aquarium(raw):
    result = {}
    class_names = None
    for split, folder in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
        candidates = list(raw.rglob(folder + '/annotations_without_background.json'))
        if not candidates:
            candidates = list(raw.rglob(folder + '/_annotations.coco.json'))
        if len(candidates) != 1:
            raise ValueError((split, candidates))
        f = candidates[0]
        data = json.loads(f.read_text())
        # ODinW/Roboflow may contain an unannotated parent category.
        cats = [c for c in data['categories'] if c['name'].lower() not in ['creatures', 'background']]
        names = [c['name'] for c in sorted(cats, key=lambda c: c['id'])]
        if class_names is not None and names != class_names:
            raise ValueError('Split class names differ')
        class_names = names
        mapping = {c['id']: names.index(c['name']) for c in cats}
        anns = defaultdict(list)
        for a in data['annotations']:
            if a['category_id'] not in mapping:
                raise ValueError('Annotated parent/unknown category')
            if a.get('iscrowd', 0):
                raise ValueError('Crowd annotations require an explicit shared policy')
            anns[a['image_id']].append((mapping[a['category_id']], a['bbox'], False))
        result[split] = [(f.parent / im['file_name'], im['width'], im['height'], anns[im['id']])
                         for im in sorted(data['images'], key=lambda i: i['id'])]
    return class_names, result


def voc(raw):
    roots = list(raw.rglob('VOC2007/ImageSets/Main/train.txt'))
    if len(roots) != 1:
        raise ValueError(roots)
    root = roots[0].parents[2]
    result = {}
    for split in ['train', 'val', 'test']:
        rows = []
        for image_id in (root / 'ImageSets/Main' / (split + '.txt')).read_text().split():
            tree = ET.parse(root / 'Annotations' / (image_id + '.xml')).getroot()
            w = int(tree.findtext('size/width'))
            h = int(tree.findtext('size/height'))
            anns = []
            for obj in tree.findall('object'):
                box = [float(obj.findtext('bndbox/' + n)) for n in ['xmin', 'ymin', 'xmax', 'ymax']]
                # VOC coordinates are 1-based inclusive; COCO is 0-based xywh.
                x1, y1, x2, y2 = box
                anns.append((VOC_NAMES.index(obj.findtext('name')),
                             [x1 - 1, y1 - 1, x2 - x1 + 1, y2 - y1 + 1],
                             obj.findtext('difficult', '0') == '1'))
            rows.append((root / 'JPEGImages' / (image_id + '.jpg'), w, h, anns))
        result[split] = rows
    return VOC_NAMES, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['aquarium', 'voc2007'], required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--archives', nargs='+', required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    raw = root / 'raw'
    out = root / 'data640'
    if out.exists():
        raise FileExistsError(out)
    raw.mkdir(parents=True, exist_ok=True)
    for archive in a.archives:
        extract(Path(archive), raw)
    names, rows = (aquarium if a.dataset == 'aquarium' else voc)(raw)
    seen = {}
    duplicates = []
    invalid_boxes = []
    counts = {}
    iid = aid = 0
    for split in ['test', 'val', 'train']:
        images, anns, tasks = [], [], []
        per_class = [0] * len(names)
        difficult = 0
        for kind in ['images', 'labels']:
            (out / kind / split).mkdir(parents=True, exist_ok=True)
        for f, w, h, boxes in rows[split]:
            with Image.open(f) as img:
                rgb = img.convert('RGB')
                if rgb.size != (w, h):
                    raise ValueError(('Image dimensions disagree', f, rgb.size, w, h))
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            source = split + '/' + f.name
            if digest in seen:
                duplicates.append(dict(removed=source, kept=seen[digest]))
                continue
            seen[digest] = source
            iid += 1
            fn = f'{split}_{iid:06}.png'
            tasks.append((f, out / 'images' / split / fn))
            images.append(dict(id=iid, file_name=fn, width=640, height=640))
            lines = []
            for c, box, diff in boxes:
                if len(box) != 4 or not all(math.isfinite(v) for v in box):
                    raise ValueError(('Nonfinite box', f, box))
                x, y, bw, bh = box
                x1, y1 = max(0., x / w), max(0., y / h)
                x2, y2 = min(1., (x + bw) / w), min(1., (y + bh) / h)
                if x2 <= x1 or y2 <= y1:
                    invalid_boxes.append(dict(source=source, category=c, bbox=box,
                                              reason='Nonpositive area after clipping'))
                    continue
                bw, bh = x2 - x1, y2 - y1
                aid += 1
                per_class[c] += 1
                difficult += int(diff)
                anns.append(dict(id=aid, image_id=iid, category_id=c,
                                 bbox=[x1 * 640, y1 * 640, bw * 640, bh * 640],
                                 area=bw * bh * 640 * 640, iscrowd=0, voc_difficult=bool(diff)))
                lines.append(f'{c} {(x1+x2)/2:.10f} {(y1+y2)/2:.10f} {bw:.10f} {bh:.10f}')
            (out / 'labels' / split / (Path(fn).stem + '.txt')).write_text('\n'.join(lines) + '\n')
        def resize(task):
            src, dst = task
            with Image.open(src) as img:
                img.convert('RGB').resize((640, 640), Image.Resampling.BILINEAR).save(dst, compress_level=1)
        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(resize, tasks))
        (out / (split + '.json')).write_text(json.dumps(dict(
            images=images, annotations=anns, categories=[dict(id=i, name=n) for i, n in enumerate(names)],
            info={}, licenses=[])))
        counts[split] = dict(raw_images=len(rows[split]), images=len(images), boxes=len(anns),
                             per_class=per_class, difficult_boxes_included=difficult)
        print(split, counts[split], flush=True)
    (out / 'dataset.yaml').write_text(yaml.safe_dump(dict(path=str(out), train='images/train',
        val='images/val', test='images/test', names=dict(enumerate(names)))))
    manifest = dict(dataset=a.dataset, classes=names, counts=counts, duplicates_removed=duplicates,
        invalid_boxes_removed=invalid_boxes,
        archives={Path(f).name: dict(bytes=Path(f).stat().st_size, sha256=sha(f)) for f in a.archives},
        dedup='Exact RGB pixels; test > val > train; all original splits otherwise retained',
        preprocessing='PIL bilinear direct resize to 640 square, lossless PNG, shared boxes',
        voc_policy='All difficult boxes included as ordinary positives; not official VOC AP',
        seed=42, epochs=50, total_batch=16, selection='Best native validation AP only; common held-out test COCO AP')
    (root / 'dataset-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('PREPARED', root, flush=True)


if __name__ == '__main__':
    main()
