"""Convert local YOLO detection YAML + normalized xywh text labels to COCO JSON."""
import argparse
import json
import math
import os
from pathlib import Path
import yaml
from PIL import Image

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def schema(names):
    if isinstance(names, dict):
        if any(isinstance(k, bool) or not isinstance(k,(int,str)) or not str(k).isdigit() for k in names):
            raise ValueError('YOLO class IDs must be contiguous integers from zero')
        try:
            mapped = {int(k): v for k, v in names.items()}
        except (ValueError, TypeError) as e:
            raise ValueError('Invalid YOLO class IDs') from e
        if len(mapped) != len(names) or sorted(mapped) != list(range(len(mapped))):
            raise ValueError('YOLO class IDs must be contiguous integers from zero')
        names = [mapped[i] for i in range(len(mapped))]
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError('names must be a nonempty list/dict of unique class names')
    return [{'id': i, 'name': n} for i, n in enumerate(names)]


def label_path(image):
    parts = list(image.parts)
    positions = [i for i, part in enumerate(parts) if part == 'images']
    if not positions:
        raise ValueError('YOLO layout requires an images/ component with matching labels/: '+str(image))
    parts[positions[-1]] = 'labels'
    return Path(*parts).with_suffix('.txt')


def image_list(spec, root):
    values = spec if isinstance(spec, list) else [spec]
    images = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError('train/val must be image directories, text lists, or lists of these')
        path = Path(value).expanduser()
        path = path if path.is_absolute() else root/path
        path = path.resolve()
        if path.is_dir():
            images.extend(p.resolve() for p in path.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
        elif path.is_file() and path.suffix.lower() == '.txt':
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                p = Path(line).expanduser()
                if not p.is_absolute():
                    p = (path.parent/p) if line.startswith('./') else (root/p)
                images.append(p.resolve())
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise FileNotFoundError(path)
    if not images or len(set(images)) != len(images):
        raise ValueError('Empty split or duplicate image paths')
    for p in images:
        if not p.is_file():
            raise FileNotFoundError(p)
    return sorted(images)


def split_to_coco(images, cats, strict_labels=False):
    root = Path(os.path.commonpath([str(p.parent) for p in images]))
    records, annotations, missing = [], [], 0
    for i, image in enumerate(images):
        with Image.open(image) as im:
            width, height = im.size
        records.append({'id': i, 'file_name': image.relative_to(root).as_posix(), 'width': width, 'height': height})
        label = label_path(image)
        if not label.exists():
            missing += 1
            if strict_labels:
                raise FileNotFoundError(label)
            continue
        for number, line in enumerate(label.read_text().splitlines(), 1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f'{label}:{number}: detection needs class cx cy w h; segmentation/pose/OBB unsupported')
            try:
                cls, cx, cy, w, h = map(float, values)
            except ValueError as e:
                raise ValueError(f'{label}:{number}: nonnumeric label') from e
            if not all(math.isfinite(v) for v in [cls, cx, cy, w, h]) or not cls.is_integer() or not 0 <= cls < len(cats):
                raise ValueError(f'{label}:{number}: invalid/nonfinite class or coordinates')
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f'{label}:{number}: xywh must be normalized, with positive size')
            x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
            if min(x1,y1) < -1e-6 or max(x2,y2) > 1+1e-6:
                raise ValueError(f'{label}:{number}: box extends outside image')
            x1, y1, x2, y2 = max(0.,x1)*width, max(0.,y1)*height, min(1.,x2)*width, min(1.,y2)*height
            annotations.append({'id': len(annotations)+1, 'image_id': i, 'category_id': int(cls),
                                'bbox': [x1,y1,x2-x1,y2-y1], 'area': (x2-x1)*(y2-y1), 'iscrowd': 0})
    return {'info': {'annotation_source': 'YOLO detection text'}, 'images': records, 'annotations': annotations, 'categories': cats}, root, missing


def convert(data_yaml, output, strict_labels=False):
    source = Path(data_yaml).resolve()
    d = yaml.safe_load(source.read_text())
    cats = schema(d.get('names'))
    if 'nc' in d and d['nc'] != len(cats):
        raise ValueError('nc differs from names count')
    root = Path(d.get('path') or '.').expanduser()
    root = (root if root.is_absolute() else source.parent/root).resolve()
    splits = {n: image_list(d[n], root) for n in ['train','val']}
    if set(splits['train']) & set(splits['val']):
        raise ValueError('Training and validation images overlap')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    converted = {n: split_to_coco(splits[n],cats,strict_labels) for n in splits}
    output.mkdir(parents=True)
    result = {'source_format': 'yolo-detection', 'num_classes': len(cats), 'missing_labels_as_background': {}}
    for name,(data,images,missing) in converted.items():
        dest = output/(name+'.json')
        dest.write_text(json.dumps(data))
        result[name+'_json'] = str(dest)
        result[name+'_images'] = str(images)
        result['missing_labels_as_background'][name] = missing
    (output/'dataset.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--strict-labels',action='store_true')
    a=p.parse_args()
    print(json.dumps(convert(a.data,a.output,a.strict_labels),indent=2))


if __name__ == '__main__':
    main()
