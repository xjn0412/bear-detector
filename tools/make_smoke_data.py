"""Create a tiny synthetic three-class COCO dataset; AP is NOT an accuracy benchmark."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'images').mkdir()
    (root / 'labels').mkdir()
    cats = [{'id': 3, 'name': 'red_box'}, {'id': 17, 'name': 'green_box'}, {'id': 91, 'name': 'blue_box'}]
    for split, count, offset in [('train', 4, 0), ('val', 2, 4)]:
        images, annotations = [], []
        for n in range(count):
            i = n + offset
            name = f'{i}.jpg'
            im = Image.new('RGB', (320, 240), (50, 50, 50))
            draw = ImageDraw.Draw(im)
            for k, color in enumerate(['red', 'green', 'blue']):
                x, y, w, h = 15 + k * 90 + i, 30 + i * 3, 55, 75
                draw.rectangle((x, y, x+w, y+h), fill=color)
                annotations.append({'id': i*3+k+1, 'image_id': i, 'category_id': cats[k]['id'],
                                    'bbox': [x, y, w, h], 'area': w*h, 'iscrowd': 0})
            im.save(root/'images'/name)
            (root/'labels'/f'{i}.txt').write_text('\n'.join(f'{k} {(15+k*90+i+55/2)/320} {(30+i*3+75/2)/240} {55/320} {75/240}' for k in range(3))+'\n')
            images.append({'id': i, 'file_name': name, 'width': 320, 'height': 240})
        (root/f'{split}.json').write_text(json.dumps({'info': {}, 'images': images, 'annotations': annotations, 'categories': cats}))
        (root/f'{split}.txt').write_text('\n'.join('./images/'+im['file_name'] for im in images)+'\n')
    import yaml
    (root/'data.yaml').write_text(yaml.safe_dump({'path':'.','train':'train.txt','val':'val.txt','names':[c['name'] for c in cats]}))
    print(root)


if __name__ == '__main__':
    main()
