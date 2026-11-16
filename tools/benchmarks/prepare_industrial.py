"""Prepare shared 640-square COCO/YOLO inputs; never split exact duplicates apart.

KSDD2 is a derived box task: one box per 8-connected mask component,
without an area threshold. Official test membership is preserved.
"""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy import ndimage
import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_records(root, dataset):
    records, names = [], []
    if dataset == 'cr7' or (dataset=='neu-det' and not list(root.rglob('*.xml'))):
        names = ['inclusion', 'dents', 'oil_spots', 'pits', 'punching', 'linear', 'macular'] if dataset=='cr7' else ['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']
        labels = {p.stem: p for p in root.rglob('*.txt') if 'label' in str(p.parent).lower()}
        for image in sorted(root.rglob('*.jpg')):
            if image.stem not in labels:
                raise ValueError('Missing annotation: ' + str(image))
            w, h = Image.open(image).size
            boxes = []
            for line in labels[image.stem].read_text().splitlines():
                if not line.strip():
                    continue
                c, x, y, bw, bh = map(float, line.split())
                assert c == int(c) and 0 <= c < len(names)
                boxes.append([int(c), (x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h])
            records.append(dict(image=image, boxes=boxes, source=str(image.relative_to(root))))
    elif dataset in ['gc10', 'neu-det']:
        images = defaultdict(list)
        for p in root.rglob('*'):
            if p.suffix.lower() in ['.jpg', '.png', '.bmp', '.jpeg']:
                images[p.name].append(p)
        annotations = sorted(root.rglob('*.xml'))
        excluded_bad=[]
        if dataset=='gc10':
            canonical={str(i)+'_'+n for i,n in enumerate(['chongkong','hanfeng','yueyawan','shuiban','youban','siban','yiwu','yahen','zhehen','yaozhe'],1)}
            accepted=[]
            for path in annotations:
                tree=ET.parse(path).getroot()
                raw_names={o.findtext('name').strip() for o in tree.findall('object')}
                if any(n not in canonical and n!='10_yaozhed' for n in raw_names):
                    excluded_bad.append(tree.findtext('filename'));continue
                accepted.append(path)
            annotations=accepted
        def canonical_name(name):
            return '10_yaozhe' if dataset=='gc10' and name=='10_yaozhed' else name
        all_names = {canonical_name(o.findtext('name').strip()) for p in annotations for o in ET.parse(p).getroot().findall('object')}
        if dataset=='gc10':assert len(all_names)==10,all_names
        names = sorted(all_names)
        ids = {name: i for i, name in enumerate(names)}
        used = set()
        for p in annotations:
            tree = ET.parse(p).getroot()
            filename = Path(tree.findtext('filename', p.stem + '.jpg').replace('\\', '/')).name
            candidates = images.get(filename, [])
            if len(candidates) > 1:
                prefix = 'train' if 'train' in p.parts else 'validation' if 'validation' in p.parts else None
                if prefix:
                    candidates = [x for x in candidates if prefix in x.parts]
            if dataset == 'gc10' and len(candidates)>1:
                pixels=[]
                for candidate in candidates:
                    with Image.open(candidate) as im:
                        rgb=im.convert('RGB')
                        pixels.append(hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest())
                if len(set(pixels))!=1:
                    raise ValueError(('Same filename has different pixels',p,candidates))
            if not candidates or (len(candidates)>1 and dataset!='gc10'):
                raise ValueError(('Ambiguous/missing image', p, candidates))
            image = candidates[0]
            if any(x in used for x in candidates):
                raise ValueError(('Duplicate annotation file', p, image))
            used.update(candidates)
            boxes = []
            for o in tree.findall('object'):
                bb = o.find('bndbox')
                # Pascal VOC: one-based inclusive to zero-based half-open.
                x1, y1, x2, y2 = [float(bb.findtext(k)) for k in ['xmin','ymin','xmax','ymax']]
                boxes.append([ids[canonical_name(o.findtext('name').strip())], x1-1, y1-1, x2, y2])
            for image in candidates:
                records.append(dict(image=image, boxes=[list(bb) for bb in boxes], source=str(image.relative_to(root))))
        if len(used) != sum(len(x) for x in images.values()):
            if dataset!='gc10':
                raise ValueError(('Unannotated images', sum(len(x) for x in images.values())-len(used)))
            print('EXCLUDE_UNANNOTATED', [str(p.relative_to(root)) for ps in images.values() for p in ps if p not in used])
    elif dataset == 'ksdd2':
        names = ['defect']
        for maskpath in sorted(root.rglob('*_GT.png')):
            image = maskpath.with_name(maskpath.name.replace('_GT.png', '.png'))
            if not image.is_file():
                raise FileNotFoundError(image)
            mask = np.array(Image.open(maskpath).convert('L')) > 0
            labeled, count = ndimage.label(mask, structure=np.ones((3, 3)))
            boxes = []
            for sl in ndimage.find_objects(labeled):
                if sl is not None:
                    ys, xs = sl
                    boxes.append([0, xs.start, ys.start, xs.stop, ys.stop])
            split = 'test' if 'test' in image.parts else 'train'
            records.append(dict(image=image, boxes=boxes, source=str(image.relative_to(root)), official_split=split))
    else:
        raise ValueError(dataset)
    assert records and names
    assert len(names)=={'cr7':7,'gc10':10,'neu-det':6,'ksdd2':1}[dataset], (dataset,names)
    return records, names


def prepare(base, dataset):
    raw, out = base/'raw'/dataset, base/'datasets'/dataset
    if (out/'dataset-manifest.json').exists():
        print('ALREADY_PREPARED', dataset)
        return
    records, names = load_records(raw, dataset)
    groups = defaultdict(list)
    clipped = 0
    for r in records:
        with Image.open(r['image']) as im:
            im = im.convert('RGB')
            w, h = im.size
            r['pixel_sha256'] = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        r['original_size'] = [w, h]
        for box in r['boxes']:
            c,x1,y1,x2,y2 = box
            bb = [c,max(0.,min(w,x1)),max(0.,min(h,y1)),max(0.,min(w,x2)),max(0.,min(h,y2))]
            assert all(np.isfinite(bb)) and bb[3]>bb[1] and bb[4]>bb[2], (r['source'],box)
            clipped += bb != box
            box[:] = bb
        groups[r['pixel_sha256']].append(r)
    # Merge annotations of exact pixel duplicates and keep a single image.
    unique = []
    duplicate_groups = []
    for pixel_hash, rs in sorted(groups.items()):
        r = rs[0]
        if len(rs)>1:
            splits = {x.get('official_split') for x in rs}
            if len(splits)>1:
                raise ValueError(('Official split contains exact duplicates', [x['source'] for x in rs]))
            duplicate_groups.append([x['source'] for x in rs])
            r['boxes'] = [list(bb) for bb in sorted({tuple(b) for x in rs for b in x['boxes']})]
        unique.append(r)
    strata = defaultdict(list)
    for r in unique:
        strata[(r.get('official_split','all'),tuple(sorted({b[0] for b in r['boxes']})))].append(r)
    assigned = {s:[] for s in ['train','val','test']}
    rng = random.Random(42)
    for (official, classes), rs in sorted(strata.items()):
        rng.shuffle(rs)
        n = len(rs)
        if official == 'test':
            assigned['test'].extend(rs)
        elif official == 'train':
            nv = max(1,round(n*.1)) if n>=3 else 0
            assigned['val'].extend(rs[:nv]);assigned['train'].extend(rs[nv:])
        else:
            nv = max(1,round(n*.1)) if n>=3 else 0
            nt = max(1,round(n*.1)) if n>=3 else 0
            assigned['val'].extend(rs[:nv]);assigned['test'].extend(rs[nv:nv+nt]);assigned['train'].extend(rs[nv+nt:])
    categories = [dict(id=i,name=n) for i,n in enumerate(names)]
    manifest = dict(dataset=dataset, seed=42, classes=names, input='RGB direct PIL bilinear 640x640 PNG; boxes scaled independently in x/y',
                    split='Official KSDD2 test; 10% stratified validation from official train' if dataset=='ksdd2' else 'Custom approximately 80/10/10 split stratified by class-presence signature, seed42; source split not benchmarked',
                    raw_images=len(records),unique_images=len(unique),duplicate_groups=duplicate_groups,clipped_boxes=clipped,
                    annotation_conversion='8-connected mask components, no area threshold' if dataset=='ksdd2' else 'YOLO normalized boxes' if dataset in ['cr7','neu-det'] else 'VOC one-based inclusive converted to zero-based half-open',
                    counts={},sources={})
    if dataset=='gc10':
        included={r['source'] for r in records}
        manifest['excluded_unannotated_images']=[str(p.relative_to(raw)) for p in raw.rglob('*.jpg') if str(p.relative_to(raw)) not in included]
    if dataset=='gc10':
        manifest['class_aliases']={'10_yaozhed':'10_yaozhe'}
        manifest['excluded_ambiguous_annotation_images']=['img_02_425616500_00770.jpg']
        manifest['revision']='Corrected10-class schema; initial12-class runs invalidated'
    cache={}
    old=base/'datasets'/(dataset+'-invalid-12class')
    if (old/'split-membership.json').exists():
        for m in json.loads((old/'split-membership.json').read_text()):cache[m['pixel_sha256']]=old/'images'/m['split']/m['file_name']
    out.mkdir(parents=True,exist_ok=True)
    membership=[]
    for split, rs in assigned.items():
        assert rs, (dataset,split)
        ims, anns = [], []
        (out/'images'/split).mkdir(parents=True,exist_ok=True)
        (out/'labels'/split).mkdir(parents=True,exist_ok=True)
        for i,r in enumerate(sorted(rs,key=lambda x:x['source'])):
            filename=f'{i:05d}_{r["image"].stem}.png'
            w,h=r['original_size']
            destination=out/'images'/split/filename
            if r['pixel_sha256'] in cache:
                import shutil
                shutil.copyfile(cache[r['pixel_sha256']],destination)
            else:
                with Image.open(r['image']) as im:
                    im.convert('RGB').resize((640,640),Image.Resampling.BILINEAR).save(destination)
            ims.append(dict(id=i,file_name=filename,width=640,height=640))
            labels=[]
            for c,x1,y1,x2,y2 in r['boxes']:
                x,y,bw,bh=x1/w*640,y1/h*640,(x2-x1)/w*640,(y2-y1)/h*640
                anns.append(dict(id=len(anns)+1,image_id=i,category_id=c,bbox=[x,y,bw,bh],area=bw*bh,iscrowd=0))
                labels.append(f'{c} {(x+bw/2)/640:.10f} {(y+bh/2)/640:.10f} {bw/640:.10f} {bh/640:.10f}')
            (out/'labels'/split/Path(filename).with_suffix('.txt')).write_text('\n'.join(labels)+'\n' if labels else '')
            membership.append(dict(split=split,source=r['source'],pixel_sha256=r['pixel_sha256'],file_name=filename,boxes=len(r['boxes'])))
        doc=dict(info=dict(description=dataset+' derived detection benchmark'),images=ims,annotations=anns,categories=categories)
        (out/(split+'.json')).write_text(json.dumps(doc,separators=(',',':')))
        manifest['counts'][split]=dict(images=len(ims),boxes=len(anns),positive_images=len({a['image_id'] for a in anns}),per_class=dict(Counter(names[a['category_id']] for a in anns)))
        manifest['sources'][split+'_annotations_sha256']=digest(out/(split+'.json'))
    for split in ['train','val','test']:
        seen={m['pixel_sha256'] for m in membership if m['split']==split}
        for other in ['train','val','test']:
            if other!=split:assert not seen & {m['pixel_sha256'] for m in membership if m['split']==other}
    assert all(v>0 for v in manifest['counts']['train']['per_class'].values())
    assert set(manifest['counts']['train']['per_class'])==set(names)
    (out/'dataset.yaml').write_text(yaml.safe_dump(dict(path=str(out),train='images/train',val='images/val',test='images/test',names=dict(enumerate(names))),sort_keys=False))
    (out/'split-membership.json').write_text(json.dumps(membership,indent=2))
    (out/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--dataset',required=True)
    a=p.parse_args();prepare(a.base,a.dataset)
