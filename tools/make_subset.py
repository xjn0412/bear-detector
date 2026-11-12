"""Fixed random held-out subset, independent of model scores; preserve every class."""
import argparse,random,json
from pathlib import Path
from common import digest
p=argparse.ArgumentParser();p.add_argument('--annotations',required=True);p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=1000);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
d=json.loads(Path(a.annotations).read_text());images=sorted(d['images'],key=lambda x:x['id'])
if not 1<=a.count<=len(images):raise ValueError('Invalid subset size')
rng=random.Random(a.seed);rng.shuffle(images)
by_image={x['id']:set() for x in images}
for ann in d['annotations']:
 if not ann.get('iscrowd',0):
  area=ann.get('area',ann['bbox'][2]*ann['bbox'][3]);scale=0 if area<32**2 else (1 if area<96**2 else 2)
  by_image[ann['image_id']].add((ann['category_id'],scale))
uncovered=set().union(*by_image.values());chosen=[];remaining=list(images)
while uncovered:
 im=max(remaining,key=lambda x:len(by_image[x['id']]&uncovered))
 chosen.append(im);remaining.remove(im);uncovered-=by_image[im['id']]
if len(chosen)>a.count:raise ValueError('Subset too small for class/size coverage')
chosen+=rng.sample(remaining,a.count-len(chosen));ids={x['id'] for x in chosen}
sub=dict(d,images=sorted(chosen,key=lambda x:x['id']),annotations=[x for x in d['annotations'] if x['image_id'] in ids])
seen={x['category_id'] for x in sub['annotations'] if not x.get('iscrowd',0)}
if seen!={x['id'] for x in d['categories']}:raise ValueError('Subset misses categories; use a larger subset')
path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(sub))
path.with_suffix('.manifest.json').write_text(json.dumps({'seed':a.seed,'count':a.count,'source_sha256':digest(a.annotations),'subset_sha256':digest(path),'image_ids':sorted(ids),'selection':'greedy class-size coverage plus seeded random fill; no score-based selection'},indent=2))
print(path)
