"""Prepare the fixed, audited transfer-learning dataset without split leakage from exact duplicates."""

from pathlib import Path
from PIL import Image
import json,hashlib,math,yaml
import argparse,urllib.request,zipfile
parser=argparse.ArgumentParser(description='Download, exact-deduplicate and prepare African Wildlife for both detectors.')
parser.add_argument('--root',required=True)
args=parser.parse_args()
p=Path(args.root).resolve();p.mkdir(parents=True,exist_ok=True)
archive=p/'african-wildlife.zip'
if not archive.exists():
 urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip',archive)
expected='6b97a768bd8c89a4ac6f1fd72549a5af805f68568d8846fd0fe68e30c0040061'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected,'Upstream archive changed; audit the new version before comparison'
raw=p/'raw';out=p/'data640'
if out.exists():raise FileExistsError('Use a fresh root; data640 already exists')
raw.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():
  assert (raw/name).resolve().is_relative_to(raw.resolve()),'Unsafe archive path'
 z.extractall(raw)
seen={};duplicate=[];counts={};aid=0;iid=0
names=['buffalo','elephant','rhino','zebra']
for split in ['test','val','train']:
 images=[]; anns=[]; nc=[0]*4; files=sorted(x for x in (raw/'images'/split).iterdir() if x.suffix.lower() in ['.jpg','.jpeg','.png'])
 for d in ['images','labels']:(out/d/split).mkdir(parents=True,exist_ok=True)
 for f in files:
  im=Image.open(f).convert('RGB'); digest=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
  if digest in seen:
   duplicate.append({'removed':split+'/'+f.name,'kept':seen[digest]});continue
  seen[digest]=split+'/'+f.name;iid+=1;fn=f'{split}_{iid:06}.png'
  im.resize((640,640),Image.Resampling.BILINEAR).save(out/'images'/split/fn)
  images.append(dict(id=iid,file_name=fn,width=640,height=640)); lines=[]
  label=raw/'labels'/split/(f.stem+'.txt');assert label.exists(),str(label)
  for line in label.read_text().splitlines():
   if not line.strip():continue
   v=list(map(float,line.split()));assert len(v)==5 and all(map(math.isfinite,v)),(f,line)
   c,x,y,w,h=v;c=int(c);assert 0<=c<4
   x1=max(0,x-w/2);y1=max(0,y-h/2);x2=min(1,x+w/2);y2=min(1,y+h/2);assert x2>x1 and y2>y1
   w=x2-x1;h=y2-y1;aid+=1;nc[c]+=1
   anns.append(dict(id=aid,image_id=iid,category_id=c,bbox=[x1*640,y1*640,w*640,h*640],area=w*h*640*640,iscrowd=0))
   lines.append(f'{c} {(x1+x2)/2:.10f} {(y1+y2)/2:.10f} {w:.10f} {h:.10f}')
  (out/'labels'/split/(Path(fn).stem+'.txt')).write_text('\n'.join(lines)+'\n')
 (out/(split+'.json')).write_text(json.dumps(dict(images=images,annotations=anns,categories=[dict(id=i,name=n) for i,n in enumerate(names)],info={},licenses=[])))
 counts[split]=dict(raw_images=len(files),images=len(images),boxes=len(anns),per_class=nc)
(out/'dataset.yaml').write_text(yaml.safe_dump(dict(path=str(out),train='images/train',val='images/val',test='images/test',names=dict(enumerate(names)))))
manifest=dict(dataset='African Wildlife',source='https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip',archive_sha256=hashlib.sha256((p/'african-wildlife.zip').read_bytes()).hexdigest(),counts=counts,duplicates_removed=duplicate,dedup='Exact RGB pixels; priority test > val > train',preprocessing='PIL bilinear direct resize to 640 square; PNG lossless output',classes=names,seed=42,epochs=50,total_batch=32,augmentation='Horizontal flip p=0.5 only',optimizer='AdamW lr=1e-4 weight_decay=1e-4',selection='Validation only; held-out test common COCO bbox AP')
(p/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
