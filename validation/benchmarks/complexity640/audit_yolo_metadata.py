"""Research metadata audit for fixed official YOLO26 v8.4.0 release assets.

Uses inert module/tensor stubs, not a general or inference checkpoint loader.
Sizes/digests are upstream API values; full archive SHA is not verified here.
"""
from pathlib import Path
import pickle,io,struct,json,math,urllib.request,concurrent.futures
import argparse
parser=argparse.ArgumentParser(description='Audit public YOLO checkpoint tensor metadata without executing a model')
parser.add_argument('--cache-dir',default='results/yolo-metadata')
parser.add_argument('--output',default='results/yolo-storage.json')
args=parser.parse_args()
r=Path(args.cache_dir);r.mkdir(parents=True,exist_ok=True)
class Stub:pass
class Tensor:
 def __init__(self,storage,offset,shape,stride,*args,**kwargs):self.storage=storage;self.shape=shape
class Storage:
 def __init__(self,dtype,key,numel):self.dtype=dtype;self.key=key;self.numel=numel
def tensor(*args,**kwargs):return Tensor(*args,**kwargs)
def parameter(t,*args,**kwargs):return t
class Reader(pickle.Unpickler):
 def find_class(self,module,name):
  if module.startswith('ultralytics.') or module.startswith('torch.nn.'):return Stub
  if module=='torch._utils':
   if name.startswith('_rebuild_tensor'):return tensor
   if name.startswith('_rebuild_parameter'):return parameter
  if module=='torch' and name.endswith('Storage'):return name
  return super().find_class(module,name)
 def persistent_load(self,pid):
  _,dtype,key,loc,numel=pid
  return Storage(dtype,key,numel)
def read(a):
 p=r/a['name']
 if p.exists() and p.stat().st_size>524288:data=p.open('rb').read(524288)
 else:
  req=urllib.request.Request(a['url'],headers={'Range':'bytes=0-524287'})
  with urllib.request.urlopen(req,timeout=90) as f:
   data=f.read(524288)
  (r/(a['name']+'.metadata-prefix')).write_bytes(data)
 assert data[:4]==b'PK\x03\x04'
 name_len,extra_len=struct.unpack_from('<HH',data,26)
 filename=data[30:30+name_len].decode()
 assert filename.endswith('/data.pkl'),filename
 ck=Reader(io.BytesIO(data[30+name_len+extra_len:])).load()
 model=ck['ema'] if isinstance(ck.get('ema'),Stub) else ck['model']
 seen=set();ps=[];bs=[]
 def walk(m):
  if id(m) in seen:return
  seen.add(id(m))
  ps.extend(x for x in m.__dict__.get('_parameters',{}).values() if isinstance(x,Tensor))
  bs.extend(x for x in m.__dict__.get('_buffers',{}).values() if isinstance(x,Tensor))
  for child in m.__dict__.get('_modules',{}).values():
   if child is not None:walk(child)
 walk(model)
 def stats(items):
  unique={id(t):t for t in items};d={}
  for t in unique.values():d[t.storage.dtype]=d.get(t.storage.dtype,0)+math.prod(t.shape)
  return d
 result={**a,'full_checkpoint_parameters':sum(stats(ps).values()),'parameter_elements_by_dtype':stats(ps),'buffer_elements_by_dtype':stats(bs),'checkpoint_keys':list(ck),'optimizer_present':ck.get('optimizer') is not None,'method':'Parsed zip data.pkl from metadata prefix with inert module/tensor stubs; no forward or tensor values evaluated','full_archive_sha_verified':False}
 print(json.dumps(result),flush=True);return result
assets=[{"name":"yolo26l.pt","size":53211173,"upstream_sha256":"9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26","url":"https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt"},{"name":"yolo26m.pt","size":44255705,"sha256":"401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7","url":"https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt"},{"name":"yolo26n.pt","size":5544453,"sha256":"9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef","url":"https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"},{"name":"yolo26s.pt","size":20422725,"sha256":"646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b","url":"https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt"},{"name":"yolo26x.pt","size":118667365,"sha256":"9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92","url":"https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt"}]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(read,assets))
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'release':'ultralytics/assets v8.4.0','models':results},indent=2)+'\n')
