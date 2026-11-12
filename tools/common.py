"""Shared class schema, checkpoint and inference contract."""
from pathlib import Path
import sys, json, hashlib, math
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'source'))

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def categories(data):
    if any(not isinstance(c.get('id'),int) or isinstance(c.get('id'),bool) or not isinstance(c.get('name'),str) or not c['name'].strip() for c in data['categories']):
        raise ValueError('Category IDs must be integers and names nonempty strings')
    cats=sorted(data['categories'],key=lambda c:c['id'])
    ids=[c['id'] for c in cats]; names=[c['name'] for c in cats]
    if not cats or len(set(ids))!=len(ids) or len(set(names))!=len(names):
        raise ValueError('Require a nonempty, unique category ID/name schema')
    return cats

def normalize(data, schema):
    """Keep all categories even when a validation split lacks some instances."""
    ids={c['id']:i for i,c in enumerate(schema)}
    expected={c['id']:c['name'] for c in schema}
    actual={c['id']:c['name'] for c in categories(data)}
    if any(expected.get(k)!=v for k,v in actual.items()):
        raise ValueError('Train/validation category meanings differ')
    out=dict(data);out['categories']=[dict(c,id=i) for i,c in enumerate(schema)]
    image_ids=[x['id'] for x in data['images']]
    if len(set(image_ids))!=len(image_ids):raise ValueError('Duplicate image IDs')
    seen=set(image_ids); anns=[]; ann_ids=set()
    for a in data['annotations']:
        if a['category_id'] not in ids:raise ValueError('Unknown annotation category')
        if a['image_id'] not in seen:raise ValueError('Annotation references missing image')
        if a['id'] in ann_ids:raise ValueError('Duplicate annotation ID')
        ann_ids.add(a['id'])
        if len(a['bbox'])!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in a['bbox']) or a['bbox'][2]<=0 or a['bbox'][3]<=0:raise ValueError('Invalid bbox: require finite coordinates and positive size')
        anns.append(dict(a,category_id=ids[a['category_id']],iscrowd=a.get('iscrowd',0),area=a.get('area',a['bbox'][2]*a['bbox'][3])))
    out['annotations']=anns
    return out

def is_class_parameter(k):
    return k.startswith(('decoder.enc_score_head.','decoder.dec_score_head.','decoder.denoising_class_embed.'))

def transfer(target, source):
    """Reset ALL classifier/DN embeddings, even if class counts happen to match."""
    adapted={};reset=[]
    for k,v in target.items():
        if is_class_parameter(k):adapted[k]=v;reset.append(k)
        else:
            if k not in source or source[k].shape!=v.shape:raise ValueError('Architecture mismatch: '+k)
            adapted[k]=source[k]
    extra=set(source)-set(target)
    if extra:raise ValueError('Unexpected checkpoint keys: '+str(sorted(extra)))
    return adapted,reset

def weights(checkpoint):
    import torch
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if isinstance(ck.get('ema'),dict) and 'module' in ck['ema']:return ck['ema']['module'],ck
    return ck['model'],ck

def config(path):
    import backbone,compat
    from engine.core import YAMLConfig
    return YAMLConfig(str(path))

def load_model(path, checkpoint, deploy=False):
    cfg=config(path); m=cfg.model
    state,ck=weights(checkpoint);m.load_state_dict(state,strict=True)
    del state,ck
    return cfg,(m.deploy() if deploy else m.eval())

class Inference: # factory avoids importing torch for schema-only tools
    @staticmethod
    def create(model,post,amp=False):
        import torch
        class Wrapper(torch.nn.Module):
            def __init__(self):super().__init__();self.core=model;self.post=post.deploy()
            def forward(self, images, orig_target_sizes):
                # Explicit FP32 decoder: same contract in PyTorch and exported ONNX.
                with torch.autocast(device_type=images.device.type,dtype=torch.float16,enabled=amp):
                    features=self.core.encoder(self.core.backbone(images))
                with torch.autocast(device_type=images.device.type,enabled=False):
                    outputs=self.core.decoder([f.float() for f in features])
                    return self.post(outputs,orig_target_sizes)
        return Wrapper().eval()
