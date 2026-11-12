"""Runtime optimizations; no source checkout modifications or loss changes."""
from pathlib import Path
import os
import inspect
import torch
from torch import nn
from engine.misc import dist_utils
from engine.core.yaml_config import YAMLConfig

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.allow_tf32=True


def bind_cpu():
    # Optional topology-specific tuning; portable fine-tuning defaults to OS scheduling.
    if os.getenv('CPU_AFFINITY','0')!='1':return
    visible=os.getenv('CUDA_VISIBLE_DEVICES','0')
    if not all(x.isdigit() for x in visible.split(',')):return
    physical=[int(x) for x in os.getenv('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7').split(',')]
    rank=int(os.getenv('LOCAL_RANK','0'))
    if rank>=len(physical) or not hasattr(os,'sched_setaffinity'):return
    nodes=[0,0,0,0,1,1,1,1]
    gpu=physical[rank];node=nodes[gpu]
    node_file=Path(f'/sys/devices/system/node/node{node}/cpulist')
    if not node_file.is_file():return
    spec=node_file.read_text().strip()
    base=[]
    for part in spec.split(','):
        if '-' in part:
            lo,hi=map(int,part.split('-'));base.extend(range(lo,hi+1))
        else:base.append(int(part))
    peers=[g for g in physical if nodes[g]==node]
    slot=peers.index(gpu);cpus=base[slot::len(peers)]
    allowed=set(os.sched_getaffinity(0));cpus=sorted(set(cpus)&allowed)
    if cpus:os.sched_setaffinity(0,cpus)

bind_cpu()


def fp32_tree(value):
    if isinstance(value,torch.Tensor):return value.float() if value.is_floating_point() else value
    if isinstance(value,dict):return {k:fp32_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [fp32_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(fp32_tree(v) for v in value)
    return value


# All possible loss keys for the retained six-layer DEIMv2 decoder. Fixed keys
# avoid rank-dependent message sizes and per-step Python-object collectives.
BASE_LOSSES=('loss_mal','loss_bbox','loss_giou','loss_fgl','loss_ddf')
SUFFIXES=('', '_pre','_enc_0','_dn_pre')+tuple(f'_aux_{i}' for i in range(5))+tuple(f'_dn_{i}' for i in range(6))
LOSS_KEYS=tuple(sorted(k+s for k in BASE_LOSSES for s in SUFFIXES))

def reduce_losses(data,avg=True):
    unknown=set(data)-set(LOSS_KEYS)
    if unknown:raise RuntimeError(f'Unexpected loss keys: {unknown}')
    world=dist_utils.get_world_size()
    with torch.no_grad():
        ref=next(iter(data.values()))
        values=torch.stack([data[k].detach().float() if k in data else
            ref.new_zeros((),dtype=torch.float32) for k in LOSS_KEYS])
        if world>1:
            torch.distributed.all_reduce(values)
            if avg:values/=world
        return dict(zip(LOSS_KEYS,values))

dist_utils.reduce_dict=reduce_losses


class QueryAttention(nn.MultiheadAttention):
    """Decoder callers discard attention weights; use the optimized SDPA path."""
    def forward(self,query,key,value,key_padding_mask=None,need_weights=False,
                attn_mask=None,average_attn_weights=True,is_causal=False):
        return super().forward(query,key,value,key_padding_mask=key_padding_mask,
            need_weights=False,attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,is_causal=is_causal)


def optimize_query_attention(model):
    if os.getenv('FUSED_QUERY_ATTENTION','1')=='1':
        for name,module in model.named_modules():
            if name.startswith('decoder.') and isinstance(module,nn.MultiheadAttention):
                module.__class__=QueryAttention
    return model


def wrap_model(model,sync_bn=False,dist_mode='ddp',find_unused_parameters=True,**kwargs):
    model=optimize_query_attention(model)
    if os.getenv('CHANNELS_LAST','1')=='1':model=model.to(memory_format=torch.channels_last)
    if sync_bn:raise ValueError('The optimized configuration uses local normalization')
    if dist_utils.is_dist_available_and_initialized():
        rank=dist_utils.get_rank()
        model=nn.parallel.DistributedDataParallel(model,device_ids=[rank],output_device=rank,
            find_unused_parameters=find_unused_parameters,gradient_as_bucket_view=True,
            broadcast_buffers=False,bucket_cap_mb=25)
    return model

dist_utils.warp_model=wrap_model


def wrap_loader(loader,shuffle=False):
    if not dist_utils.is_dist_available_and_initialized():return loader
    sampler=torch.utils.data.DistributedSampler(loader.dataset,shuffle=shuffle)
    wrapped=torch.utils.data.DataLoader(loader.dataset,batch_size=loader.batch_size,
        sampler=sampler,drop_last=loader.drop_last,collate_fn=loader.collate_fn,
        pin_memory=True,num_workers=loader.num_workers,
        # Epoch-dependent dataset AND collate state must be refreshed in workers.
        persistent_workers=False)
    # Upstream loop calls set_epoch on the custom loader.
    def set_epoch(epoch):
        wrapped.dataset.set_epoch(epoch);wrapped.collate_fn.set_epoch(epoch)
    wrapped.set_epoch=set_epoch
    return wrapped

dist_utils.warp_loader=wrap_loader


def optimizer_groups(cfg,model):
    grouped={};seen=set()
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        assert id(p) not in seen;seen.add(id(p))
        lr=cfg['lr']
        if name.startswith('backbone.body.'):
            # Stage-wise decay protects the pretrained early convolutional stages.
            depth=4
            if '.stages.' in name:depth=3-int(name.split('.stages.')[1].split('.')[0])
            lr=cfg['lr']*.05*(.8**depth)
        decay=0. if p.ndim<=1 or any(t in name for t in ('cls_token','reg_token')) else cfg['weight_decay']
        grouped.setdefault((lr,decay),[]).append(p)
    return [dict(params=params,lr=lr,weight_decay=decay) for (lr,decay),params in grouped.items()]

YAMLConfig.get_optim_params=staticmethod(optimizer_groups)


def finite_flag(value,label):
 flag=(~torch.isfinite(value).all()).to(dtype=torch.int32)
 if dist_utils.is_dist_available_and_initialized():
  torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MAX)
 if flag.item():raise FloatingPointError('Nonfinite '+label+'; update blocked')

def finite_outputs(value):
 result=fp32_tree(value)
 for k in ['pred_boxes','pred_logits']:finite_flag(result[k],k)
 return result

def scaler_guard(scaler):
 scale=scaler.get_scale()
 if not __import__('math').isfinite(scale) or scale<1e-8:
  raise FloatingPointError('Dynamic loss scale collapsed; diagnose without retry')
 if dist_utils.is_dist_available_and_initialized():
  lo=torch.tensor(scale,device='cuda',dtype=torch.float64);hi=lo.clone()
  torch.distributed.all_reduce(lo,op=torch.distributed.ReduceOp.MIN)
  torch.distributed.all_reduce(hi,op=torch.distributed.ReduceOp.MAX)
  if lo.item()!=hi.item():raise FloatingPointError('GradScaler scales differ across ranks')

def check_raw_losses(losses):
 values=torch.stack([v.detach().float().reshape(()) for v in losses.values()])
 flag=(~torch.isfinite(values).all()).to(torch.int32)
 if dist_utils.is_dist_available_and_initialized():
  torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MAX)
 if flag.item():
  bad={k:float(v.detach()) for k,v in losses.items() if not torch.isfinite(v.detach()).all()}
  raise FloatingPointError('Nonfinite raw criterion components, never masked: '+str(bad))

def install_stable_swiglu():
 from engine.deim.deim_utils import SwiGLUFFN
 from engine.deim.deim_criterion import DEIMCriterion
 if not getattr(SwiGLUFFN,'_fp16_stable_gate',False):
  def stable_forward(self,x):
   x12=self.w12(x)
   with torch.autocast(device_type=x.device.type,enabled=False):
    x1,x2=x12.float().chunk(2,dim=-1)
    return self.w3(torch.nn.functional.silu(x1)*x2)
  SwiGLUFFN.forward=stable_forward
  SwiGLUFFN._fp16_stable_gate=True
 if not getattr(DEIMCriterion,'_raw_loss_guard',False):
  import textwrap
  source=textwrap.dedent(inspect.getsource(DEIMCriterion.forward))
  before='losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}'
  assert before in source,'Pinned criterion source changed'
  source=source.replace(before,'check_raw_losses(losses)')
  namespace=dict(DEIMCriterion.forward.__globals__,check_raw_losses=check_raw_losses)
  exec(compile(source,'<criterion_unmasked_finite_guard>','exec'),namespace)
  DEIMCriterion.forward=namespace['forward']
  DEIMCriterion._raw_loss_guard=True

def install_training_loop():
 install_stable_swiglu()
 # Authorized decoder FP32 repair: FP16 backbone/neck remain unchanged.
 from engine.deim.deim_decoder import DEIMTransformer
 if not getattr(DEIMTransformer,'_full_fp32_decoder',False):
  original_decoder_forward=DEIMTransformer.forward
  def decoder_fp32_forward(self,*args,**kwargs):
   with torch.autocast(device_type='cuda',enabled=False):
    return original_decoder_forward(self,*fp32_tree(args),**fp32_tree(kwargs))
  DEIMTransformer.forward=decoder_fp32_forward
  DEIMTransformer._full_fp32_decoder=True
 import engine.solver.det_engine as engine
 import engine.solver.det_solver as solver
 source=inspect.getsource(engine.train_one_epoch)
 replacements={
  'with torch.autocast(device_type=str(device), cache_enabled=True):':"with torch.autocast(device_type='cuda', dtype=torch.float16, cache_enabled=True):",
  'outputs = model(samples, targets=targets)':'outputs = finite_outputs(model(samples, targets=targets))',
  'samples = samples.to(device)':"samples = samples.to(device, non_blocking=True, memory_format=torch.channels_last if os.getenv('CHANNELS_LAST','1')=='1' else torch.contiguous_format)",
  'v.to(device)':'v.to(device, non_blocking=True)',
  'optimizer.zero_grad()':'optimizer.zero_grad(set_to_none=True)',
  'scaler.scale(loss).backward()':'scaler_guard(scaler)\n            scaler.scale(loss).backward()',
  'loss = sum(loss_dict.values())':"loss = sum(loss_dict.values())\n            finite_flag(loss, 'loss')",
  'scaler.step(optimizer)\n            scaler.update()':"scale_before = scaler.get_scale()\n            observed_before = getattr(optimizer, '_observed_steps', None)\n            scaler.step(optimizer)\n            scaler.update()\n            optimizer_did_step = (optimizer._observed_steps > observed_before) if observed_before is not None else (scaler.get_scale() >= scale_before and scaler.get_scale() > 0)\n            if not optimizer_did_step and dist_utils.is_main_process():\n                print('AMP_SKIPPED_UPDATE', global_step, scaler.get_scale(), flush=True)",
  'if ema is not None:':'if ema is not None and (scaler is None or optimizer_did_step):'
 }
 # GradScaler must handle overflow by skipping the update and lowering its scale.
 # Gradient clipping follows unscale_; do not raise for recoverable scaled overflow.
 for before,after in replacements.items():
  if before not in source:raise RuntimeError('Pinned source changed: '+before)
  source=source.replace(before,after)
 engine.__dict__.update(fp32_tree=fp32_tree,finite_outputs=finite_outputs,finite_flag=finite_flag,scaler_guard=scaler_guard,os=os)
 exec(compile(source,'<fp16_fp32criterion_training>','exec'),engine.__dict__)
 solver.train_one_epoch=engine.train_one_epoch
 original_evaluate=solver.evaluate
 def evaluate_consistent(model,*args,**kwargs):
  if dist_utils.is_dist_available_and_initialized():
   for buffer in model.buffers():torch.distributed.broadcast(buffer,src=0)
  return original_evaluate(model,*args,**kwargs)
 solver.evaluate=evaluate_consistent
 def stats(cfg,*args,**kwargs):
  count=sum(p.numel() for p in cfg.model.parameters())
  return count,{'parameters':count,'FLOPs':'not profiled'}
 solver.stats=stats
