"""Identical real-image preprocessing, top-300 outputs, COCO AP and CUDA-event timing."""
import argparse,json,time,contextlib,statistics
from pathlib import Path
from common import load_model,Inference,digest,categories
p=argparse.ArgumentParser();p.add_argument('--backend',choices=['torch-fp32','torch-amp','tensorrt'],required=True)
p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--engine');p.add_argument('--annotations',required=True);p.add_argument('--images',required=True);p.add_argument('--output',required=True);p.add_argument('--warmup',type=int,default=50);p.add_argument('--iterations',type=int,default=200);a=p.parse_args()
if a.warmup<0 or a.iterations<1:p.error('--warmup must be >=0 and --iterations >=1')
import torch,numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.mha.set_fastpath_enabled(False)
ann=json.loads(Path(a.annotations).read_text());cats=categories(ann);catids=[c['id'] for c in cats]
if not ann['images']:raise ValueError('Empty validation split')
if a.backend=='tensorrt' and not a.engine:p.error('--engine is required for TensorRT')
if a.backend=='tensorrt':
 import tensorrt as trt
 meta=json.loads((Path(a.engine).parent/'export.json').read_text())
 if meta['checkpoint_sha256']!=digest(a.checkpoint):raise ValueError('Engine/checkpoint mismatch')
 if meta['engine_sha256']!=digest(a.engine):raise ValueError('Engine/metadata hash mismatch')
 logger=trt.Logger(trt.Logger.WARNING);runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(Path(a.engine).read_bytes());ctx=engine.create_execution_context();buffers={}
 dtypes={trt.float32:torch.float32,trt.float16:torch.float16,trt.int32:torch.int32,trt.int64:torch.int64,trt.bool:torch.bool}
 for i in range(engine.num_io_tensors):
  name=engine.get_tensor_name(i);buffers[name]=torch.empty(tuple(engine.get_tensor_shape(name)),dtype=dtypes[engine.get_tensor_dtype(name)],device='cuda');ctx.set_tensor_address(name,buffers[name].data_ptr())
 def forward(x,s):
  buffers['images'].copy_(x);buffers['orig_target_sizes'].copy_(s)
  if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):raise RuntimeError('TensorRT execution failed')
  return tuple(buffers[n] for n in ['labels','boxes','scores'])
 num_classes=meta['classes']
else:
 cfg,core=load_model(a.config,a.checkpoint,deploy=True);num_classes=cfg.yaml_cfg['num_classes'];model=Inference.create(core,cfg.postprocessor,a.backend=='torch-amp').cuda()
 forward=model
if num_classes!=len(catids):raise ValueError('Annotation categories do not match model; use complete class schema')
# Serialized annotations are canonical sorted-ID order. Custom runs use normalized JSON + classes.json for original IDs.
mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
def image(im):
 img=Image.open(Path(a.images)/im['file_name']).convert('RGB');w,h=img.size
 if (w,h)!=(im['width'],im['height']):raise ValueError('Image dimensions differ from annotations')
 arr=torch.from_numpy(np.array(img.resize((640,640),Image.Resampling.BILINEAR),copy=True)).permute(2,0,1).float()/255
 return ((arr-mean)/std).unsqueeze(0).cuda(),torch.tensor([[w,h]],dtype=torch.int64,device='cuda')
rows=[];times=[];wall=[];torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
 x,sz=image(ann['images'][0])
 for _ in range(a.warmup):forward(x,sz)
 torch.cuda.synchronize()
 for _ in range(a.iterations):
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
  start.record();out=forward(x,sz);end.record();end.synchronize();times.append(start.elapsed_time(end))
 for idx,im in enumerate(ann['images']):
  t=time.perf_counter();x,sz=image(im);labels,boxes,scores=forward(x,sz)
  labels=labels[0].cpu();boxes=boxes[0].float().cpu();scores=scores[0].float().cpu()
  if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():raise FloatingPointError('Nonfinite predictions')
  if labels.min()<0 or labels.max()>=num_classes:raise ValueError('Class index out of range')
  boxes[:,2:]-=boxes[:,:2]
  rows.extend({'image_id':im['id'],'category_id':catids[int(l)],'bbox':b,'score':float(s)} for l,b,s in zip(labels,boxes.tolist(),scores))
  wall.append((time.perf_counter()-t)*1000)
  if idx%100==0:print('EVAL',a.backend,idx,flush=True)
outpath=Path(a.output);outpath.parent.mkdir(parents=True,exist_ok=True);pred=outpath.with_suffix('.predictions.json');pred.write_text(json.dumps(rows))
gt=COCO(a.annotations);gt.dataset.setdefault('info',{});dt=gt.loadRes(rows);ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=[im['id'] for im in ann['images']];ev.evaluate();ev.accumulate();ev.summarize()
report={'backend':a.backend,'checkpoint_sha256':digest(a.checkpoint),'annotations_sha256':digest(a.annotations),'images':len(ann['images']),'input':[1,3,640,640],'class_count':num_classes,'AP':float(ev.stats[0]*100),'AP50':float(ev.stats[1]*100),'AP75':float(ev.stats[2]*100),'AP_small':float(ev.stats[3]*100),'AP_medium':float(ev.stats[4]*100),'AP_large':float(ev.stats[5]*100),'gpu_mean_ms':statistics.mean(times),'gpu_median_ms':statistics.median(times),'gpu_p95_ms':float(np.percentile(times,95)),'gpu_FPS':1000/statistics.mean(times),'image_to_cpu_output_mean_ms':statistics.mean(wall),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'timing_scope':'batch1 eager enqueue, GPU inputs through top300 postprocessing; TensorRT includes device-buffer copies; no CUDA graph; wall time includes image decode/resize/normalize/transfers/CPU outputs; excludes JSON/evaluator'}
outpath.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
