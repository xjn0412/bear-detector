"""Export same-checkpoint deployment graph; FP16 engine keeps entire decoder FP32."""
import argparse,json,time
from pathlib import Path
from common import load_model,Inference,digest
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--build',action='store_true');p.add_argument('--build-only',action='store_true');p.add_argument('--precision',choices=['mixed','fp32-backbone','fp32'],default='mixed');a=p.parse_args()
import torch,onnx
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
torch.backends.mha.set_fastpath_enabled(False)
if not a.build_only:
 cfg,core=load_model(a.config,a.checkpoint,deploy=True)
 m=Inference.create(core,cfg.postprocessor,amp=False)
 x=torch.zeros(1,3,640,640);sz=torch.tensor([[640,640]],dtype=torch.int64)
 from portable_sampling import grid_sample as portable_grid_sample
 native_grid_sample=torch.nn.functional.grid_sample
 torch.nn.functional.grid_sample=portable_grid_sample
 with torch.inference_mode():
  torch.onnx.export(m,(x,sz),str(out/'model.onnx'),opset_version=17,dynamo=False,do_constant_folding=True,input_names=['images','orig_target_sizes'],output_names=['labels','boxes','scores'])
 torch.nn.functional.grid_sample=native_grid_sample
 onnx.checker.check_model(str(out/'model.onnx'))
 report={'checkpoint_sha256':digest(a.checkpoint),'onnx_sha256':digest(out/'model.onnx'),'classes':int(cfg.yaml_cfg['num_classes']),'shape':[1,3,640,640],'size_order':'width,height','precision':'FP16 backbone/context/neck; entire decoder FP32; sensitive reductions FP32','accuracy_validated':False,'sampling':'explicit bilinear GatherElements; no ONNX GridSample'}
else:
 report=json.loads((out/'export.json').read_text())
 if report['checkpoint_sha256']!=digest(a.checkpoint) or report['onnx_sha256']!=digest(out/'model.onnx'):raise ValueError('Export provenance mismatch')
classes_file=Path(a.config).parent/'classes.json'
if classes_file.exists():
 report['class_mapping']=json.loads(classes_file.read_text())
 (out/'classes.json').write_text(classes_file.read_text())
if a.build:
 import tensorrt as trt
 if int(trt.__version__.split('.')[0]) != 10:
  raise RuntimeError('This precision-constrained exporter is validated with TensorRT10.13.3.9; TensorRT11 requires a strong-typing export path. Install tensorrt-cu13==10.13.3.9.')
 logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
 network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));parser=trt.OnnxParser(network,logger)
 if not parser.parse_from_file(str(out/'model.onnx')):raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
 config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,2<<30)
 if a.precision!='fp32':config.set_flag(trt.BuilderFlag.FP16)
 config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS);config.clear_flag(trt.BuilderFlag.TF32);config.builder_optimization_level=3
 protected=[];decoder=[]
 sensitive=[trt.LayerType.SOFTMAX,trt.LayerType.REDUCE]
 if hasattr(trt.LayerType,'NORMALIZATION'):sensitive.append(trt.LayerType.NORMALIZATION)
 for i in range(network.num_layers):
  layer=network.get_layer(i)
  dec='decoder' in layer.name
  if dec:decoder.append(layer.name)
  if dec or layer.type in sensitive or (a.precision=='fp32-backbone' and 'backbone' in layer.name):
   # Shape/int tensors have no floating arithmetic precision to constrain.
   if layer.type not in [trt.LayerType.CONSTANT,trt.LayerType.SHAPE] and any(layer.get_output(j).dtype in [trt.float32,trt.float16] for j in range(layer.num_outputs)):
    layer.precision=trt.float32
    for j in range(layer.num_outputs):
     if layer.get_output(j).dtype in [trt.float32,trt.float16]:layer.set_output_type(j,trt.float32)
    protected.append(layer.name)
 if len(decoder)<50:raise RuntimeError('Decoder scope missing from ONNX; refusing unsafe FP16 build')
 t=time.time();plan=builder.build_serialized_network(network,config)
 if plan is None:raise RuntimeError('Engine build failed')
 (out/'model.engine').write_bytes(bytes(plan))
 report['precision']={'fp32':'FP32','fp32-backbone':'FP32 backbone/context and decoder; FP16 neck','mixed':'FP16 backbone/context/neck; FP32 decoder and sensitive reductions'}[a.precision]
 report.update(engine_sha256=digest(out/'model.engine'),build_seconds=time.time()-t,tensorrt=trt.__version__,protected_layers=len(protected),decoder_layers=len(decoder))
(out/'export.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
