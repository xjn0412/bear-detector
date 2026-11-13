"""Shared GPU deployment loader; each benchmark process owns one model/context."""
import json
from pathlib import Path
from common import digest, load_model, Inference


def load_backend(backend, config, checkpoint, engine_path=None):
    import torch
    if backend != 'tensorrt':
        cfg, core = load_model(config, checkpoint, deploy=True)
        model = Inference.create(core, cfg.postprocessor, backend == 'torch-amp').cuda().eval()
        return model, int(cfg.yaml_cfg['num_classes']), {'precision': 'FP16 bulk + FP32 decoder' if backend == 'torch-amp' else 'FP32'}
    import tensorrt as trt
    path = Path(engine_path)
    meta = json.loads((path.parent / 'export.json').read_text())
    if meta['checkpoint_sha256'] != digest(checkpoint) or meta['engine_sha256'] != digest(path):
        raise ValueError('Engine/checkpoint provenance mismatch')
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError('TensorRT deserialization failed')
    context = engine.create_execution_context()
    dtypes = {trt.float32: torch.float32, trt.float16: torch.float16,
              trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool}
    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        buffers[name] = torch.empty(tuple(engine.get_tensor_shape(name)), device='cuda',
                                   dtype=dtypes[engine.get_tensor_dtype(name)])
        context.set_tensor_address(name, buffers[name].data_ptr())

    def forward(images, sizes):
        buffers['images'].copy_(images)
        buffers['orig_target_sizes'].copy_(sizes)
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT execute failed')
        return tuple(buffers[n] for n in ['labels', 'boxes', 'scores'])
    # Retain all TensorRT owners in the closure for the lifetime of forward.
    forward.owners = (logger, runtime, engine, context, buffers)
    return forward, int(meta['classes']), {'tensorrt': trt.__version__, 'precision': meta['precision'],
        'engine_sha256': meta['engine_sha256']}


def image_tensor(path):
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert('RGB')
        w, h = im.size
        x = torch.from_numpy(np.array(im.resize((640, 640), Image.Resampling.BILINEAR), copy=True))
    x = x.permute(2, 0, 1).float() / 255
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    return (x - mean) / std, torch.tensor([w, h], dtype=torch.int64)
