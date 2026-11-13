# Model zoo and measured results

[English](MODEL_ZOO.md) | [简体中文](MODEL_ZOO.zh-CN.md) | [Documentation index](../README.md)

Full COCO val2017, 5,000 images, 640×640, EMA weights, single-scale PyTorch validation:

| Model | Input | AP50:95 (PT / TRT) | Params M (full / deploy) | GFLOPs | PyTorch AMP ms | TensorRT ms | Mixed MB |
|---|---:|---:|---:|---:|---:|---:|---|
| [Bear-T](../../configs/T.yml) | 640 | 56.4851 / 56.4449 | 52.05 / 51.10 | 132.18 | 12.84 | 2.75 | [115.08](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt.gz) |
| [Bear-S](../../configs/S.yml) | 640 | 57.8864 / 57.8563 | 73.68 / 72.73 | 201.21 | 16.44 | 3.34 | [155.18](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-mixed.pt.gz) |
| [Bear-B](../../configs/B.yml) | 640 | 59.0192 / 58.9243 | 112.11 / 111.16 | 310.75 | 15.69 | 3.82 | [226.34](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-mixed.pt.gz) |
| [Bear-L](../../configs/L.yml) | 640 | 60.0748 / 60.0236 | 221.40 / 220.45 | 622.32 | 15.20 | 4.79 | [428.79](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-mixed.pt.gz) |



Final snapshot: 2026-11-14. All selected T/S/B/L EMA checkpoints are published. Tiny, fresh Base and Large used the entire FP32 decoder from epoch1; Small was repaired and resumed after epoch30. Tiny/Base used patience10, Large ultimately used patience20; Small was manually stopped. These are not controlled backbone-only or FP16-versus-BF16 ablations. DINOv3 used external LVD-1689M backbone pretraining; detection supervision used COCO train2017.

Historical **Base epoch41 EMA** (not the final Base release) checkpoint, fixed 1,000-image coverage-stratified COCO val subset, RTX 5090:

| Inference backend | Subset mAP50:95 | GPU mean latency | FPS |
|---|---:|---:|---:|
| PyTorch FP32 | 59.4238 | 19.4644 ms | 51.38 |
| PyTorch FP16 bulk + FP32 decoder | 59.4149 | 19.1562 ms | 52.20 |
| TensorRT FP16 bulk + FP32 decoder | 59.3778 | 4.0203 ms | 248.74 |

Batch1, 640, no CUDA Graph; includes top-300 postprocessing, excludes image preprocessing/host transfers. TensorRT includes device-buffer copies. **Subset AP is not full COCO AP.** That historical subset did not validate final-model full-COCO parity or Tiny/Small speed; see the current deployment results below. [Benchmark protocol and evidence](../benchmarks/BENCHMARKS.md).

## Architecture choices

- ConvNeXt T/S/B/L pyramid at strides 4, 8, 16, 32; no ViT backbone in these models.
- Two six-head context blocks only at stride32: 20×20 = 400 tokens.
- Cheap stride4 detail branch feeds stride8 features; there is no stride4 decoder memory level.
- Three 256-channel neck outputs, six query-decoder layers, 300 queries.
- FP16 autocast for backbone/context/neck; **entire decoder, floating predictions, matching and losses in FP32**. FP32 master parameters/optimizer and dynamic GradScaler. Do not call `model.half()`.
- Fine-tuning resets every class head and denoising class embedding, including when the old and new class counts happen to match. Resume is a separate operation.
- TensorRT export uses explicit bilinear GatherElements sampling instead of native ONNX GridSample, which failed our TensorRT accuracy checks.

See [weight downloads](WEIGHTS.md), [architecture](ARCHITECTURE.md) and [recorded evidence](../../validation/README.md).

[Complexity and reproduction](../benchmarks/COMPLEXITY.md) · [Weight size versus parameters](WEIGHT_SIZE.md).

## Current release deployment results

The table above uses fresh mixed-checkpoint validation; original epoch AP histories are unchanged. [Current PyTorch/TRT latency and memory](../benchmarks/DEPLOYMENT5090.md) supersede historical Base41 timings for these final selected weights. Base41 subset results remain historical evidence.
