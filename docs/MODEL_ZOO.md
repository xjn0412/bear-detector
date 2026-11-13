# Model zoo and measured results

[English](MODEL_ZOO.md) | [简体中文](MODEL_ZOO.zh-CN.md) | [Documentation index](README.md)

Full COCO val2017, 5,000 images, 640×640, EMA weights, single-scale PyTorch validation:

| Model | Total parameters (80 classes) | Backbone parameters | Selected epoch | mAP50:95 |
|---|---:|---:|---:|---:|
| Base | 112.11M | 87.56M | 42 | **59.0008** |
| Small | 73.68M | 49.45M | 57 | **57.8987** |
| Tiny | 52.05M | 27.82M | 58 | **56.4597** |
| Large | 221.40M | 196.23M | 24 | **60.0728** |

Snapshot: 2026-11-12. The earlier Base and Small runs were stopped by the author at completed epochs56 and68. Tiny is training; fresh Base then Large are queued under the [10-epoch patience policy](TRAINING_POLICY.md). The Base result belongs to the earlier run and its detector checkpoint is lost. Tiny is an interim training-validation result, using FP16 bulk forward and an FP32 decoder/criterion. These are selected checkpoints from repaired runs, not completed, uninterrupted benchmark reproductions. Base resumed from epoch20 and Small from epoch30 after numerical repairs. Architecture stayed fixed; the repaired runtime keeps the entire decoder FP32. These results do **not** establish a controlled FP16-versus-BF16 comparison. Detector supervision used COCO train2017; the pretrained DINOv3 backbone used external LVD-1689M pretraining, so this is **not all-data COCO-only training** and is not zero-shot COCO detection.

Same **Base epoch41 EMA** checkpoint, fixed 1,000-image coverage-stratified COCO val subset, RTX 5090:

| Inference backend | Subset mAP50:95 | GPU mean latency | FPS |
|---|---:|---:|---:|
| PyTorch FP32 | 59.4238 | 19.4644 ms | 51.38 |
| PyTorch FP16 bulk + FP32 decoder | 59.4149 | 19.1562 ms | 52.20 |
| TensorRT FP16 bulk + FP32 decoder | 59.3778 | 4.0203 ms | 248.74 |

Batch1, 640, no CUDA Graph; includes top-300 postprocessing, excludes image preprocessing/host transfers. TensorRT includes device-buffer copies. **Subset AP is not full COCO AP.** Full-set TensorRT parity and Small/Tiny inference speed have not been validated. [Benchmark protocol and evidence](BENCHMARKS.md).

## Architecture choices

- ConvNeXt T/S/B/L pyramid at strides 4, 8, 16, 32; no ViT backbone in these models.
- Two six-head context blocks only at stride32: 20×20 = 400 tokens.
- Cheap stride4 detail branch feeds stride8 features; there is no stride4 decoder memory level.
- Three 256-channel neck outputs, six query-decoder layers, 300 queries.
- FP16 autocast for backbone/context/neck; **entire decoder, floating predictions, matching and losses in FP32**. FP32 master parameters/optimizer and dynamic GradScaler. Do not call `model.half()`.
- Fine-tuning resets every class head and denoising class embedding, including when the old and new class counts happen to match. Resume is a separate operation.
- TensorRT export uses explicit bilinear GatherElements sampling instead of native ONNX GridSample, which failed our TensorRT accuracy checks.

See [weight downloads](WEIGHTS.md), [architecture](ARCHITECTURE.md) and [recorded evidence](../validation/README.md).
