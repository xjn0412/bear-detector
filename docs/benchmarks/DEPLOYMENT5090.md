# RTX5090 deployment validation and latency

[English](DEPLOYMENT5090.md) | [简体中文](DEPLOYMENT5090.zh-CN.md)

Measured2026-11-14 on final selected T/S/B/L EMA checkpoints, with no additional training. Mixed storage is recommended; original FP32 archives remain available.

| Model | Input | AP50:95 (PT / TRT) | Params M (full / deploy) | GFLOPs | PyTorch AMP ms | TensorRT ms | Mixed .pt MB |
|---|---:|---:|---:|---:|---:|---:|---|
| [Bear-T](../../configs/T.yml) | 640 | 56.4851 / 56.4449 | 52.05 / 51.10 | 132.18 | 12.84 | 2.75 | [124.43](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt) |
| [Bear-S](../../configs/S.yml) | 640 | 57.8864 / 57.8563 | 73.68 / 72.73 | 201.21 | 16.44 | 3.34 | [167.75](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-mixed.pt) |
| [Bear-B](../../configs/B.yml) | 640 | 59.0192 / 58.9243 | 112.11 / 111.16 | 310.75 | 15.69 | 3.82 | [244.60](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-mixed.pt) |
| [Bear-L](../../configs/L.yml) | 640 | 60.0748 / 60.0236 | 221.40 / 220.45 | 622.32 | 15.20 | 4.79 | [463.18](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-mixed.pt) |

## Latency and resident memory / 延迟与驻留显存

| Model | Backend | Mean ms | P95 ms | FPS | Round-mean CV % | Process resident MiB |
|---|---|---:|---:|---:|---:|---:|
| Bear-T | PyTorch FP32 original | 14.4467 | 14.6240 | 69.22 | 0.321 | 1018 |
| Bear-T | PyTorch AMP mixed | 12.8412 | 13.1246 | 77.87 | 0.874 | 1004 |
| Bear-T | TensorRT mixed | 2.7513 | 2.7625 | 363.46 | 0.174 | 830 |
| Bear-S | PyTorch FP32 original | 17.0492 | 17.1206 | 58.65 | 0.046 | 1100 |
| Bear-S | PyTorch AMP mixed | 16.4438 | 16.4917 | 60.81 | 0.123 | 1086 |
| Bear-S | TensorRT mixed | 3.3376 | 3.3465 | 299.62 | 0.010 | 874 |
| Bear-B | PyTorch FP32 original | 19.1829 | 19.2441 | 52.13 | 0.022 | 1338 |
| Bear-B | PyTorch AMP mixed | 15.6863 | 15.8985 | 63.75 | 0.907 | 1264 |
| Bear-B | TensorRT mixed | 3.8242 | 3.8334 | 261.50 | 0.010 | 960 |
| Bear-L | PyTorch FP32 original | 26.3672 | 26.4625 | 37.93 | 0.114 | 1866 |
| Bear-L | PyTorch AMP mixed | 15.2012 | 15.3345 | 65.78 | 0.935 | 1774 |
| Bear-L | TensorRT mixed | 4.7892 | 4.8058 | 208.80 | 0.061 | 1204 |

One NVIDIA GeForce RTX5090 GPU7, driver615.71.09, PyTorch2.14.1+cu130, TensorRT10.13.3.9, Python3.11.18. Power limit575W; no new clock settings/resets. Batch1×3×640×640,80 classes,100 warmups and3 rounds of300 CUDA-event samples, no CUDA Graph, TF32 disabled. Same real COCO image, PIL bilinear direct resize/ImageNet normalization, top300 outputs, no NMS. Mining/training and other GPU builds were stopped before timing. T/S engines built on GPU7; B/L built concurrently on identical GPU5/6, then all build contexts exited before serial GPU7 timing resumed.

GPU inputs through GPU predictions/top300 are timed; file IO, preprocessing, host transfers and COCO evaluation are excluded. TRT includes device-input copies. PyTorch enables cuDNN benchmarking and disables MHA fastpath, matching the traceable export path. These are eager baselines, not maximum torch.compile/CUDA Graph throughput. Latency need not increase monotonically with FLOPs: kernel choices, channel alignment and CPU launch overhead matter.

Memory is nvidia-smi process residency after warmup/timing in a separate process, **not a continuously sampled peak or training requirement**. Raw JSON also records PyTorch-managed peak allocations; this does not capture all TRT workspace and must not be compared as total backend memory. Mixed disk storage does not automatically halve CUDA residency when loaded into FP32 parameters.

## Full COCO accuracy / 全量COCO精度

| Model | Original-storage PT AMP | Mixed-storage PT AMP | Storage delta | Mixed TRT | TRT−mixed PT |
|---|---:|---:|---:|---:|---:|
| Bear-T | 56.47438 | 56.48514 | +0.01076 | 56.44489 | -0.04025 |
| Bear-S | 57.88084 | 57.88645 | +0.00561 | 57.85629 | -0.03015 |
| Bear-B | 59.01324 | 59.01919 | +0.00594 | 58.92433 | -0.09486 |
| Bear-L | 60.07185 | 60.07484 | +0.00299 | 60.02360 | -0.05124 |

All5,000 COCO val2017 images at640, original category IDs, standard box AP50:95 using faster-coco-eval/maxDets100 summary from top300 predictions. Same images/preprocessing and finite-output guards throughout. Both PyTorch storage variants load into FP32 models and run AMP at batch8; TensorRT is fixed batch1, so backend differences include batch-dependent numerics. Storage and deployment gates each allow at most0.1 AP-point loss; every model passed. This is not elementwise identity or a guarantee for other datasets, precisions or hardware. Small AP changes are not training improvements; historical epoch AP is unchanged.

## Reproduce / 复现

Use the [tested installation](../getting-started/INSTALL.md), matching config/classes and [weights](../models/WEIGHTS.md). Commands run from the repository root on an otherwise idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_engine.py --config configs/T.yml \
  --checkpoint weights/bear-t-coco-mixed.pt --output exports/T --build
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_latency.py --backend torch-amp \
  --config configs/T.yml --checkpoint weights/bear-t-coco-mixed.pt \
  --image assets/bear-input.jpg --output results/T-torch.json
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_latency.py --backend tensorrt \
  --config configs/T.yml --checkpoint weights/bear-t-coco-mixed.pt \
  --engine exports/T/model.engine --image assets/bear-input.jpg --output results/T-trt.json
CUDA_VISIBLE_DEVICES=0 python tools/validate_deployment.py --backend tensorrt \
  --config configs/T.yml --checkpoint weights/bear-t-coco-mixed.pt \
  --engine exports/T/model.engine --annotations data/coco/annotations/instances_val2017.json \
  --images data/coco/val2017 --output results/T-trt-accuracy.json
```

The sample image above is a convenient user example; recorded timings used the COCO image hash in raw results. Engines must be rebuilt on the target hardware/software; they are not published as portable downloads. The tested exporter uses TensorRT10 weak-typing constraints, not TensorRT11 strong typing.

[Raw per-model JSON, configuration/checkpoint/engine/input hashes](../../validation/benchmarks/deployment5090-20261114/README.md) · [Storage policy](../models/WEIGHTS.md).
