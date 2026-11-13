# Comparison with related detectors

[English](COMPARISON.md) | [简体中文](COMPARISON.zh-CN.md)

Snapshot: 2026-11-14. Other models below are **authors’ published results, not rerun by us**. All report COCO val2017 box AP50:95 at 640. Pretraining, training schedules, parameter-count conventions and inference backends differ; this is context, not a controlled architecture ablation.

| Model | Parameters | COCO AP50:95 | Evidence / training distinction |
|---|---:|---:|---|
| Bear-T | 52.05M | 56.4597 | Full COCO validation; EMA epoch58 |
| Bear-L | 221.40M | 60.0728 | Full COCO validation; EMA epoch24 |
| Bear-B | 112.11M | 59.0008 | Our measured EMA epoch42; DINOv3 ConvNeXt-B external backbone pretraining; COCO detection supervision |
| Bear-S | 73.68M | 57.90 | Our measured EMA epoch57; DINOv3 ConvNeXt-S external backbone pretraining; COCO detection supervision |
| DEIMv2-L | 32.2M | 56.0 | Official model zoo; DINOv3-based detector |
| DEIMv2-X | 50.3M | 57.8 | Official model zoo; DINOv3-based detector |
| D-FINE-L | 31M | 54.0 | Official COCO row |
| D-FINE-X | 62M | 55.8 | Official COCO row |
| D-FINE-X, Objects365→COCO | 62M | 59.3 | Additional detection pretraining; separate regime |
| RT-DETRv2-L | 42M | 53.4 | Official COCO row |
| RT-DETRv2-X | 76M | 54.3 | Official COCO row |
| YOLO26-L, default head | 24.8M | 55.0 | Official fused/deployed parameter count; AP uses NMS head |
| YOLO26-X, default head | 55.7M | 57.5 | Official fused/deployed parameter count; AP uses NMS head |

YOLO26 official NMS-free AP is 54.4 for L and 56.9 for X; its published speed uses that NMS-free path. Its released recipe includes Objects365 pretraining. Do not label it all-data COCO-only. Comparisons are tied to the cited model versions; later releases are not evaluated here.

**Interpretation:** Bear-B has higher measured AP than several published COCO rows, but uses substantially more parameters and external backbone pretraining. It does not establish a superior accuracy/latency tradeoff. DEIMv2 is the closest design reference; D-FINE and RT-DETRv2 are decoder-family references; YOLO26 is an important deployment baseline. Naming the sizes L/X/S does not mean equal model capacity across projects.

## Speed comparison policy

Keep our measured RTX5090 result in [BENCHMARKS.md](BENCHMARKS.md) separate from other authors’ T4 timings. Do not divide those latencies to claim a speedup. A fair speed comparison needs the same GPU, software, 640 input, batch1, precision policy, warmup, synchronization, preprocessing and postprocessing boundaries, and accuracy-checked exported checkpoints. No new baseline GPU benchmarks were run for this table. Bear FLOPs are now CPU-profiled including attention matrix products and bilinear sampling; see [counting scope](COMPLEXITY.md). External FLOPs use different profilers and are not matched speed measurements.

## Primary sources

- [DEIMv2 model zoo](https://github.com/Intellindust-AI-Lab/DEIMv2#1-model-zoo)
- [D-FINE model zoo](https://github.com/Peterande/D-FINE#model-zoo)
- [RT-DETR official results](https://github.com/lyuwenyu/RT-DETR)
- [YOLO26 official metrics](https://docs.ultralytics.com/models/yolo26/)
- [YOLO26 training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/)

[Bear FLOPs / 计算量](COMPLEXITY.md) · [Checkpoint size audit / 权重大小审计](../models/WEIGHT_SIZE.md).
