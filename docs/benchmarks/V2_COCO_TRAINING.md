# V2 COCO training and GPU scaling

COCO training launched2026-11-16 in the requested **L → B → S → T** sequence. Each selected run uses8 RTX5090 GPUs, input640, micro-batch1/GPU and effective global batch8. Each model has an independent fresh detector initialized with its corresponding DINOv3 ConvNeXt backbone. Timing updates are discarded before formal training.

Detection supervision is **COCO train2017 only** (118,287 images); every formal epoch validates the selected EMA on all5,000 val2017 images. Maximum100 epochs/model; stop after20 consecutive epochs without AP50:95 improvement. V2 weights and AP are not yet available. DINOv3 representation pretraining uses external LVD data, as disclosed for v1.

## Four versus eight GPUs

Throughput includes real augmentation, criterion, backward, fused AdamW and EMA. Both counts keep effective batch8:4 GPUs accumulate2 micro-batches,8 GPUs accumulate1. The4-GPU placement is GPU0–3;8-GPU placement is GPU0–7. No mining or other GPU jobs ran during timing.

| Model | 4 GPU images/s | 8 GPU images/s | 4 GPU estimated min/epoch | 8 GPU estimated min/epoch | Selected GPUs |
|---|---:|---:|---:|---:|---:|
| Bear-L v2 | 10.01 | 18.34 | 198.73 | 108.70 | **8** |
| Bear-B v2 | 14.07 | 21.90 | 141.54 | 90.95 | **8** |
| Bear-S v2 | 14.32 | 24.90 | 139.30 | 80.16 | **8** |
| Bear-T v2 | 21.31 | 26.56 | 93.84 | 75.16 | **8** |

For one epoch of each model in the requested sequential order, the estimates sum to **573.42 minutes on4 GPUs versus 354.97 minutes on8 GPUs**, about **38.1% less elapsed time** for equal epoch budgets. This selects the minimum among the measured4/8-GPU sequential configurations. It is not an exhaustive search over batch sizes, parallel model schedules or time to convergence.

These are **short-window estimates**, not measured complete epochs. A fixed1,024-image timing subset of train2017 and512-image timing subset of val2017 were CRC checked and identified in the membership manifest. Training is extrapolated to118,287 images and validation to5,000. Checkpoint I/O is excluded and common to both counts. L uses20 warmup +20 timed optimizer windows; B/S/T use60 warmup +20 timed windows. Both GPU counts use the same warmup within each model. The sampled training phase is epoch index5, including the heavier augmentation recipe; initial epochs may differ.

The first B/4-GPU attempt had one scaled-gradient overflow in its timed window and was rejected. Its raw result is preserved separately; the longer-warmup replacement completed all20 timed updates. Every accepted result has zero timed update skips. GradScaler backoff during warmup is recorded rather than hidden.

## Numeric safeguards and checkpoints

FP32 master parameters, four-scale projections/context/residual injection, complete decoder and criterion remain protected. Checked FP16 AMP is used for the CNN/neck. Nonfinite forward/loss values cannot be published as valid training; scaled-gradient overflow causes every rank to skip together, lower the loss scale, reset growth tracking and leave EMA unchanged. FP32 fallback policy and optimizer/scaler/EMA state are saved.

Gradient accumulation advances the LR schedule per optimizer window. Differential CPU tests verify accumulated updates match a direct batch and that overflow blocks optimizer/EMA updates. A two-GPU synthetic training run verifies actual DDP accumulation, checkpoint control, EMA and patience. These checks are not detector accuracy results.

Each model writes `epochs.csv`, `epochs.jsonl` and `status.json`, with per-epoch AP50:95/AP50, loss, scale, updates/skips and timing. `best.pt` contains selected FP32 EMA inference weights; `best-full.pth` and `last.pth` retain full resume state. Patience and best-validation metadata survive resume.

[Raw timing reports, memberships and hardware](../../validation/training/v2-coco-20261116/README.md) · [Architecture and commands](../models/ARCHITECTURE_V2.md)
