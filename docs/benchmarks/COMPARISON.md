# Model comparison and performance analysis

[English](COMPARISON.md) | [简体中文](COMPARISON.zh-CN.md)

Snapshot: **2026-11-14**. Presentation follows the clear plots and model tables in [Ultralytics' GitHub repository](https://github.com/ultralytics/ultralytics). Bear results are our full 5,000-image COCO val2017 measurements; external results are published figures, not rerun. AP means bounding-box AP averaged over IoU 0.50–0.95, in points on a 0–100 scale.

## Accuracy versus model cost at 640

![Accuracy versus parameters and FLOPs](../../assets/benchmarks/comparison-efficiency.png)

All rows in this section use 640 input. Bear uses the current mixed-storage PyTorch AMP evaluation (batch8, full FP32 decoder); historical training AP is preserved separately in [training history](TRAINING_HISTORY.md). Tiny/Small/Base/Large names indicate relative sizes within a family, not matched capacities across families.

| Model | AP50:95 | Params M¹ | GFLOPs² | Source |
|---|---:|---:|---:|---|
| Bear-T | 56.49 | 51.10 | 132.18 | [Measured](https://github.com/xjn0412/bear-detector/blob/main/docs/benchmarks/DEPLOYMENT5090.md) |
| Bear-S | 57.89 | 72.73 | 201.21 | [Measured](https://github.com/xjn0412/bear-detector/blob/main/docs/benchmarks/DEPLOYMENT5090.md) |
| Bear-B | 59.02 | 111.16 | 310.75 | [Measured](https://github.com/xjn0412/bear-detector/blob/main/docs/benchmarks/DEPLOYMENT5090.md) |
| Bear-L | 60.07 | 220.45 | 622.32 | [Measured](https://github.com/xjn0412/bear-detector/blob/main/docs/benchmarks/DEPLOYMENT5090.md) |
| YOLO26-n | 40.90 | 2.40 | 5.50 | [YOLO26](https://github.com/ultralytics/ultralytics#models) |
| YOLO26-s | 48.60 | 9.50 | 20.90 | [YOLO26](https://github.com/ultralytics/ultralytics#models) |
| YOLO26-m | 53.10 | 20.40 | 68.40 | [YOLO26](https://github.com/ultralytics/ultralytics#models) |
| YOLO26-l | 55.00 | 24.80 | 86.80 | [YOLO26](https://github.com/ultralytics/ultralytics#models) |
| YOLO26-x | 57.50 | 55.70 | 194.40 | [YOLO26](https://github.com/ultralytics/ultralytics#models) |
| YOLO11-n | 39.50 | 2.60 | 6.50 | [YOLO11](https://docs.ultralytics.com/models/yolo11/) |
| YOLO11-s | 47.00 | 9.40 | 21.60 | [YOLO11](https://docs.ultralytics.com/models/yolo11/) |
| YOLO11-m | 51.50 | 20.10 | 68.10 | [YOLO11](https://docs.ultralytics.com/models/yolo11/) |
| YOLO11-l | 53.40 | 25.30 | 87.20 | [YOLO11](https://docs.ultralytics.com/models/yolo11/) |
| YOLO11-x | 54.70 | 56.90 | 195.30 | [YOLO11](https://docs.ultralytics.com/models/yolo11/) |
| DEIMv2-S | 50.90 | 9.70 | 49.50 | [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#1-model-zoo) |
| DEIMv2-M | 53.00 | 18.10 | 84.10 | [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#1-model-zoo) |
| DEIMv2-L | 56.00 | 32.20 | 144.50 | [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#1-model-zoo) |
| DEIMv2-X | 57.80 | 50.30 | 200.00 | [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#1-model-zoo) |
| D-FINE-S | 48.50 | 10.00 | 25.00 | [D-FINE](https://github.com/Peterande/D-FINE#model-zoo) |
| D-FINE-M | 52.30 | 19.00 | 57.00 | [D-FINE](https://github.com/Peterande/D-FINE#model-zoo) |
| D-FINE-L | 54.00 | 31.00 | 91.00 | [D-FINE](https://github.com/Peterande/D-FINE#model-zoo) |
| D-FINE-X | 55.80 | 62.00 | 202.00 | [D-FINE](https://github.com/Peterande/D-FINE#model-zoo) |
| D-FINE-S O365 | 50.70 | 10.00 | 25.00 | [D-FINE](https://github.com/Peterande/D-FINE#model-zoo) |
| D-FINE-M O365 | 55.10 | 19.00 | 57.00 | [D-FINE](https://github.com/Peterande/D-FINE#model-zoo) |
| D-FINE-L O365 | 57.30 | 31.00 | 91.00 | [D-FINE](https://github.com/Peterande/D-FINE#model-zoo) |
| D-FINE-X O365 | 59.30 | 62.00 | 202.00 | [D-FINE](https://github.com/Peterande/D-FINE#model-zoo) |
| RT-DETRv2-S | 48.10 | 20.00 | 60.00 | [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR#implementations) |
| RT-DETRv2-M | 51.90 | 36.00 | 100.00 | [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR#implementations) |
| RT-DETRv2-L | 53.40 | 42.00 | 136.00 | [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR#implementations) |
| RT-DETRv2-X | 54.30 | 76.00 | 259.00 | [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR#implementations) |

¹ Bear uses deploy counts here; full checkpoint counts are 52.05/73.68/112.11/221.40M. External counts retain the authors' conventions; YOLO uses fused/deploy counts. ² FLOPs use each project's profiler, not an identical accounting method. Bear explicitly includes attention products and bilinear sampling. Neither axis is a measured latency substitute.

YOLO26 rows use default NMS-head AP. Official end-to-end AP is 40.1/47.8/52.5/54.4/56.9 for n/s/m/l/x; its published latency uses the end-to-end path. Its [recipe includes Objects365](https://docs.ultralytics.com/guides/yolo26-training-recipe/). D-FINE O365 is explicitly a separate detection-pretraining regime. DEIMv2 uses DINOv3 pretrained/distilled features. Bear uses DINOv3 LVD-1689M backbone pretraining and COCO train2017 detection supervision; it is not all-data COCO-only or zero-shot. The dashed hollow-marker YOLO27 curve is an official unreleased preview, separated from released model rows below; its figures may change.

## Official YOLO27 preview (unreleased)

| Model | AP at640 | Deploy params M | GFLOPs |
|---|---:|---:|---:|
| YOLO27-n | 42.6 | 3.0 | 7.2 |
| YOLO27-s | 50.1 | 11.8 | 28.2 |
| YOLO27-m | 56.4 | 24.6 | 64.6 |
| YOLO27-l | 60.5 | 72.3 | 166.7 |

[Official preview source](https://docs.ultralytics.com/models/yolo27/): weights and implementation are not public, and results remain preliminary. At640 the preview Large is0.43 AP above Bear-L with about one-third of its deployment parameters and27% of its reported FLOPs. This is a serious efficiency target if reproduced after release. Its RTX PRO6000/TensorRT11 latency is not comparable to our5090/TRT10 timing, and we do not assume its pretraining data.

## Context from the linked leaderboard

![Selected leaderboard models and Bear](../../assets/benchmarks/comparison-leaderboard.png)

The exact [SOTA2 COCO 2017 (val) page](https://www.sota2.com/research/sota/object-detection-on-coco-2017-val) was retrieved directly, not replaced with the similarly named COCO test-dev or COCO (val) pages. Its leading displayed result was Focal-Stable-DINO at 64.6 AP. We selected identifiable rows and checked their values against papers. This is a dated selection, not a complete leaderboard, an official Bear submission, or an assigned rank. The first rendered page contained 100 rows, including duplicate model variants; we do not infer a global ranking from it.

| Selected published model | val AP | Protocol | Primary source |
|---|---:|---|---|
| Focal-Stable-DINO | 64.60 | IN-22K + Objects365; no TTA; larger-resolution recipe | [Source](https://arxiv.org/html/2304.13027v1) |
| DINO Swin-L (TTA) | 63.20 | IN-22K + Objects365; TTA | [Source](https://arxiv.org/html/2304.13027v1#S2) |
| Grounding DINO-L (2000) | 63.00 | O365 + OI + GoldG; COCO fine-tuned; enlarged input | [Source](https://arxiv.org/html/2303.05499v5) |
| Co-DINO Swin-L (36ep LSJ) | 60.70 | Primary-source supplemental baseline; different resolution/recipe | [Source](https://github.com/Sense-X/Co-DETR#model-zoo) |

Co-DINO is a supplemental primary-source comparison, not a claim that its row appeared on the retrieved first page. Larger resolution, extra detection/grounding pretraining, mask supervision and TTA can change AP substantially. The chart keeps these regimes visible rather than treating them as 640 matched experiments. Grounding DINO's fine-tuned result is distinct from its zero-shot result. The 66.0 Co-DETR **test-dev** headline is excluded from the val comparison; the official repository separately lists a 65.9 val model under another pretrained configuration.

## What the results say about Bear

**Absolute accuracy is competitive with many real-time baselines, but parameter efficiency needs work.** Bear-S reaches 57.89 AP versus DEIMv2-X's 57.8 while using 72.73M versus 50.3M parameters: +0.09 AP for about 45% more parameters. That small difference does not establish a meaningful accuracy advantage without repeated runs. Bear-T has nearly the same reported deployment size as DEIMv2-X (51.10M vs50.3M), but is 1.31 AP lower. This is the clearest evidence against claiming an architecture-efficiency win.

Bear-B improves over YOLO26-x's published NMS-head result by 1.52 AP, with about 2.00× the reported deployment parameters and 1.60× the reported FLOPs. Bear-L improves by 2.57 AP, with about 3.96× parameters and 3.20× FLOPs. D-FINE-X with Objects365 reaches 59.3 AP at62M parameters, slightly above Bear-B's59.02 at111.16M. These are cross-recipe comparisons, but they expose the cost of obtaining our higher AP through scaling.

**Within the Bear family, Base is a useful balance; Large buys the highest accuracy.** Under the same RTX5090 TensorRT protocol, Base gives58.92 AP at3.82ms; Large gives60.02 at4.79ms. Moving Base→Large adds1.10 AP and about25% latency, while nearly doubling parameters. Tiny is the fastest at2.75ms/363.46 model-only FPS. These are conclusions within Bear; external T4/A100 numbers cannot establish a cross-family speedup.

**We remain below the linked leaderboard's leading displayed result.** Bear-L's60.07 is4.53 AP below Focal-Stable-DINO's64.6, and4.93 below65.0. Focal's689M parameter count is about3.13× Bear-L's deploy count, but it also uses Objects365 and a higher-resolution recipe. No matched latency result exists here, so neither absolute AP nor parameter ratios prove better deployment efficiency. The linked snapshot is incomplete as a universal SOTA inventory: newer or differently categorized results may live elsewhere.

## Object size and likely next experiments

| Model | AP small | AP medium | AP large |
|---|---:|---:|---:|
| Bear-T | 39.50 | 60.88 | 73.50 |
| Bear-S | 41.03 | 62.93 | 75.52 |
| Bear-B | 42.72 | 63.99 | 75.86 |
| Bear-L | 43.77 | 64.89 | 76.90 |
| Focal-Stable-DINO (published, different recipe) | 50.40 | 68.50 | 78.50 |

Against Focal-Stable-DINO's published breakdown, Bear-L's gaps are6.63 AP on small objects,3.61 on medium, and1.60 on large. Small-object accuracy is the largest observed gap in this comparison. Different resolutions and recipes prevent assigning the cause solely to our architecture.

The next useful ablations would be: (1) evaluate and then fine-tune at768/960 while measuring small-object AP and latency; (2) test a narrower neck/decoder or distillation to improve the 50–110M-parameter region; (3) benchmark DEIMv2-X and YOLO26-x on the same5090 with validated exports; and (4) compare equal training budgets and multiple seeds. These are proposed experiments, not work launched by this documentation update, and none guarantees65 AP.

## Measured speed within Bear

![Bear backend-specific accuracy versus measured latency](../../assets/benchmarks/comparison-latency.png)

Single RTX5090, batch1, fixed640,100 warmups,3×300 CUDA-event samples, no CUDA Graph. GPU inputs through top300; file IO, preprocessing and host transfers excluded; TensorRT includes device-input copies. Full COCO AP uses PyTorch batch8 versus TensorRT batch1, so each plotted backend uses its own measured AP. PyTorch latency is not monotonic with parameter count; kernels, alignment and launch overhead matter. [Complete protocol, memory and raw results](DEPLOYMENT5090.md).

## Data and reproduction

[Exact plotted data and source URLs](../../validation/benchmarks/comparison-20261114/comparison.json) · [Selected SOTA2 row provenance](../../validation/benchmarks/comparison-20261114/leaderboard-source.json). SVG versions sit beside the PNGs in assets/benchmarks. Regenerate with `python tools/plot_comparison.py` after installing matplotlib; this performs no model inference or training. Original source links and snapshot dates are retained so later metric changes are not silently mixed into these figures.
