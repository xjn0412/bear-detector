# Supplemental context: extra detection data and unverified regimes

[Primary COCO-only comparison](COMPARISON.md) · [中文](COMPARISON_EXTRA_DATA.zh-CN.md)

These rows are excluded from the primary comparison. YOLO26 and D-FINE O365 use extra detection pretraining; YOLO11 provenance was not established in this audit; YOLO27 is unreleased and its training data was not verified. The figures do not establish superiority over COCO-only detector training.

| Model | COCO val AP50:95 | Params M¹ | GFLOPs² | Source |
|---|---:|---:|---:|---|
| YOLO26-n | 40.90 | 2.40 | 5.50 | [Source](https://github.com/ultralytics/ultralytics#models) |
| YOLO26-s | 48.60 | 9.50 | 20.90 | [Source](https://github.com/ultralytics/ultralytics#models) |
| YOLO26-m | 53.10 | 20.40 | 68.40 | [Source](https://github.com/ultralytics/ultralytics#models) |
| YOLO26-l | 55.00 | 24.80 | 86.80 | [Source](https://github.com/ultralytics/ultralytics#models) |
| YOLO26-x | 57.50 | 55.70 | 194.40 | [Source](https://github.com/ultralytics/ultralytics#models) |
| YOLO11-n | 39.50 | 2.60 | 6.50 | [Source](https://docs.ultralytics.com/models/yolo11/) |
| YOLO11-s | 47.00 | 9.40 | 21.60 | [Source](https://docs.ultralytics.com/models/yolo11/) |
| YOLO11-m | 51.50 | 20.10 | 68.10 | [Source](https://docs.ultralytics.com/models/yolo11/) |
| YOLO11-l | 53.40 | 25.30 | 87.20 | [Source](https://docs.ultralytics.com/models/yolo11/) |
| YOLO11-x | 54.70 | 56.90 | 195.30 | [Source](https://docs.ultralytics.com/models/yolo11/) |
| D-FINE-S O365 | 50.70 | 10.00 | 25.00 | [Source](https://github.com/Peterande/D-FINE#model-zoo) |
| D-FINE-M O365 | 55.10 | 19.00 | 57.00 | [Source](https://github.com/Peterande/D-FINE#model-zoo) |
| D-FINE-L O365 | 57.30 | 31.00 | 91.00 | [Source](https://github.com/Peterande/D-FINE#model-zoo) |
| D-FINE-X O365 | 59.30 | 62.00 | 202.00 | [Source](https://github.com/Peterande/D-FINE#model-zoo) |
| YOLO27-n (preview) | 42.60 | 3.00 | 7.20 | [Source](https://docs.ultralytics.com/models/yolo27/) |
| YOLO27-s (preview) | 50.10 | 11.80 | 28.20 | [Source](https://docs.ultralytics.com/models/yolo27/) |
| YOLO27-m (preview) | 56.40 | 24.60 | 64.60 | [Source](https://docs.ultralytics.com/models/yolo27/) |
| YOLO27-l (preview) | 60.50 | 72.30 | 166.70 | [Source](https://docs.ultralytics.com/models/yolo27/) |

## Linked leaderboard context / 指定排行榜背景

| Model | AP | Data / protocol | Source |
|---|---:|---|---|
| Focal-Stable-DINO | 64.60 | IN-22K + Objects365; no TTA; larger-resolution recipe | [Source](https://arxiv.org/html/2304.13027v1) |
| DINO Swin-L (TTA) | 63.20 | IN-22K + Objects365; TTA | [Source](https://arxiv.org/html/2304.13027v1#S2) |
| Grounding DINO-L (2000) | 63.00 | O365 + OI + GoldG; COCO fine-tuned; enlarged input | [Source](https://arxiv.org/html/2303.05499v5) |

[SOTA2 exact page](https://www.sota2.com/research/sota/object-detection-on-coco-2017-val) · [Selected provenance](../../validation/benchmarks/comparison-20261114/leaderboard-source.json). Snapshot2026-11-14; first100renderedrows, not a global rank. Different detection data, resolution and TTA; no matched speed claim. Bear-L60.07 versus Focal64.6 is a4.53-point numerical gap across regimes, not an equal-data performance judgment. Extra-data rows are retained solely to preserve the user's requested leaderboard context.
