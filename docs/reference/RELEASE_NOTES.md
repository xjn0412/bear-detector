# Releases · 发布记录

Recommended downloads now use uniform FP16-bulk/FP32-decoder mixed storage for all sizes, validated on fullCOCO and TensorRT. Original FP32 EMA .pt files remain archival. The retired Small compact filename remains retired. / 全部尺寸统一混合存储为默认，原始FP32包保留归档。

[Current latency, accuracy and storage results](../benchmarks/DEPLOYMENT5090.md) · [当前部署结果](../benchmarks/DEPLOYMENT5090.zh-CN.md). The epoch/AP rows below refer to original training validation.

| Model | Release | Selected epoch | Full COCO AP50:95 |
|---|---|---:|---:|
| Tiny | [v0.2.0-tiny](https://github.com/xjn0412/bear-detector/releases/tag/v0.2.0-tiny) | 58 | 56.4597 |
| Small | [v0.1.0-weights](https://github.com/xjn0412/bear-detector/releases/tag/v0.1.0-weights) | 57 | 57.8987 |
| Base | [v0.2.1-base](https://github.com/xjn0412/bear-detector/releases/tag/v0.2.1-base) | 42 | 59.0008 |
| Large | [v0.3.0-large](https://github.com/xjn0412/bear-detector/releases/tag/v0.3.0-large) | 24 | 60.0728 |

[Download and verify](../models/WEIGHTS.md) · [Training policy](../benchmarks/TRAINING_POLICY.md). Selected training AP is not a new post-packaging rerun or TensorRT accuracy claim.
