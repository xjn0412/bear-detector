# Complexity and performance-table conventions

[English](COMPLEXITY.md) | [简体中文](COMPLEXITY.zh-CN.md)

Measured2026-11-14, batch1×3×640×640,80 classes. CPU-only; no mining changes, training restart or TensorRT rebuild.

| Model | Input | AP50:95 | Params M (full / deploy) | GFLOPs | Config | Download MB (lossless gzip) |
|---|---:|---:|---:|---:|---|---|
| Bear-T | 640 | 56.4597 | 52.05 / 51.10 | 132.18 | [T.yml](../../configs/T.yml) | [193.77](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-ema.pt.gz) |
| Bear-S | 640 | 57.8987 | 73.68 / 72.73 | 201.21 | [S.yml](../../configs/S.yml) | [274.17](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt.gz) |
| Bear-B | 640 | 59.0008 | 112.11 / 111.16 | 310.75 | [B.yml](../../configs/B.yml) | [416.91](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-ema.pt.gz) |
| Bear-L | 640 | 60.0728 | 221.40 / 220.45 | 622.32 | [L.yml](../../configs/L.yml) | [822.79](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-ema.pt.gz) |

## Counting scope / 统计范围

Uses PyTorch2.14.1 `FlopCounterMode` on a seed42-initialized FP32 model in eval/deploy mode. No weights or dataset are needed for these shape-based counts; initialized outputs imply no accuracy claim. Deployment fuses reparameterizable convolutions and removes unused intermediate score heads. Full checkpoint parameters must not be compared silently with YOLO deployed parameters.

Convolution/linear/matmul use2 FLOPs per MAC. SDPA is forced to its math decomposition so QKᵀ and AV are included rather than losing CPU flash-attention operations. Deformable bilinear sampling adds7 arithmetic operations per sampled output element (4 multiplies+3 sums). Sampling-coordinate arithmetic, activations, normalization, softmax, residual elementwise operations, indexing/top-k, preprocessing, memory traffic, criterion and backward are excluded. GFLOPs are therefore a **scoped arithmetic estimate, not an exact count of every hardware operation**.

No32×32 proxy or area extrapolation is used: query selection and global attention do not generally scale linearly with image area. The actual640 forward is executed. FP32 CPU profiling changes arithmetic precision, not tensor geometry; deployment retains FP16 bulk+entire FP32 decoder. FLOPs do not predict latency: sampling, precision and memory behavior matter.

## Breakdown / 组成

| Model | Backbone + context/detail GFLOPs | Neck GFLOPs | Decoder GFLOPs | Context alone GFLOPs |
|---|---:|---:|---:|---:|
| Bear-T | 77.69 | 47.73 | 6.76 | 3.33 |
| Bear-S | 146.72 | 47.73 | 6.76 | 3.33 |
| Bear-B | 256.26 | 47.73 | 6.76 | 3.49 |
| Bear-L | 567.83 | 47.73 | 6.76 | 3.80 |

Context is a subset of the backbone column; do not add it twice. Tiny spends36.1% of counted arithmetic in the neck, versus2.5% in context alone. A future speed ablation should prioritize neck width/depth rather than assume removing attention yields a large gain. Base/Large are increasingly CNN-backbone dominated. This is a proposed direction; no architecture, training or weights were changed.

## Reproduce / 复现

Run from the repository root with the documented PyTorch environment:

```bash
python tools/profile_model.py --config configs/T.yml --output results/T-profile.json
python tools/profile_model.py --config configs/S.yml --output results/S-profile.json
python tools/profile_model.py --config configs/B.yml --output results/B-profile.json
python tools/profile_model.py --config configs/L.yml --output results/L-profile.json
```

[Raw profiles and storage audits](../../validation/benchmarks/complexity640/README.md) include config hashes, counts by module/operator and exact exclusions. The tool requires PyTorch with `torch.utils.flop_counter` and math SDPA (tested2.14.1). It deliberately does not advertise CPU profiling time as inference latency.

## Reference / 参考

[YOLO26 performance](https://docs.ultralytics.com/models/yolo26/#performance-metrics) reports fused/deployed parameters and two FLOPs per MAC via[Ultralytics profiling code](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py). Its FLOPs use THOP, not this counter; omitted operations can differ. Its T4 TensorRT speed is not comparable directly with historical Bear RTX5090 speed. [Existing same-checkpoint backend timings](BENCHMARKS.md).
