# Weight size versus parameter count

[English](WEIGHT_SIZE.md) | [简体中文](WEIGHT_SIZE.zh-CN.md)

**Update: recommended downloads now use validated FP16-bulk/FP32-decoder mixed storage for all four sizes. The FP32 size table below is retained as an audit of original archives.** [Current downloads](WEIGHTS.md) · [Deployment validation](../benchmarks/DEPLOYMENT5090.md).

**Bear files are not unexpectedly bloated; the architecture is genuinely heavy.** Original Bear archives retain full FP32 EMA tensors at approximately4 bytes per parameter. YOLO26 public checkpoints store FP16 parameters at approximately2 bytes each. YOLO performance tables also report fused/deployed counts, while downloads retain a larger training architecture.

MB=1,000,000 bytes; M=1,000,000 parameters. Gzip bytes are download/storage size, not GPU memory. AMP training dtype does not determine checkpoint storage dtype.

## Bear releases

| Model | Full params M | Parameter bytes FP32 MB | Actual .pt MB | Gzip download MB | Excess over parameter bytes |
|---|---:|---:|---:|---:|---:|
| Bear-T | 52.04883 | 208.20 | 208.86 | 193.77 | 0.317% |
| Bear-S | 73.68339 | 294.73 | 295.46 | 274.17 | 0.247% |
| Bear-B | 112.10825 | 448.43 | 449.16 | 416.91 | 0.162% |
| Bear-L | 221.39830 | 885.59 | 886.32 | 822.79 | 0.082% |

Audit: every floating state tensor is FP32. Packages contain model, last_epoch and metadata, with no duplicate EMA copy, optimizer or GradScaler. The extra0.08–0.32% is buffers (anchors/normalization statistics), metadata and serialization overhead. Gzip is lossless; decompression changes no tensors. The originals remain archival; separately validated mixed packages are now the defaults.

## YOLO26 public checkpoints

| Model | Published deployed params M | Checkpoint params M | Stored parameter dtype | Expected param MB | Actual .pt MB | Published GFLOPs |
|---|---:|---:|---|---:|---:|---:|
| YOLO26n | 2.4 | 2.57228 | FP16 | 5.14 | 5.54 | 5.5 |
| YOLO26s | 9.5 | 10.00978 | FP16 | 20.02 | 20.42 | 20.9 |
| YOLO26m | 20.4 | 21.89625 | FP16 | 43.79 | 44.26 | 68.4 |
| YOLO26l | 24.8 | 26.29970 | FP16 | 52.60 | 53.21 | 86.8 |
| YOLO26x | 55.7 | 58.99337 | FP16 | 117.99 | 118.67 | 194.4 |

File sizes come from the official `ultralytics/assets` `v8.4.0` release API. Full checkpoint counts/dtypes were parsed from tensor metadata in ZIP data.pkl; no YOLO forward, full-archive SHA verification, AP/FLOPs or speed rerun was performed. Tensor shapes/dtypes can be audited without evaluating weight values. See the[official performance table](https://docs.ultralytics.com/models/yolo26/#performance-metrics) explanation of deployment versus checkpoint counts.

## Is Bear larger than expected?

YOLO26-X stores58.99M checkpoint parameters in a118.67MB FP16 file. Bear-B has112.11M parameters and a449.16MB FP32 file: approximately1.90× as many parameters and2× the bytes per parameter, explaining roughly3.8× the file size. Comparing Bear gzip with an uncompressed YOLO .pt would mix packaging conventions.

However, Bear-Tiny is not an edge-small detector:52.05M full/51.10M deployed parameters are already close to YOLO26-X's55.7M deployed model. Tiny's pretrained CNN is27.82M; Bear adds24.23M parameters, including a10.40M neck and9.90M decoder. Size names are relative within the Bear family. Future architectural slimming should test a narrower/shallower neck and smaller decoder, with retraining and accuracy evaluation, rather than just compress files.

FP16 storage could theoretically halve floating bytes but rounds weights and needs per-model accuracy validation; do not call model.half() on a model requiring an FP32 decoder. A lossless format such as safetensors can reduce small serialization overhead, not turn FP32's4 bytes into2.

[Raw measurement evidence](../../validation/benchmarks/complexity640/README.md) · [Downloads](WEIGHTS.md) · [Complexity](../benchmarks/COMPLEXITY.md).

## Current mixed-storage downloads

| Model | Original gzip MB | Mixed gzip MB | Reduction |
|---|---:|---:|---:|
| Bear-T | 193.77 | 115.08 | 40.6% |
| Bear-S | 274.17 | 155.18 | 43.4% |
| Bear-B | 416.91 | 226.34 | 45.7% |
| Bear-L | 822.79 | 428.79 | 47.9% |
