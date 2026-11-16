# TensorRT precision revalidation — 2026-11-16

Rebuilt the selected CR7-DET Bear-L checkpoint on RTX5090 using TensorRT10.13.3.9, CUDA13, PyTorch2.14.1 and ONNX1.19.1. This validates this checkpoint/task and fixed batch1 RGB640 inputs; it is not a COCO-wide or all-model certification. Checkpoint SHA256: `8e8e3df65c6473aea4e4aa8acb7a500006f6e21f6256b5148f8cbaf98fe1dafd`.

| Backend | Test AP50:95 (%) | Delta vs FP32 (points) | Mean ms/image | FPS |0.20-point accuracy target |
|---|---:|---:|---:|---:|---|
| PyTorch FP32 |58.5209|0|27.26–28.01|35.7–36.7|Reference|
| TensorRT mixed |58.2883|−0.2326|4.810|207.9|Outside target|
| TensorRT FP32 backbone/context + decoder, FP16 neck |58.3454|−0.1755|13.254|75.4|Pass|

Both engines produced finite predictions on the full410-image held-out test split and on8 captured industrial training inputs used only as finite-output stress inputs. The latter are not extra labeled AP samples. P5 projection, decoder and sensitive reductions are explicitly constrained to FP32. The conservative engine additionally keeps the backbone/context FP32. Full engine/checkpoint/ONNX hashes and precision scopes are in the export reports.

Precision checking used the410-image validation split first (FP32 AP59.7993; mixed59.8183; FP32-backbone59.7305), with an absolute0.20-point AP target set before test. The mixed mode passed validation but missed that target on held-out test; its result remains reported. A separately declared conservative engine was then built and checked on both splits. This is a deployment precision comparison on one fixed checkpoint, not test-based training or detector checkpoint selection. Neither mode is claimed to be exactly equivalent to FP32.

Latency uses isolated GPU inputs through top300 detections, batch1, no CUDA Graph,50 warmup calls and3×100 timed calls. Preprocessing and host transfers are excluded; device-buffer copies and runtime finite guards are included. Each TensorRT mode and its FP32 reference ran alone on its assigned5090 (GPU0 for mixed, GPU1 for conservative). This is model inference timing, not camera-to-alarm throughput.

COCO ground-truth annotation IDs were reindexed to1..N in memory for all accuracy passes. Images, boxes and checkpoint remain unchanged. No invalid output was converted to zero or silently discarded. The static TensorRT engine has no Python FP32 retry; the runtime wrapper refuses nonfinite outputs.

Use the conservative engine when requiring the tighter measured accuracy target; the mixed engine offers the larger speedup with the measured0.23-point AP loss. Previously built engines remain unchanged and require rebuilding.

## Reproduce

The installed export environment was isolated from the training environment. The older TensorRT10.13.3.9 dependency metadata refers to a deprecated CUDA runtime stub; installation used its matching frontend/libs/bindings wheels with `--no-deps` only after verifying that PyTorch's existing CUDA13 runtime loads `libcudart.so.13`. The installed versions and actual successful engine build are evidence of this environment, not a recommendation to omit unknown dependencies.

```bash
CUDA_VISIBLE_DEVICES=0 <export-python> tools/benchmarks/validate_industrial_tensorrt.py --base <base> --dataset cr7 --precision mixed --gpu 0
CUDA_VISIBLE_DEVICES=1 <export-python> tools/benchmarks/validate_industrial_tensorrt.py --base <base> --dataset cr7 --precision fp32-backbone --gpu 1
```

The worker adopts existing ONNX/export metadata where available, saves every build/evaluation log, and writes `COMPLETE.json`. See [`validation/benchmarks/industrial-20261116/tensorrt-cr7/`](../../validation/benchmarks/industrial-20261116/tensorrt-cr7/) for raw accuracy/latency/stress/export reports. Large GPU-specific engine binaries and raw datasets are not committed.
