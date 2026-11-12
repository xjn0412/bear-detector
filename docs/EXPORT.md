# ONNX and TensorRT export

[English](EXPORT.md) | [简体中文](EXPORT.zh-CN.md)

Install the optional export dependencies. Run from the repository root and keep the matching config, checkpoint and classes schema together. Only trust checkpoints from known sources; these full training checkpoints use Python/PyTorch serialization.

## Export the selected checkpoint

```bash
python tools/export_engine.py --config runs/my_dataset/config.yml \
  --checkpoint runs/my_dataset/training/best_stg1.pth \
  --output exports/my_dataset --build
```

Without `--build`, this exports/checks ONNX on CPU only. Default TensorRT precision is FP16 backbone/context/neck and FP32 decoder/sensitive reductions; master checkpoint staysFP32. `export.json` records checkpoint/graph/engine hashes and class count. It initially marks accuracy as unvalidated. Keep `classes.json` alongside the engine; it is copied from the prepared run automatically.

The graph is fixed batch1×3×640×640 with `images` float32 and `orig_target_sizes` int64 shape[1,2] in **width,height** order. Outputs: `labels`, original-image pixel `boxes` inxyxy, and `scores`. New class count, weights, input shape or incompatible TensorRT/GPU requires a new engine. Preprocessing is identical to [INFERENCE.md](INFERENCE.md).

The exporter uses an explicit equivalent bilinear sampling graph (`GatherElements`) instead of ONNX `GridSample`. Native TensorRT GridSample failed our checks even withFP32; do not remove this workaround based only on finite outputs. Native PyTorch training/inference still uses grid_sample.

## Check held-out accuracy before deployment

Use the **same** checkpoint, normalized validation JSON, images and model config for all backends:

```bash
python tools/evaluate.py --backend torch-fp32 --config runs/my_dataset/config.yml \
  --checkpoint runs/my_dataset/training/best_stg1.pth \
  --annotations runs/my_dataset/val.json --images data/my_dataset/val \
  --output results/my_dataset-fp32.json
python tools/evaluate.py --backend torch-amp --config runs/my_dataset/config.yml \
  --checkpoint runs/my_dataset/training/best_stg1.pth \
  --annotations runs/my_dataset/val.json --images data/my_dataset/val \
  --output results/my_dataset-amp.json
python tools/evaluate.py --backend tensorrt --config runs/my_dataset/config.yml \
  --checkpoint runs/my_dataset/training/best_stg1.pth --engine exports/my_dataset/model.engine \
  --annotations runs/my_dataset/val.json --images data/my_dataset/val \
  --output results/my_dataset-trt.json
python tools/compare_results.py --reference results/my_dataset-fp32.json \
  --candidate results/my_dataset-trt.json --max-drop 0.1 --output results/my_dataset-parity.json
```

Repeat comparison for the AMP candidate as well. The gate rejects differing checkpoint/annotation hashes, input shapes and class counts. AP drop tolerance0.1 means **0.1 AP percentage points**, not10%. Passing a small subset is a regression check, not proof of full-dataset or application accuracy. Inspect per-class/small-object performance and representative detections too.

For faster COCO regression tests, prepare a reproducible subset without model-score selection:

```bash
python tools/make_subset.py --annotations data/coco/annotations/instances_val2017.json \
  --output data/coco/annotations/val1k.json --count 1000 --seed 42
```

Use full validation for official results. Do not train on this held-out subset.

## Timing protocol and troubleshooting

Use an otherwise idle GPU. The evaluator runs50 warm-ups/200 timed iterations with CUDA events, includes top300, excludes CPU preprocessing/transfers; TensorRT additionally includes device-buffer copies. It separately reports image-to-CPU-output wall time. Do not compare this eager path to CUDA Graph or different batch/precision measurements.

- Missing decoder scope: exporter refuses an unsafe build; preserve stable ONNX names.
- Large AP gap: verify checkpoint/engine hashes, sampling implementation, RGB/normalization, original size order and label mapping. Finite outputs are insufficient.
- Out of memory: reduce training batch; keep inference batch1. Do not silently lower decoder precision.
- TensorRT11/strongly-typed API changes: not verified; use the documented TensorRT10 environment.
