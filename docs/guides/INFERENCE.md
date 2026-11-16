# PyTorch inference

[English](INFERENCE.md) | [简体中文](INFERENCE.zh-CN.md)

Use the config matching your checkpoint and the exact preprocessing used during validation. FP32 is the reference. `--amp` enables FP16 only for backbone/context/neck and keeps the decoderFP32. Never use `model.half()`.

```bash
python tools/predict.py --config runs/my_dataset/config.yml \
  --checkpoint runs/my_dataset/training/best_stg1.pth \
  --classes runs/my_dataset/classes.json --image data/my_dataset/val/example.jpg \
  --amp --device cuda --threshold 0.3 --output results/example.json
```

Omit `--amp` for FP32; use `--device cpu` for CPU FP32 inference. This command writes JSON detections: `class_index`, original-image pixel `xyxy`, `score`, and optionally the original `category_id` and `name`. The confidence threshold filters output for application display; do not use a high threshold when measuring AP.

Input images are converted to RGB, resized directly to640×640 with PIL bilinear, scaled to[0,1] and normalized with mean[.485,.456,.406]/std[.229,.224,.225]. This is **not letterbox preprocessing**. The second input is original `[width,height]`, not `[height,width]`. Outputs are top300 ranked class/query pairs; no additional NMS is applied. Match these conventions in your application.

For batch1 PyTorch GPU accuracy/timing, `tools/evaluate.py --backend torch-fp32` or `torch-amp` uses the same path as prediction. TensorRT evaluation is documented in [EXPORT.md](EXPORT.md).

## Draw the recorded mascot result

```bash
python tools/draw_predictions.py --image assets/bear-input.jpg \
  --predictions assets/bear-inference.json --output results/bear-boxes.jpg
```

This draws actual stored predictions only. The illustration uses Base41 EMA CPU FP32; the image is AI-generated, not benchmark evidence.

## Numerical safety during inference

FP16 overflow can occur in the backbone or neck even in eval mode; inference
does not use GradScaler. The shared PyTorch inference wrapper now checks
features, decoder outputs and final predictions. With AMP enabled, a nonfinite
result retries the same image/batch in FP32 and retains FP32 for subsequent
calls. A warning and `precision_report` disclose the fallback; prediction JSON
and deployment-validation reports include it. Invalid inputs or invalid FP32
results still raise. Keep FP32 master weights and use autocast rather than
`model.half()`. Finite outputs alone do not establish accuracy: FP16
underflow/rounding can change scores without producing NaNs, so compare AP
against FP32 on representative data.

The fallback is eager PyTorch code. It is deliberately excluded from static
exports using `Inference.create(..., checked=False)`; checked wrappers refuse
tracing to prevent a misleading frozen fallback branch. TensorRT export now
explicitly protects the P5 projection as well as decoder/sensitive reductions
in FP32 and refuses a build if the P5 convolution scope cannot be located.
Existing engines must be rebuilt. TensorRT runtime rejects nonfinite predictions
and tells the caller to use the FP32 PyTorch backend; it has no automatic retry
inside the engine. Rebuilt engines require their own finite-output and AP
validation. The inference fix was tested on CUDA PyTorch, including deployed
Bear-L; a TensorRT engine has not been built or validated for this change.


Update2026-11-16: TensorRT has now been rebuilt and validated for the selected
CR7 Bear-L checkpoint, including full validation/test, captured-input finite
stress and batch1 timing. See [measured scope and results](../benchmarks/TENSORRT_REVALIDATION_20261116.md). This supersedes the earlier statement that no rebuilt engine had yet been validated; it does not certify other checkpoints.
