# PyTorch inference

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
