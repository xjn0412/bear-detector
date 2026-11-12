# Weight files and smaller downloads

[English](WEIGHTS.md) | [简体中文](WEIGHTS.zh-CN.md)

The B/S download packages contain EMA model tensors and metadata only; full training checkpoints with optimizer/scaler/EMA history remain separately preserved. The release assets are prepared but **not yet uploaded**. No working GitHub weight-download link is claimed here.

Sizes below are measured decimal MB (1 MB = 1,000,000 bytes), not parameter counts.

| Model | FP32 EMA `.pt` | Lossless `.pt.gz` | Compact `.pt` | Compact `.pt.gz` |
|---|---:|---:|---:|---:|
| Base, selected epoch41 | 449.16 MB | 416.91 MB | 244.61 MB | 226.35 MB |
| Small, selected epoch57 | 295.46 MB | 274.17 MB | 167.76 MB | 155.18 MB |

- **Lossless:** decompression reproduces the original checkpoint byte for byte; SHA256 round-trip checks passed. It preserves the selected checkpoint’s measured accuracy.
- **Compact candidate:** stores FP16 backbone/context/neck tensors and unchanged FP32 decoder tensors. It reduces download size by about 49.6% (B) / 47.5% (S), including gzip. Decoder tensors are verified bit-identical. The other weights are rounded: this is not lossless, and the original checkpoint AP must not be assigned to this variant. Full COCO mAP and TensorRT parity are pending.
- Both formats retain all model tensors needed for fresh fine-tuning. Neither is a full optimizer/scaler training-resume checkpoint.

After downloading a `.pt.gz` file, decompress it before passing it to the existing tools:

```bash
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('bear-b-coco-ema.pt.gz','rb'),open('bear-b-coco-ema.pt','wb'))"
python tools/predict.py --config configs/B.yml --checkpoint bear-b-coco-ema.pt \
  --image assets/bear-input.jpg --device cpu --output results/bear.json
```

For a compact candidate, substitute its filename. Load into the ordinary **FP32 model** through the repository tools. FP16 storage does not mean `model.half()` or an FP16 decoder: the runtime policy remains FP16 autocast for backbone/context/neck and an FP32 decoder. Restoring stored FP16 tensors into FP32 parameters does not undo their rounding.

Compression changes transfer size, not parameter count or FLOPs; it does not itself make inference faster. A smaller architecture (Tiny) or separately calibrated quantization is needed for further runtime reductions. Original selected B/S FP32 EMA checkpoints remain the accuracy reference. Tiny is training and has no released trained checkpoint yet.

Published assets will need checksums, matching configs/class mapping, the selected epoch and measured AP, plus the retained upstream license terms. Build TensorRT engines locally and revalidate the selected weights; do not distribute an older engine as if it contains these weights.
