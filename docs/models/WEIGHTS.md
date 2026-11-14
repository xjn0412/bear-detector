# Pretrained weights

[English](WEIGHTS.md) | [简体中文](WEIGHTS.zh-CN.md)

**Default: FP16 backbone/context/neck storage with the entire original FP32 decoder.** All T/S/B/L mixed packages passed paired full-COCO and TRT checks. Original FP32 EMA archives remain available. The retired Small-only compact filename remains retired; all sizes now use uniform mixed filenames.

| Model | Selected epoch | Mixed .pt MB | Original FP32 .pt MB |
|---|---:|---|---|
| Tiny | 58 | [124.43](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt) | [208.86](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-ema.pt) |
| Small | 57 | [167.75](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-mixed.pt) | [295.46](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt) |
| Base | 42 | [244.60](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-mixed.pt) | [449.16](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-ema.pt) |
| Large | 24 | [463.18](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-mixed.pt) | [886.32](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-ema.pt) |

Architecture and parameter counts are unchanged. Downloads are plain .pt checkpoints and need no decompression. FP16 conversion rounds bulk weights; decoder tensor dtypes/values are preserved exactly. Packages omit optimizer/scaler resume states. Mixed packages load into FP32 models for inference or fresh fine-tuning; originals preserve unrounded initialization values for exact reproduction.

Direct `.pt` files load without an extraction step. / `.pt`文件可直接加载，无需解压。

## Download and verify / 下载与校验

```bash
mkdir -p weights
curl -fL https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt -o weights/bear-t-coco-mixed.pt
echo "ac51e9e68bc141a9782005589d56df574ba285def88857ac2c4275413d782150  weights/bear-t-coco-mixed.pt" | sha256sum -c -
python tools/predict.py --config configs/T.yml --checkpoint weights/bear-t-coco-mixed.pt \
  --classes configs/coco-classes.json --image assets/bear-input.jpg \
  --device cuda --amp --output results/bear.json
```

For S/B/L, change the download, hash, config and checkpoint together. CPU: `--device cpu` without `--amp`. **Never call `model.half()`.** Fine-tuning preparation loads FP32 parameters and resets classifier heads/optimizer.

## Mixed-package SHA256

```text
ac51e9e68bc141a9782005589d56df574ba285def88857ac2c4275413d782150  bear-t-coco-mixed.pt
afc119b4b8a4aa4c0d82dbf9f7a4d929c007b7f161a404c9a19f354568e8bf28  bear-s-coco-mixed.pt
c89b634cd61686a8df589df7d0c4bcccc6003ee18bf1304e00ca5622ccfd72c6  bear-b-coco-mixed.pt
e096089a14152fd17e610a2ad2f407db52eeea77d66375e02933c43fd720694b  bear-l-coco-mixed.pt
```

Each release SHA256SUMS also covers original archives, configs and validation reports. The mixed checkpoint records its conversion-time pending-validation flag; the final measured results are in deployment-validation.json for the exact same SHA. We did not repack metadata to change the tested checkpoint bytes. Paired full-COCO storage AP changes were approximately0.003–0.011 points; this is not a custom-task accuracy guarantee. [Full results](../benchmarks/DEPLOYMENT5090.md).

## Convert your own original EMA / 自行转换原始EMA

```bash
python tools/compact_weights.py --checkpoint weights/bear-t-coco-ema.pt \
  --output weights/bear-t-coco-mixed.pt
```

Requires an original FP32 EMA. New nonfinite bulk values are rejected; the known masked invalid-anchor Inf buffer in the decoder is preserved unchanged. Existing outputs are never overwritten. Validate converted weights on your own dataset.

[Fine-tuning](../guides/FINETUNE.md) · [Export](../guides/EXPORT.md). Upstream non-commercial and DINOv3 terms apply.
