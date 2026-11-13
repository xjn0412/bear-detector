# Pretrained weights

[English](WEIGHTS.md) | [简体中文](WEIGHTS.zh-CN.md)

**Default: FP16 backbone/context/neck storage with the entire original FP32 decoder.** All T/S/B/L mixed packages passed paired full-COCO and TRT checks. Original FP32 EMA archives remain available. The retired Small-only compact filename remains retired; all sizes now use uniform mixed filenames.

| Model | Selected epoch | Mixed gzip MB | Original FP32 gzip MB |
|---|---:|---|---|
| Tiny | 58 | [115.08](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt.gz) | [193.77](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-ema.pt.gz) |
| Small | 57 | [155.18](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-mixed.pt.gz) | [274.17](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt.gz) |
| Base | 42 | [226.34](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-mixed.pt.gz) | [416.91](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-ema.pt.gz) |
| Large | 24 | [428.79](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-mixed.pt.gz) | [822.79](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-ema.pt.gz) |

Architecture and parameter counts are unchanged. Gzip itself is lossless, but FP16 conversion rounds bulk weights; decoder tensor dtypes/values are preserved exactly. Packages omit optimizer/scaler resume states. Mixed packages load into FP32 models for inference or fresh fine-tuning; originals preserve unrounded initialization values for exact reproduction.

## Download and verify / 下载与校验

```bash
mkdir -p weights
curl -fL https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt.gz -o weights/bear-t-coco-mixed.pt.gz
echo "ec8ea1d16b446a48fb0831fa298e8ce2022fca357cbec4a6c569c20c4d04d17c  weights/bear-t-coco-mixed.pt.gz" | sha256sum -c -
gzip -dk weights/bear-t-coco-mixed.pt.gz
python tools/predict.py --config configs/T.yml --checkpoint weights/bear-t-coco-mixed.pt \
  --classes configs/coco-classes.json --image assets/bear-input.jpg \
  --device cuda --amp --output results/bear.json
```

For S/B/L, change the download, hash, config and checkpoint together. CPU: `--device cpu` without `--amp`. **Never call `model.half()`.** Fine-tuning preparation loads FP32 parameters and resets classifier heads/optimizer.

## Mixed-package SHA256

```text
ec8ea1d16b446a48fb0831fa298e8ce2022fca357cbec4a6c569c20c4d04d17c  bear-t-coco-mixed.pt.gz
dd0ad084051eef80d8cd84f916f19109839af1100bd5d15fb72487832cc070e7  bear-s-coco-mixed.pt.gz
4f1a6bdd96930fa331a1f87af6867c039ddd9fce00b42146540b7e4c616ac1e0  bear-b-coco-mixed.pt.gz
643f39d452b85452a747362b2afe667fb03e531cace1584fefb0ea7cf0082320  bear-l-coco-mixed.pt.gz
```

Each release SHA256SUMS also covers original archives, configs and validation reports. The mixed checkpoint records its conversion-time pending-validation flag; the final measured results are in deployment-validation.json for the exact same SHA. We did not repack metadata to change the tested checkpoint bytes. Paired full-COCO storage AP changes were approximately0.003–0.011 points; this is not a custom-task accuracy guarantee. [Full results](../benchmarks/DEPLOYMENT5090.md).

## Convert your own original EMA / 自行转换原始EMA

```bash
python tools/compact_weights.py --checkpoint weights/bear-t-coco-ema.pt \
  --output weights/bear-t-coco-mixed.pt
```

Requires an original FP32 EMA. New nonfinite bulk values are rejected; the known masked invalid-anchor Inf buffer in the decoder is preserved unchanged. Existing outputs are never overwritten. Validate converted weights on your own dataset.

[Fine-tuning](../guides/FINETUNE.md) · [Export](../guides/EXPORT.md). Upstream non-commercial and DINOv3 terms apply.
