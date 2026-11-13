# Pretrained weights

[English](WEIGHTS.md) | [简体中文](WEIGHTS.zh-CN.md)

All models ship **original FP32 EMA tensors in lossless gzip**, with no quantization or storage rounding. The Small compact download has been retired; historical compression evidence is retained under validation. Packages support inference and fresh fine-tuning; they do not include full optimizer/scaler resume states.

| Model | Parameters | Selected epoch | Training AP50:95 | Download (MB) | Config |
|---|---:|---:|---:|---|---|
| Tiny | 52.05M | 58 | 56.4597 | [193.77](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-ema.pt.gz) | [configs/T.yml](../../configs/T.yml) |
| Small | 73.68M | 57 | 57.8987 | [274.17](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt.gz) | [configs/S.yml](../../configs/S.yml) |
| Base | 112.11M | 42 | 59.0008 | [416.91](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-ema.pt.gz) | [configs/B.yml](../../configs/B.yml) |
| Large | 221.40M | 24 | 60.0728 | [822.79](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-ema.pt.gz) | [configs/L.yml](../../configs/L.yml) |

MB = 1,000,000 bytes. AP is full COCO val2017 at 640 with selected EMA weights. Small also has a separate reference rerun of 57.8962 AP; this is not a different training checkpoint.

## Download, verify and predict / 下载、校验与预测

Run from the repository root after [installation](../getting-started/INSTALL.md):

```bash
mkdir -p weights
curl -fL https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-ema.pt.gz -o weights/bear-t-coco-ema.pt.gz
echo "386bd76c044b969947b887054678f345e2a06b9b4191d9adae4fdb5d9ff7433c  weights/bear-t-coco-ema.pt.gz" | sha256sum -c -
gzip -dk weights/bear-t-coco-ema.pt.gz
python tools/predict.py --config configs/T.yml --checkpoint weights/bear-t-coco-ema.pt \
  --classes configs/coco-classes.json --image assets/bear-input.jpg \
  --device cuda --amp --output results/bear.json
```

For S/B/L, use the matching download, hash, config and checkpoint name. CPU: `--device cpu` without `--amp`. Load into an FP32 model; AMP keeps the **entire decoder FP32**. Do not call `model.half()`. / 更换尺寸时须同时更换下载、哈希、配置及文件名。CPU去掉`--amp`，使用`--device cpu`。不要将整个模型转为half。

## SHA256

```text
386bd76c044b969947b887054678f345e2a06b9b4191d9adae4fdb5d9ff7433c  bear-t-coco-ema.pt.gz
e2b7a9565a2ea9e646e070dc6d1f59b70a675f390dd2958f38218f465cb9710d  bear-s-coco-ema.pt.gz
2b984258c048631d1d7c460a9989a6c8c294b224ff25d6d7787d996815a58d36  bear-b-coco-ema.pt.gz
58303e467260ccae3634bd80e6f4efa8702f27ac6f781dceac550eb136bccf0b  bear-l-coco-ema.pt.gz
```

Public downloads require no login. Gzip preserves the source checkpoint bytes; it reduces download/storage size, not model parameters or inference compute. Export a new engine from your selected weights and validate accuracy on your own dataset; historical TensorRT results do not validate these final exports. / 公开下载无需登录；无损压缩不降低参数量或计算量。最终权重导出后仍须验证部署精度。

[Fine-tuning: COCO and YOLO annotations](../guides/FINETUNE.md) · [Export](../guides/EXPORT.md) · [Model zoo](MODEL_ZOO.md). Upstream non-commercial and DINOv3 terms apply.
