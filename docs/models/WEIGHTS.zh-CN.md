# 预训练权重

[English](WEIGHTS.md) | [简体中文](WEIGHTS.zh-CN.md)

**默认选择混合存储：FP16骨干/上下文/颈部，整个解码器保留原始FP32。** 全部T/S/B/L已做成对全量COCO与TRT验证；原始FP32版本仍作为归档。旧Small单独的compact文件名已撤下，现在四种尺寸统一使用mixed文件名。

| Model | Selected epoch | Mixed .pt MB | Original FP32 .pt MB |
|---|---:|---|---|
| Tiny | 58 | [124.43](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt) | [208.86](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-ema.pt) |
| Small | 57 | [167.75](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-mixed.pt) | [295.46](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt) |
| Base | 42 | [244.60](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-mixed.pt) | [449.16](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-ema.pt) |
| Large | 24 | [463.18](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-mixed.pt) | [886.32](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-ema.pt) |

模型结构和参数量不变。下载直接提供.pt文件，无需解压。转换到FP16存储会舍入主体权重；解码器逐张量dtype/数值保持原样。包不含optimizer/scaler续训状态。混合包可加载到FP32模型用于推理或新微调；原始包保留未舍入的初始化值，适合精确复现。

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

其他尺寸同时替换下载、哈希、配置及检查点。CPU使用`--device cpu`并去掉`--amp`。**不要调用`model.half()`。** 训练准备会加载到FP32参数，重建类别头和优化器。

## Mixed-package SHA256

```text
ac51e9e68bc141a9782005589d56df574ba285def88857ac2c4275413d782150  bear-t-coco-mixed.pt
afc119b4b8a4aa4c0d82dbf9f7a4d929c007b7f161a404c9a19f354568e8bf28  bear-s-coco-mixed.pt
c89b634cd61686a8df589df7d0c4bcccc6003ee18bf1304e00ca5622ccfd72c6  bear-b-coco-mixed.pt
e096089a14152fd17e610a2ad2f407db52eeea77d66375e02933c43fd720694b  bear-l-coco-mixed.pt
```

每个release的`SHA256SUMS`也包含原始归档、配置和验证报告。混合检查点内部记录转换时的待验证标记；最终实测结果以同一SHA对应的`deployment-validation.json`为准，未通过改写元数据来改变已测权重文件。各模型存储舍入的全量COCO AP变化约0.003–0.011点；这不是自定义任务精度保证。[完整结果](../benchmarks/DEPLOYMENT5090.zh-CN.md)。

## Convert your own original EMA / 自行转换原始EMA

```bash
python tools/compact_weights.py --checkpoint weights/bear-t-coco-ema.pt \
  --output weights/bear-t-coco-mixed.pt
```

仅接受原始FP32 EMA；若主体转换产生非有限值则拒绝，解码器的已知无效anchor Inf哨兵原样保留。输出不会覆盖已有文件；转换后必须验证自己的数据集。

[Fine-tuning](../guides/FINETUNE.md) · [Export](../guides/EXPORT.md). Upstream non-commercial and DINOv3 terms apply.
