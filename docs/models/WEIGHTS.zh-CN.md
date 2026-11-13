# 预训练权重

[English](WEIGHTS.md) | [简体中文](WEIGHTS.zh-CN.md)

**默认选择混合存储：FP16骨干/上下文/颈部，整个解码器保留原始FP32。** 全部T/S/B/L已做成对全量COCO与TRT验证；原始FP32版本仍作为归档。旧Small单独的compact文件名已撤下，现在四种尺寸统一使用mixed文件名。

| Model | Selected epoch | Mixed gzip MB | Original FP32 gzip MB |
|---|---:|---|---|
| Tiny | 58 | [115.08](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt.gz) | [193.77](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-ema.pt.gz) |
| Small | 57 | [155.18](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-mixed.pt.gz) | [274.17](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt.gz) |
| Base | 42 | [226.34](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-mixed.pt.gz) | [416.91](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-ema.pt.gz) |
| Large | 24 | [428.79](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-mixed.pt.gz) | [822.79](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-ema.pt.gz) |

模型结构和参数量不变。gzip压缩无损，但转换到FP16存储会舍入主体权重；解码器逐张量dtype/数值保持原样。包不含optimizer/scaler续训状态。混合包可加载到FP32模型用于推理或新微调；原始包保留未舍入的初始化值，适合精确复现。

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

其他尺寸同时替换下载、哈希、配置及检查点。CPU使用`--device cpu`并去掉`--amp`。**不要调用`model.half()`。** 训练准备会加载到FP32参数，重建类别头和优化器。

## Mixed-package SHA256

```text
ec8ea1d16b446a48fb0831fa298e8ce2022fca357cbec4a6c569c20c4d04d17c  bear-t-coco-mixed.pt.gz
dd0ad084051eef80d8cd84f916f19109839af1100bd5d15fb72487832cc070e7  bear-s-coco-mixed.pt.gz
4f1a6bdd96930fa331a1f87af6867c039ddd9fce00b42146540b7e4c616ac1e0  bear-b-coco-mixed.pt.gz
643f39d452b85452a747362b2afe667fb03e531cace1584fefb0ea7cf0082320  bear-l-coco-mixed.pt.gz
```

每个release的`SHA256SUMS`也包含原始归档、配置和验证报告。混合检查点内部记录转换时的待验证标记；最终实测结果以同一SHA对应的`deployment-validation.json`为准，未通过改写元数据来改变已测权重文件。各模型存储舍入的全量COCO AP变化约0.003–0.011点；这不是自定义任务精度保证。[完整结果](../benchmarks/DEPLOYMENT5090.zh-CN.md)。

## Convert your own original EMA / 自行转换原始EMA

```bash
python tools/compact_weights.py --checkpoint weights/bear-t-coco-ema.pt \
  --output weights/bear-t-coco-mixed.pt
```

仅接受原始FP32 EMA；若主体转换产生非有限值则拒绝，解码器的已知无效anchor Inf哨兵原样保留。输出不会覆盖已有文件；转换后必须验证自己的数据集。

[Fine-tuning](../guides/FINETUNE.md) · [Export](../guides/EXPORT.md). Upstream non-commercial and DINOv3 terms apply.
