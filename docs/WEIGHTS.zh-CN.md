# 权重下载

[English](WEIGHTS.md) | [简体中文](WEIGHTS.zh-CN.md)

[Small发布页、manifest与SHA256SUMS](https://github.com/xjn0412/bear-detector/releases/tag/v0.1.0-weights)。仓库和权重已公开，网页/curl下载无需登录。两个包都包含第57轮选定EMA张量与元数据，不含优化器/scaler完整续训状态；支持推理与新任务微调。

| Small包 | 下载大小 | 全量COCO FP32 AP50:95 | 全量COCO AMP AP50:95 |
|---|---:|---:|---:|
| [紧凑包](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-compact.pt.gz) | 155.18 MB | 57.89035 | 57.87800 |
| [无损参考包](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt.gz) | 274.17 MB | 57.89617 | 57.88758 |

十进制MB，1 MB=1,000,000字节。未压缩Small FP32文件295.46 MB，紧凑下载比它小47.5%，比无损gzip小43.4%。压缩不改变参数量或FLOPs，也不会自动提高速度。

## 精度与存储策略

全量COCO val2017 5,000张，640，batch2。紧凑包FP32下降**0.00581 AP点**，AMP下降**0.00959点**，均通过预设0.1点阈值。所有输出有限。[完整指标与评测文件哈希](../validation/S_compact_coco.json)。参考包此次复测比历史训练验证57.89868低0.00251点；压缩对比应使用上表同条件复测结果。

紧凑包主体以FP16存储，解码器保留原始FP32。使用仓库工具加载到普通**FP32模型**，不要调用`model.half()`。AMP推理仍保持整个解码器FP32。加载回FP32不能恢复已舍入的信息；无损包解压后与原文件字节完全一致。

发布包仅增加验证元数据，逐张量核对键、dtype与值完全相同。[Manifest](../validation/S_release_manifest.json)保留评测文件和发布文件哈希对应。gzip往返字节一致，上传后核对GitHub资产SHA256。**紧凑权重TensorRT精度尚未验证**，需要重新构建引擎并验证；旧引擎包含不同权重。自定义任务也需单独验证。本次与训练并行，不用于速度比较。

## 下载与使用

点击发布页下载，或使用免登录curl下载：

```bash
curl -fLO https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-compact.pt.gz
curl -fLO https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/SHA256SUMS
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('bear-s-coco-compact.pt.gz','rb'),open('bear-s-coco-compact.pt','wb'))"
python tools/predict.py --config configs/S.yml --checkpoint bear-s-coco-compact.pt \
  --classes configs/coco-classes.json --image assets/bear-input.jpg \
  --device cuda --amp --output results/bear.json
```

解压前核对SHA256（见英文页或发布页SHA256SUMS）。无损包替换为`bear-s-coco-ema.pt(.gz)`。CPU用`--device cpu`，去掉`--amp`。安装、COCO/YOLO任意类别微调、导出方法见其他指南。

本地全量复测命令：

```bash
python tools/validate_compact.py --config configs/S.yml \
  --reference bear-s-coco-ema.pt --candidate bear-s-coco-compact.pt \
  --annotations data/coco/annotations/instances_val2017.json \
  --images data/coco/val2017 --output results/compact-comparison.json
```

旧Base检查点因源服务器故障暂时无法访问；Base重新训练正在准备/启动中，未上传未经验证的Base权重。Tiny正在训练，尚无发布权重。请遵循DEIMv2非商业许可与DINOv3单独条款。
