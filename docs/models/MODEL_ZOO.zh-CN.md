# 模型列表与实测结果

[English](MODEL_ZOO.md) | [简体中文](MODEL_ZOO.zh-CN.md) | [文档索引](../README.md)

**M 表示百万参数。** COCO val2017 全部5,000张，640、单尺度、EMA、PyTorch验证（Tiny主体FP16、解码器FP32）：

| 模型 | 输入 | AP50:95 | 参数 M（完整 / 部署） | GFLOPs | 配置 | 下载 MB（无损gzip） |
|---|---:|---:|---:|---:|---|---|
| Bear-T | 640 | 56.4597 | 52.05 / 51.10 | 132.18 | [T.yml](../../configs/T.yml) | [193.77](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-ema.pt.gz) |
| Bear-S | 640 | 57.8987 | 73.68 / 72.73 | 201.21 | [S.yml](../../configs/S.yml) | [274.17](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt.gz) |
| Bear-B | 640 | 59.0008 | 112.11 / 111.16 | 310.75 | [B.yml](../../configs/B.yml) | [416.91](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-ema.pt.gz) |
| Bear-L | 640 | 60.0728 | 221.40 / 220.45 | 622.32 | [L.yml](../../configs/L.yml) | [822.79](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-ema.pt.gz) |


2026-11-14：所有训练已结束，T/S/B/L最佳EMA均已发布。Tiny、全新Base和Large从匹配预训练骨干开始，均从第1轮使用完整FP32解码器；Small在第30轮后修复并恢复。Tiny/Base早停耐心为10，Large最终为20；Small手动停止。这不是骨干或训练精度的严格控制实验。DINOv3使用外部LVD-1689M预训练，检测监督为COCO train2017。

历史Base第41轮EMA（不是最终Base发布权重），在固定的1,000张 COCO验证子集上、RTX5090、batch1：

| 后端 | 子集 mAP50:95 | GPU平均延迟 | FPS |
|---|---:|---:|---:|
| PyTorch FP32 | 59.4238 | 19.4644ms | 51.38 |
| PyTorch FP16主体 + FP32解码器 | 59.4149 | 19.1562ms | 52.20 |
| TensorRT FP16主体 + FP32解码器 | 59.3778 | 4.0203ms | 248.74 |

子集AP不能当作全量COCO AP。计时无CUDA Graph，包含top300后处理，不含图像预处理与主机传输；TensorRT额外包含设备缓冲区拷贝。全量COCO TensorRT精度一致性、S/T部署速度尚未验证。[基准说明](../benchmarks/BENCHMARKS.zh-CN.md)。

结构细节见[架构指南](ARCHITECTURE.zh-CN.md)，下载见[权重页](WEIGHTS.zh-CN.md)，原始记录见[验证资料索引](../../validation/README.md)。

[计算量及复现](../benchmarks/COMPLEXITY.zh-CN.md) · [文件大小与参数关系](WEIGHT_SIZE.zh-CN.md)。
