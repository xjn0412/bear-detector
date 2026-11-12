# 模型列表与实测结果

[English](MODEL_ZOO.md) | [简体中文](MODEL_ZOO.zh-CN.md) | [文档索引](README.md)

**M 表示百万参数。** COCO val2017 全部5,000张，640、单尺度、EMA、PyTorch FP32验证：

| 版本 | 总参数 | 骨干参数 | 选定轮次 | mAP50:95 |
|---|---:|---:|---:|---:|
| Bear-B | 112.11M | 87.56M | 41 | 58.9306 |
| Bear-S | 73.68M | 49.45M | 57 | 57.8987 |
| Bear-T | 52.05M | 27.82M | — | 尚未测得 |

2026-11-12：此前B/S训练由作者停止，分别完成56/68轮；目前T继续训练，并开展新的Base重新训练。表中Base结果来自之前运行。不是完整、无中断的对照实验。B在第20轮后、S在第30轮后修复数值问题并从完整检查点恢复。没有达到65AP的主张。检测监督仅用COCO train2017，但DINOv3骨干使用外部LVD-1689M预训练，因此不是所有预训练数据都仅来自COCO，也不是COCO零样本检测。

相同 Base 第41轮 EMA 权重，在固定的1,000张 COCO验证子集上、RTX5090、batch1：

| 后端 | 子集 mAP50:95 | GPU平均延迟 | FPS |
|---|---:|---:|---:|
| PyTorch FP32 | 59.4238 | 19.4644ms | 51.38 |
| PyTorch FP16主体 + FP32解码器 | 59.4149 | 19.1562ms | 52.20 |
| TensorRT FP16主体 + FP32解码器 | 59.3778 | 4.0203ms | 248.74 |

子集AP不能当作全量COCO AP。计时无CUDA Graph，包含top300后处理，不含图像预处理与主机传输；TensorRT额外包含设备缓冲区拷贝。全量COCO TensorRT精度一致性、S/T部署速度尚未验证。[基准说明](BENCHMARKS.zh-CN.md)。

结构细节见[架构指南](ARCHITECTURE.zh-CN.md)，下载见[权重页](WEIGHTS.zh-CN.md)，原始记录见[验证资料索引](../validation/README.md)。
