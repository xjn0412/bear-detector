# 模型列表与实测结果

[English](MODEL_ZOO.md) | [简体中文](MODEL_ZOO.zh-CN.md) | [文档索引](../README.md)

**M 表示百万参数。** COCO val2017 全部5,000张，640、单尺度、EMA、PyTorch验证（Tiny主体FP16、解码器FP32）：

| 模型 | 输入 | AP50:95（PT / TRT） | 参数 M（完整 / 部署） | GFLOPs | PyTorch AMP ms | TensorRT ms | 混合包 MB |
|---|---:|---:|---:|---:|---:|---:|---|
| [Bear-T](../../configs/T.yml) | 640 | 56.4851 / 56.4449 | 52.05 / 51.10 | 132.18 | 12.84 | 2.75 | [115.08](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt.gz) |
| [Bear-S](../../configs/S.yml) | 640 | 57.8864 / 57.8563 | 73.68 / 72.73 | 201.21 | 16.44 | 3.34 | [155.18](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-mixed.pt.gz) |
| [Bear-B](../../configs/B.yml) | 640 | 59.0192 / 58.9243 | 112.11 / 111.16 | 310.75 | 15.69 | 3.82 | [226.34](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-mixed.pt.gz) |
| [Bear-L](../../configs/L.yml) | 640 | 60.0748 / 60.0236 | 221.40 / 220.45 | 622.32 | 15.20 | 4.79 | [428.79](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-mixed.pt.gz) |



2026-11-14：所有训练已结束，T/S/B/L最佳EMA均已发布。Tiny、全新Base和Large从匹配预训练骨干开始，均从第1轮使用完整FP32解码器；Small在第30轮后修复并恢复。Tiny/Base早停耐心为10，Large最终为20；Small手动停止。这不是骨干或训练精度的严格控制实验。DINOv3使用外部LVD-1689M预训练，检测监督为COCO train2017。

历史Base第41轮EMA（不是最终Base发布权重），在固定的1,000张 COCO验证子集上、RTX5090、batch1：

| 后端 | 子集 mAP50:95 | GPU平均延迟 | FPS |
|---|---:|---:|---:|
| PyTorch FP32 | 59.4238 | 19.4644ms | 51.38 |
| PyTorch FP16主体 + FP32解码器 | 59.4149 | 19.1562ms | 52.20 |
| TensorRT FP16主体 + FP32解码器 | 59.3778 | 4.0203ms | 248.74 |

子集AP不能当作全量COCO AP。计时无CUDA Graph，包含top300后处理，不含图像预处理与主机传输；TensorRT额外包含设备缓冲区拷贝。旧子集测试不代表最终模型；当前已完成全量COCO与四个尺寸测速，见下方最新部署结果。[基准说明](../benchmarks/BENCHMARKS.zh-CN.md)。

结构细节见[架构指南](ARCHITECTURE.zh-CN.md)，下载见[权重页](WEIGHTS.zh-CN.md)，原始记录见[验证资料索引](../../validation/README.md)。

[计算量及复现](../benchmarks/COMPLEXITY.zh-CN.md) · [文件大小与参数关系](WEIGHT_SIZE.zh-CN.md)。

## 当前发布部署结果

上表AP是本次混合权重的新验证结果；原始训练逐轮AP不变。新PyTorch/TRT延迟与显存在[部署基准](../benchmarks/DEPLOYMENT5090.zh-CN.md)，历史Base41子集结果仅作历史对照，不能替代当前全量验证。
