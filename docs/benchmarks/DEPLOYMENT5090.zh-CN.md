# RTX5090部署验证与延迟

[English](DEPLOYMENT5090.md) | [简体中文](DEPLOYMENT5090.zh-CN.md)

2026-11-14：对最终T/S/B/L最佳EMA进行全量COCO与实际GPU测试，无新增训练。推荐混合存储、完整FP32解码器；原始FP32版本仍作归档。

| 模型 | 输入 | AP50:95（PT / TRT） | 参数 M（完整 / 部署） | GFLOPs | PyTorch AMP ms | TensorRT ms | 混合.pt MB |
|---|---:|---:|---:|---:|---:|---:|---|
| [Bear-T](../../configs/T.yml) | 640 | 56.4851 / 56.4449 | 52.05 / 51.10 | 132.18 | 12.84 | 2.75 | [124.43](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt) |
| [Bear-S](../../configs/S.yml) | 640 | 57.8864 / 57.8563 | 73.68 / 72.73 | 201.21 | 16.44 | 3.34 | [167.75](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-mixed.pt) |
| [Bear-B](../../configs/B.yml) | 640 | 59.0192 / 58.9243 | 112.11 / 111.16 | 310.75 | 15.69 | 3.82 | [244.60](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-mixed.pt) |
| [Bear-L](../../configs/L.yml) | 640 | 60.0748 / 60.0236 | 221.40 / 220.45 | 622.32 | 15.20 | 4.79 | [463.18](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-mixed.pt) |

## Latency and resident memory / 延迟与驻留显存

| Model | Backend | Mean ms | P95 ms | FPS | Round-mean CV % | Process resident MiB |
|---|---|---:|---:|---:|---:|---:|
| Bear-T | PyTorch FP32 original | 14.4467 | 14.6240 | 69.22 | 0.321 | 1018 |
| Bear-T | PyTorch AMP mixed | 12.8412 | 13.1246 | 77.87 | 0.874 | 1004 |
| Bear-T | TensorRT mixed | 2.7513 | 2.7625 | 363.46 | 0.174 | 830 |
| Bear-S | PyTorch FP32 original | 17.0492 | 17.1206 | 58.65 | 0.046 | 1100 |
| Bear-S | PyTorch AMP mixed | 16.4438 | 16.4917 | 60.81 | 0.123 | 1086 |
| Bear-S | TensorRT mixed | 3.3376 | 3.3465 | 299.62 | 0.010 | 874 |
| Bear-B | PyTorch FP32 original | 19.1829 | 19.2441 | 52.13 | 0.022 | 1338 |
| Bear-B | PyTorch AMP mixed | 15.6863 | 15.8985 | 63.75 | 0.907 | 1264 |
| Bear-B | TensorRT mixed | 3.8242 | 3.8334 | 261.50 | 0.010 | 960 |
| Bear-L | PyTorch FP32 original | 26.3672 | 26.4625 | 37.93 | 0.114 | 1866 |
| Bear-L | PyTorch AMP mixed | 15.2012 | 15.3345 | 65.78 | 0.935 | 1774 |
| Bear-L | TensorRT mixed | 4.7892 | 4.8058 | 208.80 | 0.061 | 1204 |

单张NVIDIA GeForce RTX5090 GPU7，驱动615.71.09，PyTorch2.14.1+cu130，TensorRT10.13.3.9，Python3.11.18。575W功率上限，未设置新频率或重置时钟。batch1×3×640×640，80类，100预热、3轮各300次CUDA Event，无CUDA Graph、TF32关闭。输入为同一真实COCO图像，PIL双线性直接缩放、ImageNet归一化，输出top300，无NMS。所有挖矿、训练及其他GPU构建在测速前停止。T/S引擎在GPU7构建，B/L在同型号GPU5/6并行构建；构建全部结束、GPU清空后恢复GPU7串行测速。

计时从GPU输入到GPU预测，包括top300；排除文件读取、预处理、主机传输和COCO评测，TRT包含设备输入拷贝。PyTorch启用cuDNN benchmark并关闭MHA fastpath，与可追踪导出路径保持一致。这里是eager基线，不是PyTorch compile/CUDA Graph的最大吞吐。较大模型不必延迟单调增加：kernel选择、通道对齐及CPU启动开销都会影响结果。

显存是独立进程完成预热与测速后的nvidia-smi进程驻留量，**不是连续采样的峰值或训练显存**。原始JSON另有PyTorch管理的峰值分配；TRT库的全部工作区并不计入该PyTorch数值，不能把二者直接比较。混合存储减少下载大小，不会因加载到FP32参数后自动将PyTorch显存减半。

## Full COCO accuracy / 全量COCO精度

| Model | Original-storage PT AMP | Mixed-storage PT AMP | Storage delta | Mixed TRT | TRT−mixed PT |
|---|---:|---:|---:|---:|---:|
| Bear-T | 56.47438 | 56.48514 | +0.01076 | 56.44489 | -0.04025 |
| Bear-S | 57.88084 | 57.88645 | +0.00561 | 57.85629 | -0.03015 |
| Bear-B | 59.01324 | 59.01919 | +0.00594 | 58.92433 | -0.09486 |
| Bear-L | 60.07185 | 60.07484 | +0.00299 | 60.02360 | -0.05124 |

全部5,000张COCO val2017、固定640，官方原类别ID；标准COCO box AP50:95，faster-coco-eval，maxDets100汇总（每图输入top300预测）。同一图片与预处理，有限值检查全部通过。PyTorch两种存储都加载到FP32模型后AMP执行，batch8；TRT固定batch1，因此后端差异还包含batch相关数值变化。存储比较门限和TRT部署门限均为最多下降0.1 AP点，所有模型通过。这里不是逐元素完全一致保证，也不保证其他数据集、精度或硬件。小幅正负变化不代表训练提升；历史epoch AP记录不改写。

## Reproduce / 复现

Use the [tested installation](../getting-started/INSTALL.md), matching config/classes and [weights](../models/WEIGHTS.md). Commands run from the repository root on an otherwise idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_engine.py --config configs/T.yml \
  --checkpoint weights/bear-t-coco-mixed.pt --output exports/T --build
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_latency.py --backend torch-amp \
  --config configs/T.yml --checkpoint weights/bear-t-coco-mixed.pt \
  --image assets/bear-input.jpg --output results/T-torch.json
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_latency.py --backend tensorrt \
  --config configs/T.yml --checkpoint weights/bear-t-coco-mixed.pt \
  --engine exports/T/model.engine --image assets/bear-input.jpg --output results/T-trt.json
CUDA_VISIBLE_DEVICES=0 python tools/validate_deployment.py --backend tensorrt \
  --config configs/T.yml --checkpoint weights/bear-t-coco-mixed.pt \
  --engine exports/T/model.engine --annotations data/coco/annotations/instances_val2017.json \
  --images data/coco/val2017 --output results/T-trt-accuracy.json
```

The sample image above is a convenient user example; recorded timings used the COCO image hash in raw results. Engines must be rebuilt on the target hardware/software; they are not published as portable downloads. The tested exporter uses TensorRT10 weak-typing constraints, not TensorRT11 strong typing.

[Raw per-model JSON, configuration/checkpoint/engine/input hashes](../../validation/benchmarks/deployment5090-20261114/README.md) · [Storage policy](../models/WEIGHTS.md).
