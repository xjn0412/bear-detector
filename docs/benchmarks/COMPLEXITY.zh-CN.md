# 计算量与性能表口径

[English](COMPLEXITY.md) | [简体中文](COMPLEXITY.zh-CN.md)

2026-11-14实测，固定batch1×3×640×640，80类。所有测试在CPU上进行，未停止挖矿、未恢复训练或重新构建TensorRT。

| 模型 | 输入 | AP50:95 | 参数 M（完整 / 部署） | GFLOPs | 配置 | 下载 MB（.pt） |
|---|---:|---:|---:|---:|---|---|
| Bear-T | 640 | 56.4597 | 52.05 / 51.10 | 132.18 | [T.yml](../../configs/T.yml) | [208.86](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-ema.pt) |
| Bear-S | 640 | 57.8987 | 73.68 / 72.73 | 201.21 | [S.yml](../../configs/S.yml) | [295.46](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt) |
| Bear-B | 640 | 59.0008 | 112.11 / 111.16 | 310.75 | [B.yml](../../configs/B.yml) | [449.16](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-ema.pt) |
| Bear-L | 640 | 60.0728 | 221.40 / 220.45 | 622.32 | [L.yml](../../configs/L.yml) | [886.32](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-ema.pt) |

## Counting scope / 统计范围

使用PyTorch2.14.1的`FlopCounterMode`，种子42初始化、FP32、eval→deploy完整前向，不用训练权重或数据集。参数值不影响这里的算子形状与算术计数，不能据此宣称精度。主体的原始多分支卷积已融合；未使用的中间分类头等被`deploy()`删除，完整参数不能与YOLO部署参数直接混用。

每个卷积/线性/矩阵乘加计2 FLOPs。SDPA强制数学分解，明确计入QKᵀ和AV，避免漏算CPU flash attention；变形注意力双线性采样每输出元素计7次算术（4乘3加）。采样坐标计算、激活、归一化、softmax、残差逐元素运算、索引/top-k、图像预处理、访存、损失及反向传播不计入。表中GFLOPs因此是**明确范围的算术估计，不是所有硬件操作的精确总量**。

不能在32×32上计数再按面积放大：查询数量、top-k及全局注意力不一定按面积线性缩放。这里实际运行640输入。CPU前向保持FP32以便统计；部署仍保持FP16主体+整个FP32解码器，不改变模型的运算形状。FLOPs不是延迟，采样、精度和访存都会影响速度。

## Breakdown / 组成

| Model | Backbone + context/detail GFLOPs | Neck GFLOPs | Decoder GFLOPs | Context alone GFLOPs |
|---|---:|---:|---:|---:|
| Bear-T | 77.69 | 47.73 | 6.76 | 3.33 |
| Bear-S | 146.72 | 47.73 | 6.76 | 3.33 |
| Bear-B | 256.26 | 47.73 | 6.76 | 3.49 |
| Bear-L | 567.83 | 47.73 | 6.76 | 3.80 |

上下文列已包含在骨干列，不能再次相加。Tiny的颈部占36.1%算术，单独上下文仅2.5%；下一步提速可优先验证颈部宽度/深度，而不能假定删除注意力必然带来大幅提速。Base/Large主要由CNN骨干计算主导。这里只提出优化方向，没有修改架构、训练或权重。

## Reproduce / 复现

Run from the repository root with the documented PyTorch environment:

```bash
python tools/profile_model.py --config configs/T.yml --output results/T-profile.json
python tools/profile_model.py --config configs/S.yml --output results/S-profile.json
python tools/profile_model.py --config configs/B.yml --output results/B-profile.json
python tools/profile_model.py --config configs/L.yml --output results/L-profile.json
```

[Raw profiles and storage audits](../../validation/benchmarks/complexity640/README.md) include config hashes, counts by module/operator and exact exclusions. The tool requires PyTorch with `torch.utils.flop_counter` and math SDPA (tested2.14.1). It deliberately does not advertise CPU profiling time as inference latency.

## Reference / 参考

[YOLO26 performance](https://docs.ultralytics.com/models/yolo26/#performance-metrics) reports fused/deployed parameters and two FLOPs per MAC via[Ultralytics profiling code](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py). Its FLOPs use THOP, not this counter; omitted operations can differ. Its T4 TensorRT speed is not comparable directly with historical Bear RTX5090 speed. [Existing same-checkpoint backend timings](BENCHMARKS.md).

[New final-model measured latency and full COCO deployment checks](DEPLOYMENT5090.md); no FLOPs recount was needed because architecture is unchanged.
