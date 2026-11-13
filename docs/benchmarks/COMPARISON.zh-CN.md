# 与相近检测器的比较

[English](COMPARISON.md) | [简体中文](COMPARISON.zh-CN.md)

资料日期：2026-11-14。其他模型使用**作者发布的结果，本仓库没有重新测试**。表中均为640输入、COCO val2017上的框AP50:95。预训练数据、训练周期、参数统计口径及推理后端不同，不能视为严格的架构消融实验。

| 模型 | 参数量 | COCO AP50:95 | 结果与训练说明 |
|---|---:|---:|---|
| Bear-T | 52.05M | 56.4597 | Full COCO validation; EMA epoch58 |
| Bear-L | 221.40M | 60.0728 | Full COCO validation; EMA epoch24 |
| Bear-B | 112.11M | 59.0008 | 本项目实测EMA第42轮；DINOv3 ConvNeXt-B外部预训练；COCO检测监督 |
| Bear-S | 73.68M | 57.90 | 本项目实测EMA第57轮；DINOv3 ConvNeXt-S外部预训练；COCO检测监督 |
| DEIMv2-L | 32.2M | 56.0 | 官方模型库，DINOv3检测器 |
| DEIMv2-X | 50.3M | 57.8 | 官方模型库，DINOv3检测器 |
| D-FINE-L | 31M | 54.0 | 官方COCO结果 |
| D-FINE-X | 62M | 55.8 | 官方COCO结果 |
| D-FINE-X，Objects365→COCO | 62M | 59.3 | 使用额外检测预训练，单独列出 |
| RT-DETRv2-L | 42M | 53.4 | 官方COCO结果 |
| RT-DETRv2-X | 76M | 54.3 | 官方COCO结果 |
| YOLO26-L，默认检测头 | 24.8M | 55.0 | 官方融合部署模型参数量；AP使用NMS检测头 |
| YOLO26-X，默认检测头 | 55.7M | 57.5 | 官方融合部署模型参数量；AP使用NMS检测头 |

YOLO26的无NMS检测头AP分别为L54.4、X56.9，官方速度对应无NMS路径；发布配方包含Objects365预训练，不能称为全部数据仅COCO。本表固定对应引用版本，不代表对后续新版本的评测。

**如何理解：** Bear-B的实测AP高于部分官方COCO行，但参数更多且使用外部骨干预训练，尚不能证明精度与延迟权衡更好。DEIMv2是最接近的设计参照；D-FINE和RT-DETRv2是解码器家族参照；YOLO26是重要部署基线。不同项目的S/L/X名称不代表相同模型规模。

## 速度比较原则

[BENCHMARKS.md](BENCHMARKS.zh-CN.md)中的RTX5090实测与其他作者的T4测速分开列示，不能直接相除宣称加速倍数。公平测速需要相同GPU、软件、640输入、batch1、精度规则、预热、同步、前后处理范围，并验证导出模型精度。本次没有新增基线GPU测速。Bear FLOPs暂不填写，后续需完整统计注意力和采样计算，不能从参数量猜测。

## 官方来源

- [DEIMv2模型库](https://github.com/Intellindust-AI-Lab/DEIMv2#1-model-zoo)
- [D-FINE模型库](https://github.com/Peterande/D-FINE#model-zoo)
- [RT-DETR官方结果](https://github.com/lyuwenyu/RT-DETR)
- [YOLO26官方指标](https://docs.ultralytics.com/models/yolo26/)
- [YOLO26训练配方](https://docs.ultralytics.com/guides/yolo26-training-recipe/)
