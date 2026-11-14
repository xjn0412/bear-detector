# COCO检测训练限定：比较与分析

[English](COMPARISON.md) | [简体中文](COMPARISON.zh-CN.md)

**主比较只允许COCO train2017检测监督，不包含Objects365、OpenImages或grounding检测预训练阶段。** 主成本曲线全部640输入、单尺度评测。资料日期2026-11-14；Bear为全量5,000张val2017实测，外部为作者发布值，本次没有重跑。

这里“COCO-only”限定的是**检测器训练数据**，不是骨干从未看过外部图片。Bear使用外部DINOv3 ConvNeXt预训练，DEIMv2也使用DINOv3表征，其他方法有各自骨干初始化。这与现有Bear检查点的训练条件相符；无需重新训练来修正比较分组。

## 主图与结果

![仅COCO检测训练的精度与成本](../../assets/benchmarks/comparison-efficiency.png)

| Model | COCO val AP50:95 | Params M¹ | GFLOPs² | Source |
|---|---:|---:|---:|---|
| Bear-T | 56.49 | 51.10 | 132.18 | [Source](https://github.com/xjn0412/bear-detector/blob/main/docs/benchmarks/DEPLOYMENT5090.md) |
| Bear-S | 57.89 | 72.73 | 201.21 | [Source](https://github.com/xjn0412/bear-detector/blob/main/docs/benchmarks/DEPLOYMENT5090.md) |
| Bear-B | 59.02 | 111.16 | 310.75 | [Source](https://github.com/xjn0412/bear-detector/blob/main/docs/benchmarks/DEPLOYMENT5090.md) |
| Bear-L | 60.07 | 220.45 | 622.32 | [Source](https://github.com/xjn0412/bear-detector/blob/main/docs/benchmarks/DEPLOYMENT5090.md) |
| DEIMv2-S | 50.90 | 9.70 | 49.50 | [Source](https://github.com/Intellindust-AI-Lab/DEIMv2#1-model-zoo) |
| DEIMv2-M | 53.00 | 18.10 | 84.10 | [Source](https://github.com/Intellindust-AI-Lab/DEIMv2#1-model-zoo) |
| DEIMv2-L | 56.00 | 32.20 | 144.50 | [Source](https://github.com/Intellindust-AI-Lab/DEIMv2#1-model-zoo) |
| DEIMv2-X | 57.80 | 50.30 | 200.00 | [Source](https://github.com/Intellindust-AI-Lab/DEIMv2#1-model-zoo) |
| D-FINE-S | 48.50 | 10.00 | 25.00 | [Source](https://github.com/Peterande/D-FINE#model-zoo) |
| D-FINE-M | 52.30 | 19.00 | 57.00 | [Source](https://github.com/Peterande/D-FINE#model-zoo) |
| D-FINE-L | 54.00 | 31.00 | 91.00 | [Source](https://github.com/Peterande/D-FINE#model-zoo) |
| D-FINE-X | 55.80 | 62.00 | 202.00 | [Source](https://github.com/Peterande/D-FINE#model-zoo) |
| RT-DETRv4-S | 49.80 | 10.00 | 25.00 | [Source](https://github.com/AIMIA-PKU/RT-DETRv4#-performance) |
| RT-DETRv4-M | 53.70 | 19.00 | 57.00 | [Source](https://github.com/AIMIA-PKU/RT-DETRv4#-performance) |
| RT-DETRv4-L | 55.40 | 31.00 | 91.00 | [Source](https://github.com/AIMIA-PKU/RT-DETRv4#-performance) |
| RT-DETRv4-X | 57.00 | 62.00 | 202.00 | [Source](https://github.com/AIMIA-PKU/RT-DETRv4#-performance) |

¹ Bear为部署参数，T/S/B/L完整参数52.05/73.68/112.11/221.40M；外部沿用作者口径。² FLOPs工具不同，Bear计入注意力矩阵乘法和双线性采样；参数及FLOPs不能替代统一硬件实测。Bear为当前混合权重PyTorch AMP结果，[历史训练AP](TRAINING_HISTORY.md)不变。

## 指定当前基线：YOLO26与RT-DETRv4

![明确标注检测训练数据的YOLO26、最新RT-DETRv4及Bear](../../assets/benchmarks/comparison-latest-baselines.png)

[RT-DETR官方项目](https://github.com/lyuwenyu/RT-DETR#-updates)将v4列为最新公开版本，原地址重定向至[AIMIA-PKU/RT-DETRv4](https://github.com/AIMIA-PKU/RT-DETRv4)。最新论文及仓库的S/M/L/X为49.8/53.7/55.4/57.0AP，使用修订值而不是旧49.7/53.5。参数与FLOPs来自[论文表1](https://arxiv.org/html/2510.25257v2#S4.T1)，不包含仅训练时使用的教师。v2只保留历史数据，不作为当前曲线基线。

YOLO26明确加入此表和首页图。这里采用公开检查点的默认NMS检测头AP；[官方配方](https://docs.ultralytics.com/guides/yolo26-training-recipe/)先做150轮Objects365，再微调COCO。没有伪造或替换成未经验证的COCO从头训练分数。虚线是用户指定的背景参照，仍不纳入仅COCO主分组。

| Model | AP50:95 | Params M | GFLOPs | Detector training |
|---|---:|---:|---:|---|
| Bear-T | 56.49 | 51.10 | 132.18 | COCO only |
| Bear-S | 57.89 | 72.73 | 201.21 | COCO only |
| Bear-B | 59.02 | 111.16 | 310.75 | COCO only |
| Bear-L | 60.07 | 220.45 | 622.32 | COCO only |
| YOLO26-n | 40.90 | 2.40 | 5.50 | Objects365 → COCO |
| YOLO26-s | 48.60 | 9.50 | 20.90 | Objects365 → COCO |
| YOLO26-m | 53.10 | 20.40 | 68.40 | Objects365 → COCO |
| YOLO26-l | 55.00 | 24.80 | 86.80 | Objects365 → COCO |
| YOLO26-x | 57.50 | 55.70 | 194.40 | Objects365 → COCO |
| RT-DETRv4-S | 49.80 | 10.00 | 25.00 | COCO only |
| RT-DETRv4-M | 53.70 | 19.00 | 57.00 | COCO only |
| RT-DETRv4-L | 55.40 | 31.00 | 91.00 | COCO only |
| RT-DETRv4-X | 57.00 | 62.00 | 202.00 | COCO only |

Bear-B/L比当前RT-DETRv4-X高2.02/3.07AP，报告参数约1.79/3.56倍；比YOLO26-x57.5高1.52/2.57AP，但检测训练数据不同。RT-DETRv4的DINOv3教师蒸馏属于训练技巧，不等于使用Objects365检测预训练。训练预算和特征迁移方式仍不同，不把外部T4测速与我们的5090直接比较。

## 训练条件审计

| Family | Detector training data | Backbone / training distinction |
|---|---|---|
| Bear | COCO train2017 only | DINOv3 ConvNeXt representation pretraining on LVD-1689M; Mosaic/MixUp/Copy-Blend and EMA in released config |
| DEIMv2 COCO rows | COCO only | DINOv3 ViT pretrained/distilled features; own augmentation/loss/schedule |
| D-FINE COCO rows | COCO only | Author backbone initialization and recipe; no Objects365 detector stage in selected rows |
| RT-DETRv4 COCO rows | COCO only | DINOv3 ViT-B/16 teacher distillation (DSI/GAM); teacher is absent at inference |
| YOLO26 / D-FINE O365 | Objects365 → COCO | Supplemental only; excluded from primary graph |
| YOLO27 preview / unverified rows | Not established here | Excluded from primary until detector-training provenance is verified |

来源：[DEIMv2 COCO模型库](https://github.com/Intellindust-AI-Lab/DEIMv2#1-model-zoo)与[训练论文](https://arxiv.org/abs/2509.20787)、[D-FINE明确分开的COCO/Objects365表](https://github.com/Peterande/D-FINE#model-zoo)、[RT-DETRv4最新结果](https://github.com/AIMIA-PKU/RT-DETRv4#-performance)及[论文表1](https://arxiv.org/html/2510.25257v2#S4.T1)、[YOLO26预训练配方](https://docs.ultralytics.com/guides/yolo26-training-recipe/)及本项目[配置](../../configs/B.yml)。

此前总览把仅COCO检测训练和额外检测数据混在一起，不适合作为主要比较。D-FINE-X应比较**55.8AP**，而非Objects365预训练后的59.3；YOLO26的57.5也来自Objects365预训练，因此移出主图。YOLO11因本次未完整核验检测预训练来源，仅保留补充背景；不表示认定其使用了Objects365。

数据相同不等于训练技巧相同。我们自己配置也使用Mosaic、MixUp、Copy-Blend、EMA及DEIM家族训练方法；DEIMv2有各自特征蒸馏、增强、损失和周期，RT-DETRv4使用DINOv3教师蒸馏。这里比较作者发布的整套系统，明确记录差异，不把AP差异全部归于架构。若要统一技巧，需要相同初始化、增强、损失、周期、随机种子和计算预算重跑，目前没有这样的对照结果。

## 修正后的性能解读

**在仅COCO检测训练组内，Bear的绝对AP有竞争力。** Bear-T56.49比D-FINE-X55.8高0.69AP，参数51.10M/62M、FLOPs132.18/202；Bear-B59.02比D-FINE-X和最新RT-DETRv4-X57.0分别高3.22/2.02AP，Bear-L60.07分别高4.27/3.07AP。RT-DETRv4-X为62M部署参数，Bear-B/L约为其1.79/3.56倍。DINOv3更强的骨干预训练可能贡献了差距，因此不能将其视为解码器或训练算法单独领先的证据。

**DEIMv2是更严格且接近的参照。** 二者都只用COCO检测监督，并使用DINOv3特征。DEIMv2-X为57.8AP/50.3M；Bear-T参数接近但低1.31AP；Bear-S高0.09AP但参数多约45%；Bear-B/L高1.22/2.27AP，参数约2.21/4.38倍。因此移除Objects365后，参数效率仍有改进空间，但应基于这一合适分组，而非额外数据模型。ViT/ConvNeXt骨干及配方仍不同，不能推断纯架构因果。

**Bear内部Base较均衡，Large精度最高。** 同5090 TensorRT：Base58.92AP/3.82ms，Large60.02AP/4.79ms，约25%延迟换1.10AP；Tiny最快2.75ms。不能与外部T4/A100结果直接宣称加速。

## 仅COCO检测训练的高分辨率参照

![仅COCO检测训练的精度背景](../../assets/benchmarks/comparison-leaderboard.png)

[DINO Swin-L5scale](https://github.com/IDEA-Research/DINO)在不使用Objects365和多尺度测试时为58.5AP；[Co-DINO Swin-L36轮LSJ](https://github.com/Sense-X/Co-DETR#co-detr-with-swin-l)在官方COCO表为60.7AP，与其Objects365预训练表分开。输入和训练配方不等同Bear固定640，因此只作COCO检测数据限定的背景，不混入主成本曲线。二者仍使用外部骨干预训练。不分配Bear官方排行榜名次。

用户指定的[SOTA2页面](https://www.sota2.com/research/sota/object-detection-on-coco-2017-val)混合多种训练条件；额外数据条目移至[独立附录](COMPARISON_EXTRA_DATA.zh-CN.md)，不用于评价主比较。与Focal-Stable-DINO相差4.53AP只是跨数据、跨分辨率数字差，不能视为公平条件下的落后。

## 目标大小与后续验证

| Bear | AP small | AP medium | AP large |
|---|---:|---:|---:|
| Bear-T | 39.50 | 60.88 | 73.50 |
| Bear-S | 41.03 | 62.93 | 75.52 |
| Bear-B | 42.72 | 63.99 | 75.86 |
| Bear-L | 43.77 | 64.89 | 76.90 |

小目标是Bear内部AP最低的一组。之前与Focal-Stable-DINO的差距同时受到Objects365及高分辨率影响，不能诊断为架构缺陷。后续应在同样COCO检测数据条件下做分辨率、颈部/解码器消融和多种子配方对照。本次仅修正文档，没有启动训练。

## Bear实测延迟

![Bear后端精度与延迟](../../assets/benchmarks/comparison-latency.png)

单5090、batch1、640、100次预热、3×300CUDA Event、无CUDA Graph；不含预处理和主机传输，TRT含设备输入拷贝。精度为全量COCO：PTbatch8/TRTbatch1，分别使用对应后端AP。[完整协议与显存](DEPLOYMENT5090.zh-CN.md)。

## 绘图复现

[数据、分组标记及来源](../../validation/benchmarks/comparison-20261114/comparison.json)。安装matplotlib后运行`python tools/plot_comparison.py`。主图只选择明确的`primary_coco_only`条目，排除所有额外数据、未核验及预览条目。作者发布系统之间的比较不等于统一训练技巧的消融。组织方式参考[Ultralytics GitHub](https://github.com/ultralytics/ultralytics)。
