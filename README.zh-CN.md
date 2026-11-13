# 🐻 Bear Detector · 熊熊检测器

[English](README.md) | [简体中文](README.zh-CN.md) · [文档](docs/README.md) · [权重](https://github.com/xjn0412/bear-detector/releases) · [Docker](docs/getting-started/DOCKER.zh-CN.md)

**有一点重，就像熊一样。** 作者喜欢熊，而检测器也确实有点“大块头”，于是有了这个名字。它是通用多类别检测器，并不只检测熊。

**训练已完成（2026-11-14）：** Tiny、Small、Base、Large最佳EMA权重均可下载。Tiny / Base / Large分别完成68 / 52 / 44轮；Small由作者在68轮停止。[训练策略](docs/benchmarks/TRAINING_POLICY.md)。

![Bear Detector检测小熊](assets/bear-detected.jpg)

图片由AI生成；框与**0.9637置信度**来自真实**Base第41轮EMA**，PyTorch CPU FP32、640输入。[原始结果与权重哈希](assets/bear-inference.json)。这是展示图片，不是精度基准。

## 性能 · COCO val2017 / RTX5090

新发布混合包：**FP16主体存储 + 原始FP32解码器**。全量5,000张COCO验证，单尺度640、同一最佳EMA，无额外训练；PT与TRT的AP分别列出。

| 模型 | 输入 | AP50:95（PT / TRT） | 参数 M（完整 / 部署） | GFLOPs | PyTorch AMP ms | TensorRT ms | 混合包 MB |
|---|---:|---:|---:|---:|---:|---:|---|
| [Bear-T](configs/T.yml) | 640 | 56.4851 / 56.4449 | 52.05 / 51.10 | 132.18 | 12.84 | 2.75 | [115.08](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt.gz) |
| [Bear-S](configs/S.yml) | 640 | 57.8864 / 57.8563 | 73.68 / 72.73 | 201.21 | 16.44 | 3.34 | [155.18](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-mixed.pt.gz) |
| [Bear-B](configs/B.yml) | 640 | 59.0192 / 58.9243 | 112.11 / 111.16 | 310.75 | 15.69 | 3.82 | [226.34](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-mixed.pt.gz) |
| [Bear-L](configs/L.yml) | 640 | 60.0748 / 60.0236 | 221.40 / 220.45 | 622.32 | 15.20 | 4.79 | [428.79](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-mixed.pt.gz) |

延迟为单张RTX5090 GPU7、batch1，100次预热、3×300次CUDA Event、无CUDA Graph，从GPU输入到top300后处理；不含预处理与主机传输，TRT包含设备输入拷贝。挖矿和训练均停止；引擎并行构建结束后才测速。PT使用FP16主体+完整FP32解码器，TRT10.13.3.9同样保护解码器及敏感归一化/归约。

PT精度验证batch8，TRT固定batch1；因此AP差异同时包含后端及batch数值影响。两项门限均为最多下降0.1 AP点。FLOPs沿用明确范围的算术计数，1 MAC=2 FLOPs，不等同延迟。[完整结果与复现](docs/benchmarks/DEPLOYMENT5090.zh-CN.md) · [FLOPs口径](docs/benchmarks/COMPLEXITY.zh-CN.md)。

<details>
<summary>与其他检测器比较（作者发布结果，未复测）</summary>

| 模型 | 参数量 | AP50:95 | 来源 / 预训练说明 |
|---|---:|---:|---|
| Bear-T | 52.05M | **56.4597** | 本项目全量验证；EMA第58轮 |
| Bear-S | 73.68M | **57.8987** | 本项目全量验证；EMA第57轮 |
| Bear-B | 112.11M | **59.0008** | 本项目全量验证；EMA第42轮 |
| Bear-L | 221.40M | **60.0728** | 本项目全量验证；EMA第24轮 |
| [DEIMv2-X](https://github.com/Intellindust-AI-Lab/DEIMv2) | 50.3M | 57.8 | 官方DINOv3检测器 |
| [D-FINE-X](https://github.com/Peterande/D-FINE) | 62M | 55.8 | 官方COCO结果 |
| D-FINE-X，Objects365→COCO | 62M | 59.3 | 额外检测预训练 |
| [RT-DETRv2-X](https://github.com/lyuwenyu/RT-DETR) | 76M | 54.3 | 官方COCO结果 |
| [YOLO26-X](https://docs.ultralytics.com/models/yolo26/) | 55.7M | 57.5 | 默认NMS检测头；Objects365预训练；融合部署参数量 |

预训练数据、参数口径及测试后端不同，不构成严格对照实验。Bear更重；RTX5090测速不能直接与官方T4测速相除。[比较说明](docs/benchmarks/COMPARISON.zh-CN.md)。

</details>

## 模型与下载

混合包作为默认下载，所有尺寸采用同一策略；解压后加载到FP32模型，推理使用AMP并保留整个FP32解码器。**不要调用整个模型的`half()`**。混合包也可用于微调，准备脚本重置类别头并建立FP32参数与全新优化器。

| 模型 | 参数 | 最佳轮次 | 默认混合包 |
|---|---:|---:|---|
| Tiny | 52.05M | 58 | [115.08 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt.gz) |
| Small | 73.68M | 57 | [155.18 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-mixed.pt.gz) |
| Base | 112.11M | 42 | [226.34 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-mixed.pt.gz) |
| Large | 221.40M | 24 | [428.79 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-mixed.pt.gz) |

<details>
<summary>原始FP32 EMA归档（精确复现 / 可选微调起点）</summary>

[Tiny · 193.77 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-ema.pt.gz) · [Small · 274.17 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt.gz) · [Base · 416.91 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-ema.pt.gz) · [Large · 822.79 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-ema.pt.gz)

保留原始权重用于精确复现；FP16存储舍入无法恢复。它们不包含optimizer/scaler续训状态。

</details>

[SHA256 & usage](docs/models/WEIGHTS.zh-CN.md) · [Epoch history](docs/benchmarks/TRAINING_HISTORY.md)

检测监督仅使用COCO train2017，但DINOv3骨干使用外部LVD-1689M预训练，因此不是所有预训练数据都仅来自COCO，也不是COCO零样本检测。

## 快速开始

公开NVIDIA微调镜像，无需登录：

```bash
docker pull ghcr.io/xjn0412/bear-detector:finetune-cu130
```

| 任务 | 指南 |
|---|---|
| Docker微调、持久化挂载与Small初始化 | [Docker指南](docs/getting-started/DOCKER.zh-CN.md) |
| 源码安装与离线检查 | [安装](docs/getting-started/INSTALL.zh-CN.md) |
| COCO JSON / YOLO标注、任意正整数类别微调 | [微调](docs/guides/FINETUNE.zh-CN.md) |
| PyTorch预测和验证 | [推理](docs/guides/INFERENCE.zh-CN.md) |
| ONNX/TensorRT导出与精度验证 | [导出](docs/guides/EXPORT.zh-CN.md) |
| 查询命令和目录职责 | [命令索引](docs/reference/TOOLS.md) · [仓库结构](docs/reference/REPOSITORY.md) |

镜像已通过构建、依赖与CPU流程检查；镜像内GPU反向/DDP仍需在目标主机验证。数据和权重单独挂载。[全部文档](docs/README.md) · [验证范围](docs/benchmarks/VERIFICATION.zh-CN.md)。

## 架构

DINOv3预训练**ConvNeXt CNN** → 两个stride32上下文注意力模块 → 卷积多尺度颈部及轻量P2→P3细节分支 → 六层、三尺度DEIMv2查询解码器，300查询，固定640输入。主体FP16；**整个解码器、预测、匹配和损失FP32**。这些配置不使用ViT。[架构指南](docs/models/ARCHITECTURE.zh-CN.md)。

这是基于DEIMv2 / DEIM / D-FINE / RT-DETR的研究集成，不把上游解码器宣称为原创。[训练记录](docs/benchmarks/TRAINING_HISTORY.md) · [基准方法](docs/benchmarks/BENCHMARKS.zh-CN.md) · [验证资料索引](validation/README.md)。

## 写给自己

这个个人仓库，也是我在计算机视觉领域走过七年的一份纪念。说不上多么努力，但那是我人生中的七年。希望未来的我，能找到人生的目标。

—— xjn0412

## 许可与致谢

上游DEIMv2代码仅限**非商业用途**，DINOv3权重另有条款；本仓库不授予商业部署权利。[LICENSE.md](LICENSE.md) · [THIRD_PARTY.md](THIRD_PARTY.md) · [贡献指南](CONTRIBUTING.md)。

感谢DEIMv2、DEIM、D-FINE、RT-DETR、DINOv3和timm；原始声明保留在`source/`和`licenses/`中。
