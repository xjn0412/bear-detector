# 🐻 Bear Detector · 熊熊检测器

[English](README.md) | [简体中文](README.zh-CN.md)

**有一点重，就像熊一样。** 作者喜欢熊，而这个检测器也确实有点“大块头”，于是有了 Bear Detector 这个名字。它仍然是通用多类别检测器，并不只检测熊。

![Bear Detector 检测小熊](assets/bear-detected.jpg)

小熊图片由 AI 生成；图中的框与 **0.9637 置信度**来自真实的 **Base 第41轮 EMA 权重**，使用 PyTorch CPU FP32、640输入推理。[原始结果与权重哈希](assets/bear-inference.json)。这是展示图片，不是精度基准。

## 写给自己

这个个人仓库，也是我在计算机视觉领域走过七年的一份纪念。说不上多么努力，但那是我人生中的七年。希望未来的我，能找到人生的目标。

—— xjn0412

## 模型简介

DINOv3 预训练 **ConvNeXt CNN 骨干**，配合两个最深层上下文注意力模块、卷积多尺度特征融合颈部和六层 DEIMv2 查询解码器。提供 Tiny、Small、Base 三种规模，固定输入640×640，面向 NVIDIA GPU。

这是基于 DEIMv2、DEIM、D-FINE、RT-DETR 的研究集成与修改，保留上游归属与许可证，不把上游解码器宣称为原创。支持任意正整数类别数微调、COCO JSON / YOLO检测文本标注、PyTorch推理、ONNX/TensorRT导出和同权重精度对比。

**许可证：** 上游 DEIMv2 材料仅限非商业用途；DINOv3 另有许可条款。请阅读 [LICENSE.md](LICENSE.md)、[THIRD_PARTY.md](THIRD_PARTY.md) 和 `licenses/`。本仓库不授予商业部署权利。

## 实测结果

**M 表示百万参数。** COCO val2017 全部5,000张，640、单尺度、EMA、PyTorch FP32验证：

| 版本 | 总参数 | 骨干参数 | 选定轮次 | mAP50:95 |
|---|---:|---:|---:|---:|
| Bear-B | 112.11M | 87.56M | 41 | 58.9306 |
| Bear-S | 73.68M | 49.45M | 57 | 57.8987 |
| Bear-T | 52.05M | 27.82M | — | 尚未测得 |

2026-11-12：作者已停止B/S训练，分别完成56/68轮；T开始训练。不是完整、无中断的对照实验。B在第20轮后、S在第30轮后修复数值问题并从完整检查点恢复。没有达到65AP的主张。检测监督仅用COCO train2017，但DINOv3骨干使用外部LVD-1689M预训练，因此不是所有预训练数据都仅来自COCO，也不是COCO零样本检测。

相同 Base 第41轮 EMA 权重，在固定的1,000张 COCO验证子集上、RTX5090、batch1：

| 后端 | 子集 mAP50:95 | GPU平均延迟 | FPS |
|---|---:|---:|---:|
| PyTorch FP32 | 59.4238 | 19.4644ms | 51.38 |
| PyTorch FP16主体 + FP32解码器 | 59.4149 | 19.1562ms | 52.20 |
| TensorRT FP16主体 + FP32解码器 | 59.3778 | 4.0203ms | 248.74 |

子集AP不能当作全量COCO AP。计时无CUDA Graph，包含top300后处理，不含图像预处理与主机传输；TensorRT额外包含设备缓冲区拷贝。全量COCO TensorRT精度一致性、S/T部署速度尚未验证。[基准说明](docs/BENCHMARKS.zh-CN.md)。

## 使用指南

- [安装与离线自检](docs/INSTALL.zh-CN.md)
- [架构与数值精度策略](docs/ARCHITECTURE.zh-CN.md)
- [COCO / YOLO 标注微调](docs/FINETUNE.zh-CN.md)
- [PyTorch 图片推理](docs/INFERENCE.zh-CN.md)
- [ONNX / TensorRT 导出与精度检查](docs/EXPORT.zh-CN.md)
- [已验证项目与限制](docs/VERIFICATION.zh-CN.md)

安装完成后可先运行：

```bash
python tools/verify_cpu.py --onnx --output results/cpu-verification.json
```

这会用临时目录检查随机Tiny初始化、合成三类标注、COCO/YOLO两条准备路径、CPU预测和ONNX导出，不下载检测权重，不占用GPU。合成数据仅用于代码自检，AP没有实际意义。

## COCO 和 YOLO 微调

当前代码发布不包含已训练检测权重、TensorRT引擎或数据图片。可自行提供匹配的完整检测器权重，或从公开DINOv3骨干开始：

```bash
python tools/initialize.py --config configs/T.yml --pretrained --output weights/t_initial.pt
python tools/prepare_finetune.py --config configs/T.yml --checkpoint weights/t_initial.pt \
  --data data/my_dataset/data.yaml --output runs/my_dataset --epochs 30 --batch 8
python tools/train_custom.py --run runs/my_dataset
```

此示例仅骨干预训练，检测器是新初始化，不会直接继承表中COCO精度。COCO JSON 使用 `--train-json`、`--val-json`、`--train-images`、`--val-images` 替代 `--data`；两种输入最终进入相同训练管线。详见微调指南。

## 致谢

感谢 DEIMv2、DEIM、D-FINE、RT-DETR、DINOv3 与 timm；原始版权声明保留在 `source/` 中。

[逐轮验证 mAP 记录](docs/TRAINING_HISTORY.md)

[相近检测器比较](docs/COMPARISON.zh-CN.md)：官方AP与参数量参照，明确预训练和测速口径差异。

[权重格式与较小下载包](docs/WEIGHTS.zh-CN.md)。
