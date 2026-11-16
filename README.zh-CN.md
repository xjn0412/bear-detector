# 🐻 Bear Detector v2

[English](README.md) | [简体中文](README.zh-CN.md) · [稳定版 v1](https://github.com/xjn0412/bear-detector/tree/main) · [架构与训练说明](docs/models/ARCHITECTURE_V2.md)

**完整 P2–P5 FPN/PAN、四尺度池化上下文注意力，以及四尺度查询解码器。**

v2 是实验开发分支。代码与数值测试已提供；尚无训练完成的 v2 权重或 COCO AP。已发布的 v1 成绩和权重请查看 [v1](https://github.com/xjn0412/bear-detector/tree/main)。

| 组件 | v2 配置 |
|---|---|
| 主干 | DINOv3 预训练 ConvNeXt T/S/B/L，输出四尺度 |
| 上下文 | 每尺度池化至20×20；共1600 tokens；两层256维、8头注意力 |
| 颈部 | 完整 P2–P5 自顶向下 FPN、自底向上 PAN，通道256 |
| 解码器 | 6层、300 queries；四尺度采样点4/4/2/2 |
| 数值保护 | 投影、上下文、解码器与损失 FP32；CNN/颈部使用检查后的 FP16 AMP；动态缩放与持续 FP32 回退 |
| V2-L，输入640 | **783.99 GFLOPs/图；完整224.53M /部署223.16M 参数** |

检测特征保持160²/80²/40²/20²，仅上下文分支池化。敏感运算在执行前进入 FP32；推理首个输入同时比较 AMP 与 FP32 的特征误差，因为下溢也可能产生有限的零值。这些保护不代表任意数据均无溢出、下溢，也不代表精度必然一致。

[架构、精度策略、初始化与训练命令](docs/models/ARCHITECTURE_V2.md)。配置：[Tiny](configs/V2-T.yml)、[Small](configs/V2-S.yml)、[Base](configs/V2-B.yml)、[Large](configs/V2-L.yml)。默认训练总 batch 为8，并启用主干梯度检查点。v1 检测权重及现有已发布 Docker 镜像不包含此新架构。

[验证证据](validation/v2/README.md) · [原文档](docs/README.md) · [许可证](LICENSE.md) · [第三方声明](THIRD_PARTY.md)

Bear Detector 因作者喜欢熊而得名，也是七年计算机视觉经历的纪念，希望未来能找到人生目标。v2 继续基于 DEIMv2 / DEIM / D-FINE / RT-DETR 与 DINOv3 集成；上游非商业及权重许可条款继续适用。
