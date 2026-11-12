# Documentation · 文档导航

[Project home / 项目主页](../README.md) · [中文首页](../README.zh-CN.md)

Start with Docker for a ready-made NVIDIA environment, or installation for a source checkout. Then choose weights, prepare your own classes and fine-tune. / 先选Docker或源码安装，再下载权重、准备类别和微调。

| Topic / 主题 | English | 简体中文 |
|---|---|---|
| Ready-made Docker environment / Docker环境 | [Docker](DOCKER.md) | [Docker指南](DOCKER.zh-CN.md) |
| Source installation / 源码安装 | [Installation](INSTALL.md) | [安装](INSTALL.zh-CN.md) |
| Models, parameters and results / 模型、参数与结果 | [Model zoo](MODEL_ZOO.md) | [模型列表](MODEL_ZOO.zh-CN.md) |
| Download weights / 权重下载 | [Weights](WEIGHTS.md) | [权重](WEIGHTS.zh-CN.md) |
| COCO / YOLO fine-tuning / 微调 | [Fine-tuning](FINETUNE.md) | [微调](FINETUNE.zh-CN.md) |
| PyTorch prediction / PyTorch预测 | [Inference](INFERENCE.md) | [推理](INFERENCE.zh-CN.md) |
| ONNX / TensorRT export / 导出 | [Export](EXPORT.md) | [导出](EXPORT.zh-CN.md) |
| Architecture and precision / 架构与精度策略 | [Architecture](ARCHITECTURE.md) | [架构](ARCHITECTURE.zh-CN.md) |
| Benchmark protocol / 基准方法 | [Benchmarks](BENCHMARKS.md) | [基准](BENCHMARKS.zh-CN.md) |
| Other detector comparisons / 同类模型比较 | [Comparison](COMPARISON.md) | [比较](COMPARISON.zh-CN.md) |
| Verified behavior and limits / 验证范围 | [Verification](VERIFICATION.md) | [验证](VERIFICATION.zh-CN.md) |

Shared references / 通用索引:

- [Command catalog / 命令索引](TOOLS.md)
- [Repository map / 仓库结构](REPOSITORY.md)
- [Epoch history / 逐轮记录](TRAINING_HISTORY.md)
- [Release notes / 发布说明](RELEASE_NOTES.md)
- [Raw validation evidence / 原始验证资料](../validation/README.md)
- [Contributing / 贡献指南](../CONTRIBUTING.md)

Tiny is training; fresh Base then Large are queued with10-epoch patience. Small and Tiny preview weights are downloadable. [Training policy](TRAINING_POLICY.md). / Tiny训练中，之后依次为Base、Large；连续10轮mAP无提升切换。Small和Tiny预览权重已发布。
