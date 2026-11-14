# Documentation · 文档导航

[Project home / 项目主页](../README.md) · [中文首页](../README.zh-CN.md)

Start with Docker for a ready-made NVIDIA environment, or installation for a source checkout. Then choose weights, prepare your own classes and fine-tune. / 先选Docker或源码安装，再下载权重、准备类别和微调。

| Topic / 主题 | English | 简体中文 |
|---|---|---|
| Ready-made Docker environment / Docker环境 | [Docker](getting-started/DOCKER.md) | [Docker指南](getting-started/DOCKER.zh-CN.md) |
| Source installation / 源码安装 | [Installation](getting-started/INSTALL.md) | [安装](getting-started/INSTALL.zh-CN.md) |
| Models, parameters and results / 模型、参数与结果 | [Model zoo](models/MODEL_ZOO.md) | [模型列表](models/MODEL_ZOO.zh-CN.md) |
| Download weights / 权重下载 | [Weights](models/WEIGHTS.md) | [权重](models/WEIGHTS.zh-CN.md) |
| COCO / YOLO fine-tuning / 微调 | [Fine-tuning](guides/FINETUNE.md) | [微调](guides/FINETUNE.zh-CN.md) |
| PyTorch prediction / PyTorch预测 | [Inference](guides/INFERENCE.md) | [推理](guides/INFERENCE.zh-CN.md) |
| ONNX / TensorRT export / 导出 | [Export](guides/EXPORT.md) | [导出](guides/EXPORT.zh-CN.md) |
| Architecture and precision / 架构与精度策略 | [Architecture](models/ARCHITECTURE.md) | [架构](models/ARCHITECTURE.zh-CN.md) |
| Benchmark protocol / 基准方法 | [Benchmarks](benchmarks/BENCHMARKS.md) | [基准](benchmarks/BENCHMARKS.zh-CN.md) |
| Comparison graphs, leaderboard and analysis / 比较图、排行榜与分析 | [Comparison](benchmarks/COMPARISON.md) | [比较](benchmarks/COMPARISON.zh-CN.md) |
| Fine-tuning benchmarks / 微调比较 | [B/L · three datasets](benchmarks/BEAR_L_TRANSFER.md) · [YOLO50/100](benchmarks/YOLO_LONG_BUDGET.md) | [B/L · 三组数据](benchmarks/BEAR_L_TRANSFER.zh-CN.md) · [YOLO50/100](benchmarks/YOLO_LONG_BUDGET.zh-CN.md) |
| Verified behavior and limits / 验证范围 | [Verification](benchmarks/VERIFICATION.md) | [验证](benchmarks/VERIFICATION.zh-CN.md) |

Shared references / 通用索引:

- [Command catalog / 命令索引](reference/TOOLS.md)
- [Repository map / 仓库结构](reference/REPOSITORY.md)
- [Epoch history / 逐轮记录](benchmarks/TRAINING_HISTORY.md)
- [Release notes / 发布说明](reference/RELEASE_NOTES.md)
- [Raw validation evidence / 原始验证资料](../validation/README.md)
- [Contributing / 贡献指南](../CONTRIBUTING.md)

All T/S/B/L selected EMA weights are published; no model remains in training. / T/S/B/L最佳EMA均已发布，训练已结束。

Layout follows task-oriented documentation and model-zoo conventions used by [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [EdgeCrafter / ECDet](https://github.com/Intellindust-AI-Lab/EdgeCrafter) and [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2); this does not imply their code or licensing was copied.

- [FLOPs methodology / 计算量口径](benchmarks/COMPLEXITY.md) · [中文](benchmarks/COMPLEXITY.zh-CN.md)
- [Weight size audit / 权重大小审计](models/WEIGHT_SIZE.md) · [中文](models/WEIGHT_SIZE.zh-CN.md)

- [Final RTX5090 latency, memory and deployment accuracy](benchmarks/DEPLOYMENT5090.md) · [最终部署验证与测速](benchmarks/DEPLOYMENT5090.zh-CN.md)
