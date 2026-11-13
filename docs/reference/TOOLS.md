# Command catalog · 命令索引

[Documentation / 文档](../README.md) · [Repository map / 仓库结构](REPOSITORY.md)

Run commands from the repository root, or `/opt/bear-detector` inside the image. Every entry below supports `--help`. Examples and required arguments belong to the linked guides. / 在仓库根目录或镜像的`/opt/bear-detector`运行，各命令均支持`--help`。

| Workflow / 流程 | Command / 命令 | Purpose / 用途 |
|---|---|---|
| Initialize / 初始化 | [tools/initialize.py](../../tools/initialize.py) | Matching DINOv3 backbone or random smoke initialization / 骨干初始化或随机自检 |
| Prepare / 准备 | [tools/prepare_finetune.py](../../tools/prepare_finetune.py) | COCO/YOLO, class reset, normalized labels, fresh run / 标注、类别头和新运行 |
| Train / 训练 | [tools/train_custom.py](../../tools/train_custom.py) | Single process or torchrun; same-run full resume / 单卡、多卡及完整恢复 |
| Convert / 转换 | [tools/yolo_to_coco.py](../../tools/yolo_to_coco.py) | Standalone YOLO detection label conversion / YOLO检测标注转换 |
| Predict / 预测 | [tools/predict.py](../../tools/predict.py) | PyTorch image prediction / PyTorch图片推理 |
| Draw / 绘制 | [tools/draw_predictions.py](../../tools/draw_predictions.py) | Draw saved predictions / 绘制已保存预测 |
| Evaluate / 评测 | [tools/evaluate.py](../../tools/evaluate.py) | COCO evaluation with selected backend / 指定后端COCO评测 |
| Export / 导出 | [tools/export_engine.py](../../tools/export_engine.py) | Fixed-shape ONNX and optional TensorRT / 固定形状导出 |
| Compare / 比较 | [tools/compare_results.py](../../tools/compare_results.py) | Same-checkpoint backend accuracy gate / 同权重后端精度门限 |
| Historical compression check / 历史压缩验证 | [tools/validate_compact.py](../../tools/validate_compact.py) | Paired original/compact COCO accuracy / 原始与紧凑权重对比 |
| Subset / 子集 | [tools/make_subset.py](../../tools/make_subset.py) | Reproducible validation subset / 可复现验证子集 |
| Smoke data / 自检数据 | [tools/make_smoke_data.py](../../tools/make_smoke_data.py) | Synthetic three-class COCO/YOLO wiring data / 合成三类标注 |
| Offline checks / 离线自检 | [tools/verify_cpu.py](../../tools/verify_cpu.py) | CPU initialization→preparation→prediction, optional ONNX / CPU端到端自检 |

[common.py](../../tools/common.py) and [portable_sampling.py](../../tools/portable_sampling.py) are implementation helpers, not separate user-facing workflows. / 两者是实现辅助模块。

Use [FINETUNE.md](../guides/FINETUNE.md) / [中文微调](../guides/FINETUNE.zh-CN.md) for training. `source/train.py` is the retained upstream entry point; use `tools/train_custom.py` so Bear's numerical safety runtime is installed. / 不要跳过Bear运行时直接调用上游训练入口。

CPU complexity profiling (no weights required): `python tools/profile_model.py --config configs/T.yml --output results/T-profile.json`. [Scope and reproduction](../benchmarks/COMPLEXITY.md).
