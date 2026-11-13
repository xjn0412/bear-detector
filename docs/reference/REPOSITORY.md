# Repository map · 仓库结构

[Documentation / 文档](../README.md)

| Path / 路径 | Responsibility / 职责 |
|---|---|
| `README.md`, `README.zh-CN.md` | Front pages: status, comparison, downloads, starting points / 状态、对比、下载和入口 |
| `docs/` | Bilingual user guides and reference indexes / 双语用户指南与索引 |
| `configs/` | T/S/B/L model configs and COCO class mapping / 模型配置与COCO类别映射 |
| `tools/` | Stable command-line entry points and helpers / 稳定CLI入口与辅助模块 |
| `backbone.py` | Bear's ConvNeXt pyramid and context integration / 骨干与上下文集成 |
| `runtime.py` | Precision safety and training runtime / 精度保护与训练运行时 |
| `compat.py` | Upstream compatibility adaptations / 上游兼容适配 |
| `source/` | Retained upstream engine and train entry point, with attribution / 保留归属的上游引擎 |
| `tests/` | Class schemas, sampling and YOLO conversion tests / 类别、采样与标注转换测试 |
| `validation/` | Recorded evidence, grouped by purpose / 按用途分组的验证资料 |
| `assets/` | Mascot input, actual prediction rendering and provenance / 小熊展示与推理记录 |
| `licenses/`, `LICENSE.md`, `THIRD_PARTY.md` | Upstream terms and attribution / 上游许可与归属 |
| `Dockerfile`, `.dockerignore` | NVIDIA training environment build / NVIDIA训练环境构建 |
| `.github/workflows/` | CPU checks and gated container publication / CPU检查与镜像发布 |
| `requirements*.txt`, `constraints-tested.txt` | Dependencies and recorded tested pins / 依赖与实测版本 |

Local datasets, model weights, generated runs and engines are ignored by Git. Keep them on persistent host storage and back up checkpoints off-node. Released detector downloads live in GitHub Releases; container images live in GHCR. / 数据、权重、运行输出和引擎不提交Git，应放持久化目录并异机备份；发布权重在Releases，镜像在GHCR。

The CLI paths and runtime import locations are intentionally stable. Grouping raw evidence does not change model architecture, checkpoints or training recipes. Existing guide URLs remain valid through compatibility pages. / 本次整理保持命令、运行时导入路径及原指南URL稳定，不改变模型、检查点或训练配方。

## Documentation groups

| Directory | Purpose |
|---|---|
| `docs/getting-started/` | Source installation and Docker |
| `docs/guides/` | Fine-tuning, inference and export |
| `docs/models/` | Model zoo, architecture and downloads |
| `docs/benchmarks/` | Accuracy, latency, comparison, history and verification |
| `docs/reference/` | CLI, repository map and release index |
| `docs/releases/` | Per-model final release notes |

Runtime modules, configs, tools and upstream source retain their import paths to preserve checkpoint and command compatibility.
