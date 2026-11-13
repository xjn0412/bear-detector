# Validation evidence · 验证资料

[Project](../README.md) · [Documentation](../docs/README.md) · [Verification scope](../docs/benchmarks/VERIFICATION.md)

These are recorded measurements and execution checks, not live training status. Raw evidence files were moved without changing their bytes; paths inside historical JSON are provenance, not guaranteed local paths. / 这些是历史测量和执行记录，并非实时训练状态。原始文件移动时字节保持不变，JSON中的历史路径仅用于溯源。

| Directory / 目录 | Evidence / 内容 | Entry / 入口 |
|---|---|---|
| `benchmarks/base41/` | Same Base41 checkpoint, PyTorch/TensorRT, fixed1,000-image subset / 同权重后端对比 | [Results](benchmarks/base41/RESULTS.md) |
| `releases/small/` | Full5,000-image compact/reference accuracy and released hashes / S紧凑权重全量精度与哈希 | [COCO parity](releases/small/S_compact_coco.json) · [Manifest](releases/small/S_release_manifest.json) |
| `releases/` | Weight-storage compression preparation / 权重压缩准备记录 | [Compression](releases/weight-compression.json) |
| `containers/` | Published image digest, CPU checks and installed dependencies / 镜像digest、CPU检查与依赖 | [Image](containers/docker-image.json) · [Packages](containers/docker-pip-freeze.txt) |
| `fine-tuning/` | Custom-class preparation, backward/update/resume smoke evidence / 自定义类别执行自检 | [Training smoke](fine-tuning/train-cli-evidence-scale1.json) |
| `repository/` | Source packaging, CLI and bear illustration checks / 源码、CLI与展示检查 | [Repository](repository/repository-verification.json) |
| `training/` | Historical epoch metrics / 历史逐轮指标 | [Epoch snapshot](training/epoch-map-20261112.json) |
| `environment/` | Recorded measurement environment / 测量环境 | [Environment](environment/environment.json) |

Full COCO AP, subset AP, synthetic-data smoke checks and latency measurements have different scopes. Read [benchmark methodology](../docs/benchmarks/BENCHMARKS.md) before comparing them. Compact TensorRT parity and Docker GPU training are not validated by the CPU checks. / 全量AP、子集AP、合成自检及延迟口径不同，不能混用；CPU检查不证明紧凑权重TRT精度或镜像GPU训练通过。
