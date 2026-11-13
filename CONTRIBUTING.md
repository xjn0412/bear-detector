# Contributing · 贡献指南

Start with the [repository map](docs/reference/REPOSITORY.md) and [command catalog](docs/reference/TOOLS.md). Keep user documentation in English and Chinese. Preserve upstream licenses and attribution. / 先读目录和命令索引；用户文档保持中英双语，保留上游许可与归属。

Install dependencies using the [installation guide](docs/getting-started/INSTALL.md), then run:

```bash
python tests/test_classes.py
python tests/test_sampling.py
python tests/test_yolo.py
python -m compileall -q backbone.py runtime.py compat.py tools
python tools/verify_cpu.py --output results/cpu-verification.json
```

CPU smoke checks validate execution and schema handling, not trained accuracy or GPU speed. For numerical changes, record a held-out accuracy comparison using the same checkpoint, images, preprocessing and backend settings. Keep full COCO and subset results clearly separate. / CPU自检不等于精度或速度证明；数值改动需同权重、同数据、同预处理对比，明确全量和子集口径。

Do not commit datasets, detector checkpoints, engines, credentials or generated training outputs. Store measurement evidence under the appropriate `validation/` category and explain its scope in the relevant guide. Preserve FP32 decoder/criterion safety unless a separately measured change justifies a new policy. / 不提交数据、检查点、引擎、凭据或训练输出；验证记录放对应类别，注明范围。

Existing CLI paths are part of the documented interface. If they change, update Docker examples, tests and both language guides in the same change. / 改命令路径时必须同时更新Docker示例、测试和双语指南。
