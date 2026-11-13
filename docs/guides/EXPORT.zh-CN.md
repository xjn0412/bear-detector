# ONNX / TensorRT 导出与精度检查

[English](EXPORT.md) | [简体中文](EXPORT.zh-CN.md)

先安装可选导出依赖。使用匹配的config、可信完整检查点与classes.json；全量加载PyTorch检查点需要信任其来源。导出工具选择EMA。

```bash
python tools/export_engine.py --config runs/my_dataset/config.yml \
  --checkpoint runs/my_dataset/training/best_stg1.pth --output exports/my_dataset --build
```

去掉 `--build` 只在CPU导出/检查ONNX。默认TensorRT主体FP16，整个解码器和敏感归约FP32；不能用 `model.half()` 代替。export.json记录哈希与类别数，classes.json自动复制。引擎初始标记为精度未验证。

固定batch1×3×640×640：images为FP32，orig_target_sizes为int64 `[width,height]`；输出labels、原图像素xyxy boxes与scores。RGB、PIL双线性直接缩放、ImageNet归一化，**不是letterbox**。换权重、类别数、输入形状或不兼容GPU/TensorRT版本必须重新导出。

导出使用显式双线性GatherElements，避免已观察到的TensorRT GridSample偏差；原生PyTorch仍用grid_sample。不能因为输出有限就去掉该修复。

## 相同权重、相同数据比较

```bash
python tools/evaluate.py --backend torch-fp32 --config runs/my_dataset/config.yml \
  --checkpoint runs/my_dataset/training/best_stg1.pth \
  --annotations runs/my_dataset/val.json --images data/my_dataset/val --output results/fp32.json
python tools/evaluate.py --backend torch-amp --config runs/my_dataset/config.yml \
  --checkpoint runs/my_dataset/training/best_stg1.pth \
  --annotations runs/my_dataset/val.json --images data/my_dataset/val --output results/amp.json
python tools/evaluate.py --backend tensorrt --config runs/my_dataset/config.yml \
  --checkpoint runs/my_dataset/training/best_stg1.pth --engine exports/my_dataset/model.engine \
  --annotations runs/my_dataset/val.json --images data/my_dataset/val --output results/trt.json
python tools/compare_results.py --reference results/fp32.json --candidate results/trt.json \
  --max-drop 0.1 --output results/parity.json
```

AMP结果也应与FP32比较。门限0.1是0.1个AP百分点，不是10%。检查点/标注哈希、输入与类别数不同会被拒绝。子集通过不代表全量或业务数据一致；关注逐类、小目标及真实场景表现。

计时默认50次预热、200次CUDA-event测量，无CUDA Graph；包含top300，不含CPU预处理/主机传输；TRT额外含设备缓冲区拷贝，另报图像到CPU输出的端到端时间。GPU应无竞争计算。TensorRT10弱类型精度约束已验证，TensorRT11/强类型API尚未验证。详细协议见[基准说明](../benchmarks/BENCHMARKS.zh-CN.md)。

最终发布模型的新全量COCO验证和独立测速见[部署基准](../benchmarks/DEPLOYMENT5090.zh-CN.md)。当前精度约束导出器实测版本为TensorRT10.13.3.9，遇到TensorRT11会明确拒绝，不会悄悄改变精度策略。
