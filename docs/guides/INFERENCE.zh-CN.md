# PyTorch 图片推理

[English](INFERENCE.md) | [简体中文](INFERENCE.zh-CN.md)

```bash
python tools/predict.py --config runs/my_dataset/config.yml \
  --checkpoint runs/my_dataset/training/best_stg1.pth --classes runs/my_dataset/classes.json \
  --image data/my_dataset/val/example.jpg --device cuda --amp --threshold 0.3 --output results/example.json
```

去掉amp为FP32参考；CPU使用 `--device cpu` 且不加amp。AMP只覆盖骨干/上下文/颈部，整个解码器FP32，不要调用model.half()。输出JSON包含类别索引、原图像素xyxy框、得分；提供classes.json时还含名称和原始类别ID。应用显示门限不应用于高门限AP评估。

RGB转图后，PIL双线性直接缩放640×640、除255、mean[.485,.456,.406]/std[.229,.224,.225]归一化，不是letterbox。原图尺寸顺序width,height。输出top300类/查询组合，不额外NMS。

给预测结果画框：

```bash
python tools/draw_predictions.py --image assets/bear-input.jpg \
  --predictions assets/bear-inference.json --output results/bear-boxes.jpg
```

此命令只绘制JSON中的实际坐标和得分，不执行模型、也不创造检测框。仓库小熊框来自B41EMA的CPU FP32真实推理；图像由AI生成，不能作为性能证据。

## 推理数值安全

推理时 FP16 主干或颈部也可能溢出，且没有 GradScaler。共享 PyTorch 推理封装现在
检查特征、解码器输出和最终预测；AMP 结果无效时，会用 FP32 重算同一图像/批次，
后续调用保持 FP32，并通过警告及 `precision_report` 披露回退。
输入无效或 FP32 仍无效时直接报错。参数保持 FP32，使用 autocast，
不要直接 `model.half()`。下溢或舍入也可能影响精度而不产生 NaN，
因此仍需在代表性数据上与 FP32 比较 AP。

此回退只适用于 PyTorch 动态推理，不会嵌入 ONNX/TensorRT。
静态导出明确使用 `Inference.create(..., checked=False)`。
TensorRT 导出现在强制 P5 投影、解码器和敏感归约使用 FP32；
找不到 P5 卷积作用域时拒绝构建。旧引擎需要重建，新引擎需要单独验证。
TensorRT 运行时遇到非有限预测会报错，提示改用 PyTorch FP32，不会在引擎内部自动重算。
本次已验证 CUDA PyTorch 和部署模式 Bear-L，尚未构建或验证新的 TensorRT 引擎。


2026-11-16更新：已针对 CR7 Bear-L 最佳断点重建并验证 TensorRT，完成完整验证/测试集、
8张历史输入的有限性压力检查和 batch1 计时。见[测量结果与范围](../benchmarks/TENSORRT_REVALIDATION_20261116.md)。
此前“尚未验证新引擎”的说明已被本次结果取代；不代表其他断点已通过验证。
