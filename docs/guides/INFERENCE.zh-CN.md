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
