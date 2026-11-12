# Bear Detector B/S EMA weights / B/S预训练权重

Research preview. B epoch41 full COCO val2017 AP50:95=58.9306; S epoch57=57.8987. Input640, 80 classes, EMA. Detector supervision: COCO train2017; DINOv3 ConvNeXt backbone uses external pretraining. These are selected best checkpoints from numerically repaired runs.

## Recommended / 推荐

`bear-b-coco-ema.pt.gz` and `bear-s-coco-ema.pt.gz` contain exact original FP32 EMA tensors, compressed losslessly. SHA256SUMS and weights-manifest.json identify every asset. Full optimizer/scaler resume states are excluded; the model tensors support inference and fresh fine-tuning.

推荐使用上述无损FP32 EMA包，解压后保留原始模型权重。仅模型与元数据，不含完整训练续训状态；支持推理和新任务微调。

## Experimental compact / 实验性紧凑包

`*-compact-experimental.pt.gz` stores FP16 backbone/context/neck tensors and unchanged FP32 decoder tensors. It loads into the normal FP32 model. This introduces rounding; full COCO AP and TensorRT parity are NOT validated for compact assets. Do not assign the original checkpoint’s AP to them. All packaged variants pass strict loading and real-image CPU inference, which is a smoke test rather than an accuracy benchmark.

实验包仅压缩权重存储精度，解码器保持原始FP32。全量COCO精度与TensorRT一致性尚未验证，不能直接引用原始检查点AP。真实图片CPU推理通过不等于完整精度验证。不要调用`model.half()`。

## Use / 使用

From an installed repository checkout:

```bash
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('bear-b-coco-ema.pt.gz','rb'),open('bear-b-coco-ema.pt','wb'))"
python tools/predict.py --config configs/B.yml --checkpoint bear-b-coco-ema.pt \
  --classes configs/coco-classes.json --image assets/bear-input.jpg \
  --device cpu --output results/bear.json
```

For S substitute `configs/S.yml` and the S checkpoint. For GPU inference use `--device cuda --amp`. See the bilingual fine-tuning/export guides. TensorRT engines must be built on the target environment and accuracy-validated for the selected weights; old engines are not included.

S模型替换对应配置与文件名。GPU推理使用`--device cuda --amp`。COCO/YOLO微调及导出见仓库中英双语指南。Tiny仍在训练，本次不发布Tiny权重。

## License / 许可

The retained DEIMv2 non-commercial license and separate DINOv3 terms apply. Read LICENSE.md, THIRD_PARTY.md and the repository’s licenses directory before use. Downloading these assets does not grant commercial rights.

请遵循DEIMv2非商业许可与DINOv3单独条款；下载权重不代表取得商业授权。
