# Bear Detector Small pretrained EMA weights / Small预训练权重

This release includes Small only. Base validation is pending; Tiny is training.

Full COCO val2017 (5,000 images), 640, EMA selected epoch57:

| PyTorch precision | Original AP50:95 | Compact AP50:95 | Difference |
|---|---:|---:|---:|
| fp32 | 57.89617 | 57.89035 | -0.00581 |
| amp | 57.88758 | 57.87800 | -0.00959 |

The compact checkpoint passed the preset maximum0.1AP-point drop in both modes, finite-output checks and strict loading. AMP means FP16 bulk and entire FP32 decoder. Compact storage preserves the original FP32 decoder tensors; load into a normal FP32 model, never model.half(). The reference EMA archive is lossless. Full optimizer/scaler resume states are not bundled.

紧凑包通过全量COCO双精度模式、输出有限值及加载检查；解码器保持原始FP32，加载到FP32模型中，不要调用model.half()。参考包为无损压缩；未附完整训练续训状态。

Download either archive and decompress before using repository tools:

    python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('bear-s-coco-compact.pt.gz','rb'),open('bear-s-coco-compact.pt','wb'))"
    python tools/predict.py --config configs/S.yml --checkpoint bear-s-coco-compact.pt --classes configs/coco-classes.json --image assets/bear-input.jpg --device cuda --amp --output results/bear.json

For the reference substitute bear-s-coco-ema.pt(.gz). CPU uses --device cpu without --amp. COCO/YOLO fine-tuning and export instructions are in the bilingual repository guides.

TensorRT compact parity is NOT validated; build a fresh engine and validate its accuracy before deployment. No timing claim: validation ran alongside training. COCO parity does not guarantee parity on every custom dataset.

紧凑权重的TensorRT一致性尚未验证，需重新导出并验证精度；旧引擎不能复用。本次与训练并行，不用于速度比较。自定义数据集仍需单独验证。

Detector supervision: COCO train2017; DINOv3 backbone used external pretraining. Not all-data COCO-only. DEIMv2 non-commercial and separate DINOv3 terms apply. Read LICENSE.md and THIRD_PARTY.md; no commercial rights granted. 请遵循上游非商业许可与DINOv3条款。
