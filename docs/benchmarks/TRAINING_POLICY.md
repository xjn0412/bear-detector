# Completed COCO640 training policy · 已完成训练策略

The authorized Tiny → fresh Base → Large sequence is complete; no stage remains queued. Small was manually stopped. / Tiny → 全新Base → Large已全部完成，Small由作者停止。

| Model | Completed epochs | Best epoch | Best AP50:95 | Final stopping rule |
|---|---:|---:|---:|---|
| Tiny | 68 | 58 | 56.4597 | 10 non-improving validations |
| Small | 68 | 57 | 57.8987 | Manual stop |
| Base | 52 | 42 | 59.0008 | 10 non-improving validations |
| Large | 44 | 24 | 60.0728 | 20 non-improving validations |

A strictly higher global-best full COCO val2017 box AP resets patience; ties count as no improvement (min_delta=0). Maximum100 epochs. Large patience was extended from10 to15 then20 without resetting its counter. No automatic continuation is authorized after completion.

Fixed input640, seed42, dynamic GradScaler, FP32 parameters/optimizer, FP16 backbone/context/neck and entire FP32 decoder/matching/criterion. Tiny/Base global batch64 LR0.0005; Large batch32 LR0.00025. Each fresh stage passed actual-batch finite criterion/backward preflight before DDP. Base/Large start from corresponding DINOv3 backbones plus fresh detector, not Tiny weights.

Selected exact full best checkpoints retain model/EMA/optimizer/scaler on the server. Public releases contain selected original FP32 EMA in lossless gzip, config, checksums and epoch histories. Packaging strict-loading/finite real-image CPU/gzip checks do not constitute a new full COCO or TensorRT validation.

This records the completed research queue, not an automatic multi-model feature of the fine-tuning CLI. Detector supervision is COCO train2017; DINOv3 pretraining used external data. [Epoch history](TRAINING_HISTORY.md) · [Model zoo](../models/MODEL_ZOO.md).
