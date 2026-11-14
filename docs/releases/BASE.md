# Bear Detector Base: final

Selected EMA: epoch **42**, full COCO val2017 (5,000 images), 640, **59.000843% mAP50–95**.

This stage finished under the 10-epoch patience / 100-epoch cap policy.

[Download original FP32 EMA weights (.pt)](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-ema.pt). Load directly and use configs/B.yml. Keep model parameters and the entire decoder FP32; use FP16 autocast in the backbone/neck.

Epoch JSON/CSV, config, checksums and packaging checks are attached. No quantization or FP16 storage conversion. Full optimizer/scaler resume states remain on the training server and are not included in this inference/fine-tuning archive.

The AP is the full selected training validation. Packaging passed strict loading, a real-image CPU finite-output check and gzip byte round-trip; no new full COCO packaging rerun or TensorRT test. DINOv3 used external LVD-1689M pretraining; detector supervision used COCO train2017. Retained upstream non-commercial and DINOv3 terms apply; see LICENSE.md and THIRD_PARTY.md.
