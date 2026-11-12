# Training sequence and publication

The active server queue is **Tiny → Base → Large**. Small remains stopped. The deleted Base servers are not resumed; Base starts fresh on the surviving training server using the matching DINOv3 ConvNeXt-B backbone. Large uses ConvNeXt-L. A Tiny detector checkpoint is not used to initialize Base or Large.

After every completed full COCO val2017 validation (5,000 images, input640), compare box mAP50:95 with that model’s global best. A strictly higher value resets the counter. Equal or lower values increment it. After **10 consecutive non-improving epochs**, preserve the exact best checkpoint, stop that stage, and start the next. Patience resets between models. Each model also has a100-epoch cap. Manual stop requests take precedence.

Tiny uses global batch64 and LR0.0005; fresh Base starts with the same settings. Large starts conservatively with global batch32 and LR0.00025 across8 RTX5090 GPUs. Both future stages require finite forward/backward GPU preflight at their actual per-GPU batch before DDP starts. GPU readiness and speed are not claimed before that check. Seed42; original FP32 parameters/optimizer, FP16 backbone/context/neck, entire decoder and matching/criterion FP32, dynamic loss scaling.

At stage completion, publish the selected best EMA weights, lossless gzip checksums, configuration, selected epoch/AP, and complete epoch JSON/CSV to GitHub Releases and this repository. Full optimizer/scaler resume checkpoints remain on the server. Packaging checks strict loading, finite real-image CPU predictions and gzip byte identity. AP is from the selected training validation; a separate post-packaging COCO or TensorRT rerun is only claimed if actually performed.

[Tiny preview](releases/TINY.md) · [Per-epoch JSON](../validation/training/t-statistics.json) · [CSV](../validation/training/t-statistics.csv) · [Machine-readable policy](../validation/training/training-policy.json).

This describes the operational server queue and monitor. The generic repository fine-tuning CLI does not automatically launch this multi-model sequence. DINOv3 backbone pretraining used external LVD-1689M data; detector supervision uses COCO train2017.
