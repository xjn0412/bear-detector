# 🐻 Bear Detector v2

[English](README.md) | [简体中文](README.zh-CN.md) · [Stable v1](https://github.com/xjn0412/bear-detector/tree/main) · [Architecture and training](docs/models/ARCHITECTURE_V2.md)

**Full P2–P5 FPN/PAN, pooled four-scale context attention, and a four-scale query decoder.**

V2 is an experimental development branch. Implementation and numerical smoke checks are available; trained v2 weights and COCO AP are not available yet. Published v1 results and weights belong to [v1](https://github.com/xjn0412/bear-detector/tree/main).


**COCO training launched:** L → B → S → T,8 GPUs/model selected by measured4/8-GPU scaling. [Training policy and timing estimates](docs/benchmarks/V2_COCO_TRAINING.md).

| Component | V2 |
|---|---|
| Backbone | DINOv3-pretrained ConvNeXt T/S/B/L; four spatial scales |
| Context | Pool each scale to20×20;1600 tokens; two width256/8-head attention blocks |
| Neck | Full P2–P5 top-down FPN and bottom-up PAN, width256 |
| Decoder | Six layers,300 queries; four scales, sampling4/4/2/2 |
| Numerical protection | FP32 projections/context/decoder/losses; checked FP16 CNN/neck; dynamic scaling and persistent FP32 fallback |
| V2-L at640 | **783.99 GFLOPs/image;224.53M full /223.16M deployed parameters** |

The full-resolution detection maps remain160²/80²/40²/20²; only the context branch is pooled. FP32 is entered before sensitive operations. First-input inference comparison also checks finite AMP drift, since underflow can yield finite zeros. No universal overflow/underflow or accuracy-parity guarantee is claimed.

Start with [architecture, precision policy, initialization and training commands](docs/models/ARCHITECTURE_V2.md). Configurations: [Tiny](configs/V2-T.yml), [Small](configs/V2-S.yml), [Base](configs/V2-B.yml), [Large](configs/V2-L.yml). The default total training batch is8 with backbone gradient checkpointing. V1 detector checkpoints and existing published Docker images do not contain the new architecture.

[Verification evidence](validation/v2/README.md) · [Original documentation](docs/README.md) · [License](LICENSE.md) · [Upstream notices](THIRD_PARTY.md)

Bear Detector is named for the author's love of bears and is a keepsake of seven years in computer vision, with the hope of finding a goal in life. V2 continues the DEIMv2/DEIM/D-FINE/RT-DETR and DINOv3 integration; upstream non-commercial and weight-specific terms remain applicable.
