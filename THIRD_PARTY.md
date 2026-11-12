# Attribution and license scope

This repository is a modified research integration, not a new implementation of every detector component. The vendored `source/` retains upstream file-level copyright notices and the exact DEIMv2 license at `source/LICENSE.md` (also copied to root `LICENSE.md`). Its non-commercial restriction is material: commercial deployment requires appropriate upstream permission. Do not describe this repository as MIT/Apache licensed as a whole.

| Material | Source / governing terms |
|---|---|
| Decoder, neck, criterion, training/data framework | [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2), retained LICENSE.md |
| Inherited detector components | [DEIM](https://github.com/Intellindust-AI-Lab/DEIM), [D-FINE](https://github.com/Peterande/D-FINE), [RT-DETR](https://github.com/lyuwenyu/RT-DETR); retained notices, respective upstream licenses |
| Pretrained backbone weights | [DINOv3](https://github.com/facebookresearch/dinov3), [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license) |
| Backbone implementation/distribution | [timm](https://github.com/huggingface/pytorch-image-models), Apache2.0; [Base model card](https://huggingface.co/timm/convnext_base.dinov3_lvd1689m) |
| Dataset | [COCO](https://cocodataset.org/), annotation and image terms apply separately |

Modifications: ConvNeXt context/detail backbone integration; shared B/S/T configurations; runtime finite guards and decoderFP32 safety; strict arbitrary-class transfer/schema tooling; inference/evaluation/export utilities; portable bilinear sampling for TensorRT; documentation and tests. These modifications do not remove third-party restrictions. No pretrained weights, data images or TensorRT engines are committed in this initial source release.
