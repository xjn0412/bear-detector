# 🐻 Bear Detector

[English](README.md) | [简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Weights](https://github.com/xjn0412/bear-detector/releases/tag/v0.1.0-weights) · [Docker](docs/DOCKER.md)

**A little heavy, just like a bear.** A general object detector named for the author’s love of bears and the model’s substantial size.

**Training status:** Tiny → Base → Large. After 10 consecutive full COCO validation epochs without a higher mAP50:95, save the best checkpoint and move to the next model; each stage is capped at 100 epochs. Tiny is training; fresh Base and Large are queued. Small remains stopped. [Training policy](docs/TRAINING_POLICY.md).

![Bear Detector detects its mascot](assets/bear-detected.jpg)

AI-generated image; the box and **0.9637 confidence** are actual **Base epoch41 EMA** predictions, PyTorch CPU FP32 at640. [Raw prediction and checkpoint hash](assets/bear-inference.json). An illustration, not a benchmark.

## Model comparison at640

COCO val2017 box AP50:95. **Bear rows are our measurements; other rows are author-reported results, not rerun here.** Training/pretraining and parameter-count conventions differ.

| Model | Parameters | AP50:95 | Source / pretraining distinction |
|---|---:|---:|---|
| Bear-B | 112.11M | **58.93** | Historical EMA epoch41; DINOv3 ConvNeXt-B backbone pretraining |
| Bear-S | 73.68M | **57.90** | EMA epoch57; DINOv3 ConvNeXt-S backbone pretraining |
| [DEIMv2-X](https://github.com/Intellindust-AI-Lab/DEIMv2) | 50.3M | 57.8 | Published DINOv3-based detector |
| [D-FINE-X](https://github.com/Peterande/D-FINE) | 62M | 55.8 | Published COCO row |
| D-FINE-X, Objects365→COCO | 62M | 59.3 | Additional detection pretraining |
| [RT-DETRv2-X](https://github.com/lyuwenyu/RT-DETR) | 76M | 54.3 | Published COCO row |
| [YOLO26-X](https://docs.ultralytics.com/models/yolo26/) | 55.7M | 57.5 | Default NMS head; Objects365 pretraining; fused/deployed parameters |

Bear is substantially heavier. This table does **not** establish a better accuracy/latency tradeoff. Our RTX5090 timings are separate from authors’ T4 timings. [Full comparison and source notes](docs/COMPARISON.md).

## Models and downloads

**M = one million parameters.** The Base AP above is a historical result from the previous run; its checkpoint is currently inaccessible. No65AP claim is made.

| Model | Parameters | Current availability |
|---|---:|---|
| Small | 73.68M | [Compact155.18MB](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-compact.pt.gz) · [Lossless274.17MB](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt.gz) |
| Tiny | 52.05M | [EMA epoch 58, AP 56.4597](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-ema.pt.gz) |
| Base | 112.11M | [EMA epoch 42, AP 59.0008](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-ema.pt.gz) |
| Large | 221.40M | [EMA epoch 24, AP 60.0728](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-ema.pt.gz) |

Small compact weights passed full COCO val2017 at640: **57.8904AP** in FP32 versus **57.8962** for the reference rerun. Maximum compression loss across FP32/AMP was **0.0096AP point**. Decompress before loading; keep the decoder FP32. Compact TensorRT accuracy remains unvalidated. [Checksums and usage](docs/WEIGHTS.md) · [Detailed model results and speed](docs/MODEL_ZOO.md).

Detector supervision used COCO train2017; DINOv3 backbone pretraining used external LVD-1689M data. This is not all-data COCO-only training or zero-shot COCO detection.

## Get started

The public NVIDIA fine-tuning image needs no registry login:

```bash
docker pull ghcr.io/xjn0412/bear-detector:finetune-cu130
```

| Task | Guide |
|---|---|
| Fine-tune with Docker, persistent mounts and Small weights | [Docker guide](docs/DOCKER.md) |
| Install from source and run offline checks | [Installation](docs/INSTALL.md) |
| Fine-tune any positive class count with COCO JSON or YOLO labels | [Fine-tuning](docs/FINETUNE.md) |
| Predict and evaluate in PyTorch | [Inference](docs/INFERENCE.md) |
| Export ONNX/TensorRT and check accuracy | [Export](docs/EXPORT.md) |
| Inspect every command and the project layout | [CLI reference](docs/TOOLS.md) · [Repository map](docs/REPOSITORY.md) |

The image passed build, dependency and CPU pipeline checks; GPU container backward/DDP still needs host-side verification. Weights and datasets are mounted separately. [All documentation](docs/README.md) · [Verification scope](docs/VERIFICATION.md).

## Architecture

DINOv3-pretrained **ConvNeXt CNN** → two stride32 context-attention blocks → convolutional multiscale neck with cheap P2→P3 detail → six-layer, three-level DEIMv2 query decoder,300 queries, fixed640 input. FP16 bulk forward; **entire decoder, predictions, matching and criterion FP32**. No ViT backbone in these configurations. [Architecture](docs/ARCHITECTURE.md).

This is a research integration derived from DEIMv2 / DEIM / D-FINE / RT-DETR, not an independently invented decoder. [Training history](docs/TRAINING_HISTORY.md) · [Benchmark methodology](docs/BENCHMARKS.md) · [Evidence index](validation/README.md).

## A personal note

This personal repository is also a keepsake of my seven years in computer vision. I can’t say I worked especially hard, but they were seven years of my life. I hope that, in time, I will find a goal in life.

— xjn0412

## License and acknowledgments

Upstream DEIMv2 code is **non-commercial**; DINOv3 weights have separate terms. This repository grants no commercial deployment rights. [LICENSE.md](LICENSE.md) · [THIRD_PARTY.md](THIRD_PARTY.md) · [Contribution guide](CONTRIBUTING.md).

Thanks to [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2), [DEIM](https://github.com/Intellindust-AI-Lab/DEIM), [D-FINE](https://github.com/Peterande/D-FINE), [RT-DETR](https://github.com/lyuwenyu/RT-DETR), [DINOv3](https://github.com/facebookresearch/dinov3) and [timm](https://github.com/huggingface/pytorch-image-models). Original notices remain in `source/` and `licenses/`.
