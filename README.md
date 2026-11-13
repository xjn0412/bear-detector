# 🐻 Bear Detector

[English](README.md) | [简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Weights](https://github.com/xjn0412/bear-detector/releases) · [Docker](docs/getting-started/DOCKER.md)

**A little heavy, just like a bear.** A general object detector named for the author’s love of bears and the model’s substantial size.

**Training complete (2026-11-14):** selected EMA weights for Tiny, Small, Base and Large are available. Tiny / Base / Large finished at 68 / 52 / 44 epochs; Small was manually stopped at 68. [Training policy](docs/benchmarks/TRAINING_POLICY.md).

![Bear Detector detects its mascot](assets/bear-detected.jpg)

AI-generated image; the box and **0.9637 confidence** are actual **Base epoch41 EMA** predictions, PyTorch CPU FP32 at640. [Raw prediction and checkpoint hash](assets/bear-inference.json). An illustration, not a benchmark.

## Model comparison at640

COCO val2017 box AP50:95. **Bear rows are our measurements; other rows are author-reported results, not rerun here.** Training/pretraining and parameter-count conventions differ.

| Model | Parameters | AP50:95 | Source / pretraining distinction |
|---|---:|---:|---|
| Bear-T | 52.05M | **56.4597** | Our full validation; EMA epoch 58 |
| Bear-S | 73.68M | **57.8987** | Our full validation; EMA epoch 57 |
| Bear-B | 112.11M | **59.0008** | Our full validation; EMA epoch 42 |
| Bear-L | 221.40M | **60.0728** | Our full validation; EMA epoch 24 |
| [DEIMv2-X](https://github.com/Intellindust-AI-Lab/DEIMv2) | 50.3M | 57.8 | Published DINOv3-based detector |
| [D-FINE-X](https://github.com/Peterande/D-FINE) | 62M | 55.8 | Published COCO row |
| D-FINE-X, Objects365→COCO | 62M | 59.3 | Additional detection pretraining |
| [RT-DETRv2-X](https://github.com/lyuwenyu/RT-DETR) | 76M | 54.3 | Published COCO row |
| [YOLO26-X](https://docs.ultralytics.com/models/yolo26/) | 55.7M | 57.5 | Default NMS head; Objects365 pretraining; fused/deployed parameters |

Bear is substantially heavier. This table does **not** establish a better accuracy/latency tradeoff. Our RTX5090 timings are separate from authors’ T4 timings. [Full comparison and source notes](docs/benchmarks/COMPARISON.md).

## Models and downloads

**M = one million parameters.** Every download uses original FP32 EMA tensors in lossless gzip. Decompress before loading; optimizer/scaler resume states are not included.

| Model | Parameters | AP50:95 | Best epoch | Download |
|---|---:|---:|---:|---|
| Tiny | 52.05M | 56.4597 | 58 | [EMA · 193.77 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-ema.pt.gz) |
| Small | 73.68M | 57.8987 | 57 | [EMA · 274.17 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt.gz) |
| Base | 112.11M | 59.0008 | 42 | [EMA · 416.91 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-ema.pt.gz) |
| Large | 221.40M | 60.0728 | 24 | [EMA · 822.79 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-ema.pt.gz) |

[SHA256 & usage](docs/models/WEIGHTS.md) · [Model zoo](docs/models/MODEL_ZOO.md) · [Epoch history](docs/benchmarks/TRAINING_HISTORY.md)

Detector supervision used COCO train2017; DINOv3 backbone pretraining used external LVD-1689M data. This is not all-data COCO-only training or zero-shot COCO detection.

## Get started

The public NVIDIA fine-tuning image needs no registry login:

```bash
docker pull ghcr.io/xjn0412/bear-detector:finetune-cu130
```

| Task | Guide |
|---|---|
| Fine-tune with Docker, persistent mounts and Small weights | [Docker guide](docs/getting-started/DOCKER.md) |
| Install from source and run offline checks | [Installation](docs/getting-started/INSTALL.md) |
| Fine-tune any positive class count with COCO JSON or YOLO labels | [Fine-tuning](docs/guides/FINETUNE.md) |
| Predict and evaluate in PyTorch | [Inference](docs/guides/INFERENCE.md) |
| Export ONNX/TensorRT and check accuracy | [Export](docs/guides/EXPORT.md) |
| Inspect every command and the project layout | [CLI reference](docs/reference/TOOLS.md) · [Repository map](docs/reference/REPOSITORY.md) |

The image passed build, dependency and CPU pipeline checks; GPU container backward/DDP still needs host-side verification. Weights and datasets are mounted separately. [All documentation](docs/README.md) · [Verification scope](docs/benchmarks/VERIFICATION.md).

## Architecture

DINOv3-pretrained **ConvNeXt CNN** → two stride32 context-attention blocks → convolutional multiscale neck with cheap P2→P3 detail → six-layer, three-level DEIMv2 query decoder,300 queries, fixed640 input. FP16 bulk forward; **entire decoder, predictions, matching and criterion FP32**. No ViT backbone in these configurations. [Architecture](docs/models/ARCHITECTURE.md).

This is a research integration derived from DEIMv2 / DEIM / D-FINE / RT-DETR, not an independently invented decoder. [Training history](docs/benchmarks/TRAINING_HISTORY.md) · [Benchmark methodology](docs/benchmarks/BENCHMARKS.md) · [Evidence index](validation/README.md).

## A personal note

This personal repository is also a keepsake of my seven years in computer vision. I can’t say I worked especially hard, but they were seven years of my life. I hope that, in time, I will find a goal in life.

— xjn0412

## License and acknowledgments

Upstream DEIMv2 code is **non-commercial**; DINOv3 weights have separate terms. This repository grants no commercial deployment rights. [LICENSE.md](LICENSE.md) · [THIRD_PARTY.md](THIRD_PARTY.md) · [Contribution guide](CONTRIBUTING.md).

Thanks to [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2), [DEIM](https://github.com/Intellindust-AI-Lab/DEIM), [D-FINE](https://github.com/Peterande/D-FINE), [RT-DETR](https://github.com/lyuwenyu/RT-DETR), [DINOv3](https://github.com/facebookresearch/dinov3) and [timm](https://github.com/huggingface/pytorch-image-models). Original notices remain in `source/` and `licenses/`.
