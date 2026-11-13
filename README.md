# 🐻 Bear Detector

[English](README.md) | [简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Weights](https://github.com/xjn0412/bear-detector/releases) · [Docker](docs/getting-started/DOCKER.md)

**A little heavy, just like a bear.** A general object detector named for the author’s love of bears and the model’s substantial size.

**Training complete (2026-11-14):** selected EMA weights for Tiny, Small, Base and Large are available. Tiny / Base / Large finished at 68 / 52 / 44 epochs; Small was manually stopped at 68. [Training policy](docs/benchmarks/TRAINING_POLICY.md).

![Bear Detector detects its mascot](assets/bear-detected.jpg)

AI-generated image; the box and **0.9637 confidence** are actual **Base epoch41 EMA** predictions, PyTorch CPU FP32 at640. [Raw prediction and checkpoint hash](assets/bear-inference.json). An illustration, not a benchmark.

## Performance · COCO val2017 / RTX5090

Recommended mixed-storage downloads use **FP16 bulk weights + the original FP32 decoder**. Fresh full5,000-image COCO validation at640, selected EMA, no additional training. PyTorch and TensorRT AP are reported separately.

| Model | Input | AP50:95 (PT / TRT) | Params M (full / deploy) | GFLOPs | PyTorch AMP ms | TensorRT ms | Mixed MB |
|---|---:|---:|---:|---:|---:|---:|---|
| [Bear-T](configs/T.yml) | 640 | 56.4851 / 56.4449 | 52.05 / 51.10 | 132.18 | 12.84 | 2.75 | [115.08](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt.gz) |
| [Bear-S](configs/S.yml) | 640 | 57.8864 / 57.8563 | 73.68 / 72.73 | 201.21 | 16.44 | 3.34 | [155.18](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-mixed.pt.gz) |
| [Bear-B](configs/B.yml) | 640 | 59.0192 / 58.9243 | 112.11 / 111.16 | 310.75 | 15.69 | 3.82 | [226.34](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-mixed.pt.gz) |
| [Bear-L](configs/L.yml) | 640 | 60.0748 / 60.0236 | 221.40 / 220.45 | 622.32 | 15.20 | 4.79 | [428.79](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-mixed.pt.gz) |

Latency: one RTX5090 GPU7, batch1,100 warmups,3×300 CUDA-event iterations, no CUDA Graph; GPU inputs through top300, excluding preprocessing/host transfers. TRT includes device-input copies. Mining/training were stopped; all builds finished before timing. PyTorch uses FP16 bulk+entire FP32 decoder; TensorRT10.13.3.9 also protects decoder and sensitive normalization/reductions.

PT accuracy uses batch8; TRT is fixed batch1, so AP differences include backend and batch-dependent numerics. Each acceptance gate permits at most0.1 AP-point loss. GFLOPs use the documented arithmetic scope,2 FLOPs/MAC, and do not predict latency. [Full results and reproduction](docs/benchmarks/DEPLOYMENT5090.md) · [FLOPs scope](docs/benchmarks/COMPLEXITY.md).

<details>
<summary>Comparison with other detectors (author-reported, not rerun)</summary>

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

Pretraining, parameter-count conventions and backends differ. Bear is heavier; our RTX5090 timings cannot be divided by official T4 timings to claim speedups. [Comparison notes](docs/benchmarks/COMPARISON.md).

</details>

## Models and downloads

Mixed storage is now the default for every size. Decompress, load into an FP32 model and use AMP with the entire decoder FP32. **Do not call `model.half()`**. Mixed checkpoints also support fine-tuning: preparation resets class heads and initializes FP32 parameters with a fresh optimizer.

| Model | Parameters | Selected epoch | Default mixed package |
|---|---:|---:|---|
| Tiny | 52.05M | 58 | [115.08 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt.gz) |
| Small | 73.68M | 57 | [155.18 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-mixed.pt.gz) |
| Base | 112.11M | 42 | [226.34 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-mixed.pt.gz) |
| Large | 221.40M | 24 | [428.79 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-mixed.pt.gz) |

<details>
<summary>Original FP32 EMA archives (exact source / optional fine-tuning initialization)</summary>

[Tiny · 193.77 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-ema.pt.gz) · [Small · 274.17 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt.gz) · [Base · 416.91 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-ema.pt.gz) · [Large · 822.79 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-ema.pt.gz)

Originals are retained because FP16 rounding is irreversible. These inference archives do not include optimizer/scaler resume state.

</details>

[SHA256 & usage](docs/models/WEIGHTS.md) · [Epoch history](docs/benchmarks/TRAINING_HISTORY.md)

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
