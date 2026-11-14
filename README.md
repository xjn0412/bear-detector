# 🐻 Bear Detector

[English](README.md) | [简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Weights](https://github.com/xjn0412/bear-detector/releases) · [Docker](docs/getting-started/DOCKER.md)

**A little heavy, just like a bear.** A general object detector named for the author’s love of bears and the model’s substantial size.

**Training complete (2026-11-14):** selected EMA weights for Tiny, Small, Base and Large are available. Tiny / Base / Large finished at 68 / 52 / 44 epochs; Small was manually stopped at 68. [Training policy](docs/benchmarks/TRAINING_POLICY.md).

![Bear Detector detects its mascot](assets/bear-detected.jpg)

AI-generated image; the box and **0.9637 confidence** are actual **Base epoch41 EMA** predictions, PyTorch CPU FP32 at640. [Raw prediction and checkpoint hash](assets/bear-inference.json). An illustration, not a benchmark.

## Performance · COCO val2017 / RTX5090

Recommended mixed-storage downloads use **FP16 bulk weights + the original FP32 decoder**. Fresh full5,000-image COCO validation at640, selected EMA, no additional training. PyTorch and TensorRT AP are reported separately.

| Model | Input | AP50:95 (PT / TRT) | Params M (full / deploy) | GFLOPs | PyTorch AMP ms | TensorRT ms | Mixed .pt MB |
|---|---:|---:|---:|---:|---:|---:|---|
| [Bear-T](configs/T.yml) | 640 | 56.4851 / 56.4449 | 52.05 / 51.10 | 132.18 | 12.84 | 2.75 | [124.43](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt) |
| [Bear-S](configs/S.yml) | 640 | 57.8864 / 57.8563 | 73.68 / 72.73 | 201.21 | 16.44 | 3.34 | [167.75](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-mixed.pt) |
| [Bear-B](configs/B.yml) | 640 | 59.0192 / 58.9243 | 112.11 / 111.16 | 310.75 | 15.69 | 3.82 | [244.60](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-mixed.pt) |
| [Bear-L](configs/L.yml) | 640 | 60.0748 / 60.0236 | 221.40 / 220.45 | 622.32 | 15.20 | 4.79 | [463.18](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-mixed.pt) |

Latency: one RTX5090 GPU7, batch1,100 warmups,3×300 CUDA-event iterations, no CUDA Graph; GPU inputs through top300, excluding preprocessing/host transfers. TRT includes device-input copies. Mining/training were stopped; all builds finished before timing. PyTorch uses FP16 bulk+entire FP32 decoder; TensorRT10.13.3.9 also protects decoder and sensitive normalization/reductions.

PT accuracy uses batch8; TRT is fixed batch1, so AP differences include backend and batch-dependent numerics. Each acceptance gate permits at most0.1 AP-point loss. GFLOPs use the documented arithmetic scope,2 FLOPs/MAC, and do not predict latency. [Full results and reproduction](docs/benchmarks/DEPLOYMENT5090.md) · [FLOPs scope](docs/benchmarks/COMPLEXITY.md).

## Comparison graphs and analysis

![Accuracy versus parameters and FLOPs](assets/benchmarks/comparison-efficiency.png)

![Selected leaderboard results](assets/benchmarks/comparison-leaderboard.png)

**Bear-L reaches60.07 AP at640; Bear-B reaches59.02.** Higher accuracy comes with substantial model cost: Base uses about2× YOLO26-x's published deployment parameters for +1.52AP. Bear-S is close to DEIMv2-X in AP but larger. The linked SOTA2 page's leading displayed result is64.6; Bear-L is4.53AP behind, under a different resolution/pretraining recipe. External results are author-reported; these figures establish no official rank or same-hardware speed advantage.

The dashed YOLO27 curve is an **unreleased official preview**, not a reproduced benchmark.

[Full tables, source links, object-size analysis and measured Bear latency](docs/benchmarks/COMPARISON.md) · [SOTA2 leaderboard](https://www.sota2.com/research/sota/object-detection-on-coco-2017-val). Presentation inspired by [Ultralytics](https://github.com/ultralytics/ultralytics).


## Models and downloads

Mixed storage is now the default for every size. Download the .pt file, load into an FP32 model and use AMP with the entire decoder FP32. **Do not call `model.half()`**. Mixed checkpoints also support fine-tuning: preparation resets class heads and initializes FP32 parameters with a fresh optimizer.

| Model | Parameters | Selected epoch | Default mixed package |
|---|---:|---:|---|
| Tiny | 52.05M | 58 | [124.43 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-mixed.pt) |
| Small | 73.68M | 57 | [167.75 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-mixed.pt) |
| Base | 112.11M | 42 | [244.60 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-mixed.pt) |
| Large | 221.40M | 24 | [463.18 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-mixed.pt) |

<details>
<summary>Original FP32 EMA archives (exact source / optional fine-tuning initialization)</summary>

[Tiny · 208.86 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.2.0-tiny/bear-t-coco-ema.pt) · [Small · 295.46 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt) · [Base · 449.16 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.2.1-base/bear-b-coco-ema.pt) · [Large · 886.32 MB](https://github.com/xjn0412/bear-detector/releases/download/v0.3.0-large/bear-l-coco-ema.pt)

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
