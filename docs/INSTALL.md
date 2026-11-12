# Installation

Use Linux, Python3.11 and a CUDA-capable NVIDIA GPU for training and GPU evaluation. CPU supports unit tests, initialization, PyTorch FP32 prediction and ONNX export. Run commands from the repository root. No custom CUDA extension compilation is required by these tools.

```bash
git clone https://github.com/xjn0412/coco640-convnext-detector.git
cd coco640-convnext-detector
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a matching CUDA-enabled PyTorch/torchvision pair from the [official selector](https://pytorch.org/get-started/locally/) suitable for your GPU/driver. RTX5090 requires a Blackwell-capable CUDA build. The measured host used **torch2.14.1+cu130 and torchvision0.29.1+cu130**, Python3.11.18. Do not install the stale torch2.5.1 pins from upstream DEIMv2 instructions for a 5090.

For example, when the CUDA13.0 wheel channel is appropriate for your system:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -c "import torch,torchvision,timm; print(torch.__version__,torchvision.__version__,timm.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
python tests/test_classes.py
python tests/test_sampling.py
```

The full measured version record is `validation/environment.json`; additional package versions used in source verification are in `constraints-tested.txt`. These record a tested environment, not a guarantee that historical wheels remain available for every OS. To reproduce the non-torch versions, add `-c constraints-tested.txt` to the requirements installation. CUDA/driver installation is outside this repository.

For CPU-only unit tests, substitute an official CPU torch/torchvision installation. CPU is not a training-performance target.

## Optional export dependencies

```bash
python -m pip install -r requirements-export.txt
# Install TensorRT for your CUDA platform using NVIDIA's instructions.
# Tested TensorRT10 API environment:
python -m pip install tensorrt-cu13==10.16.1.11
python -c "import onnx,tensorrt; print(onnx.__version__,tensorrt.__version__)"
```

The builder uses TensorRT10 weakly typed precision constraints. TensorRT11 and strongly typed network APIs are not verified. Engines are hardware/version specific; build on the deployment GPU. TensorRT is unnecessary for fine-tuning or PyTorch inference.

## Offline smoke test

Run the README smoke commands. They require no model download or real dataset. `--random-init` is explicitly for wiring tests. ONNX export can also be checked without a GPU:

```bash
python tools/export_engine.py --config runs/smoke/config.yml \
  --checkpoint runs/smoke/initial.pth --output exports/smoke
```

The ONNX graph is fixed at batch1×3×640×640, opset17, with `orig_target_sizes` int64 `[width,height]`. Tracer warnings about fixed image sizes/batch are expected; other shapes are unsupported.

Run the entire offline CPU pipeline with automatic temporary outputs:

```bash
python tools/verify_cpu.py --onnx --output results/cpu-verification.json
```
