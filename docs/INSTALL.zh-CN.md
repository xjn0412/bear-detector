# 安装与自检

[English](INSTALL.md) | [简体中文](INSTALL.zh-CN.md)

建议Linux、Python3.11；GPU训练/评估面向NVIDIA。CPU支持初始化、单元测试、FP32图片预测与ONNX导出，不是训练速度目标。

```bash
git clone https://github.com/xjn0412/bear-detector.git
cd bear-detector
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

从PyTorch官方渠道安装与驱动/GPU匹配的torch与torchvision。5090需要支持Blackwell的CUDA构建。实测环境为Python3.11.18、torch2.14.1+cu130、torchvision0.29.1+cu130；不要照搬上游旧版torch2.5.1配置。适合CUDA13.0时的示例：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python tests/test_classes.py
python tests/test_sampling.py
python tests/test_yolo.py
```

CPU自检可改装官方CPU版torch/torchvision。`constraints-tested.txt`记录实测非torch版本，不保证历史wheel始终可用于所有平台；驱动/CUDA安装不在仓库范围。私有仓库克隆需要GitHub访问权限。

可选导出依赖：

```bash
python -m pip install -r requirements-export.txt
# 按NVIDIA官方说明安装与你平台匹配的TensorRT；实测API版本：
python -m pip install tensorrt-cu13==10.16.1.11
python tools/verify_cpu.py --onnx --output results/cpu-verification.json
```

微调和PyTorch推理不需要TensorRT。离线CPU自检不下载权重，用临时目录验证COCO/YOLO准备、CPU预测与ONNX构建；合成数据AP不代表精度。ONNX固定batch1/640，相关固定shape tracing警告可预期。
