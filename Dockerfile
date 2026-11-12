FROM python:3.11-slim-bookworm
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="Bear Detector fine-tuning" \
      org.opencontainers.image.source="https://github.com/xjn0412/bear-detector" \
      org.opencontainers.image.revision=$SOURCE_REVISION \
      org.opencontainers.image.description="NVIDIA fine-tuning; FP16 bulk with FP32 decoder. Upstream non-commercial terms apply."
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    HF_HOME=/cache/huggingface TORCH_HOME=/cache/torch
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/bear-detector
COPY requirements.txt constraints-tested.txt ./
RUN python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130 \
    && python -m pip install -r requirements.txt -c constraints-tested.txt \
    && python -m pip check
COPY . .
RUN mkdir -p /cache/huggingface /cache/torch \
    && python -m compileall -q backbone.py runtime.py compat.py tools \
    && python -c "import torch,torchvision,timm; assert torch.__version__.startswith('2.14.1'); assert torch.version.cuda == '13.0'; print(torch.__version__,torchvision.__version__,timm.__version__)"
CMD ["bash"]
