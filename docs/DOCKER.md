# Docker fine-tuning on NVIDIA GPUs

[English](DOCKER.md) | [简体中文](DOCKER.zh-CN.md)

Image: `ghcr.io/xjn0412/bear-detector:finetune-cu130` (Linux x86-64). It includes the model source, PyTorch2.14.1, torchvision0.29.1, CUDA13.0 wheel libraries, and pinned fine-tuning dependencies. Trained weights, datasets and TensorRT are not bundled. The runtime keeps FP16 bulk forward and the entire decoder FP32; do not use `model.half()`.

## Host requirements and access

Install Docker and NVIDIA Container Toolkit on a Linux NVIDIA host. A host NVIDIA driver compatible with CUDA13.0 is required; installing this image does not install or upgrade the host driver. Test GPU access before training:

```bash
# While the package is private, first run `docker login ghcr.io -u YOUR_GITHUB_USER`.
# Enter a classic GitHub PAT with read:packages at the password prompt.
# Your GitHub account must have access to the package/repository. Do not put tokens in commands.
docker pull ghcr.io/xjn0412/bear-detector:finetune-cu130
export BEAR_IMAGE=ghcr.io/xjn0412/bear-detector:finetune-cu130
docker run --rm --gpus all "$BEAR_IMAGE" python -c \
  "import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

The package remains access-controlled while the repository is private. For reproducibility use the immutable `sha-<full-source-commit>` tag or registry digest from the successful [image workflow](https://github.com/xjn0412/bear-detector/actions/workflows/container.yml), rather than relying on the moving `finetune-cu130` tag.

## Prepare persistent host files

Choose a real persistent host directory; **do not keep your only checkpoints in the container filesystem**. On cloud pods, a host bind mount is only as durable as the pod path it points to. On Lium, use an attached persistent Volume for saved runs (or copy checkpoints there/off-node regularly); `/var/lib` is not automatically persistent.

```bash
export BEAR_WORK="$PWD/bear-work"   # Replace with your persistent absolute directory.
mkdir -p "$BEAR_WORK/data" "$BEAR_WORK/weights" "$BEAR_WORK/runs"
# On the host, download the Small release (requires gh authentication while private):
gh release download v0.1.0-weights --repo xjn0412/bear-detector \
  --pattern bear-s-coco-compact.pt.gz --dir "$BEAR_WORK/weights"
gzip -dk "$BEAR_WORK/weights/bear-s-coco-compact.pt.gz"
```

Check the SHA256 against [WEIGHTS.md](WEIGHTS.md). Use `configs/S.yml` with Small weights. Tiny/Base configs cannot load Small checkpoints. Base pretrained detector weights are currently unavailable pending server recovery. Random initialization is only for wiring tests.

Put your YOLO dataset at `$BEAR_WORK/data/my_dataset/`, with `data.yaml`, `images/train`, `images/val`, `labels/train`, `labels/val`. In `data.yaml`, use `path: .` and relative splits as in [FINETUNE.md](FINETUNE.md); host absolute paths are not visible inside the container. Mounts below remain identical between preparation, training, evaluation and resume because the generated config records container paths.

## YOLO preparation and training

Preparation resets class heads for your class count and creates a fresh training run. Use a new output directory for each experiment.

```bash
docker run --rm --shm-size=16g \
  -v "$BEAR_WORK/data:/data:ro" -v "$BEAR_WORK/weights:/weights:ro" \
  -v "$BEAR_WORK/runs:/runs" "$BEAR_IMAGE" \
  python tools/prepare_finetune.py --config configs/S.yml \
  --checkpoint /weights/bear-s-coco-compact.pt --data /data/my_dataset/data.yaml \
  --output /runs/my_dataset --epochs 30 --batch 8 --lr 0.0001

docker run --rm --gpus all --shm-size=16g \
  -v "$BEAR_WORK/data:/data:ro" -v "$BEAR_WORK/runs:/runs" \
  "$BEAR_IMAGE" python tools/train_custom.py --run /runs/my_dataset
```

This single-process command uses one GPU. For four GPUs replace the final command with:

```bash
docker run --rm --gpus all --shm-size=16g \
  -v "$BEAR_WORK/data:/data:ro" -v "$BEAR_WORK/runs:/runs" \
  "$BEAR_IMAGE" torchrun --standalone --nproc_per_node=4 \
  tools/train_custom.py --run /runs/my_dataset
```

Total batch8 must be divisible by GPU count and fit the dataset. Change batch at preparation time, not when resuming. Shared-memory size16g is an example; choose a value fitting host RAM and worker count. All GPU work must have enough free VRAM; stop other workloads yourself before training.

## COCO JSON alternative

Replace `--data` in the preparation command with these four arguments, using your mounted layout:

```text
--train-json /data/my_dataset/train.json --val-json /data/my_dataset/val.json
--train-images /data/my_dataset/train --val-images /data/my_dataset/val
```

Both formats support arbitrary positive class counts, use the same training pipeline and preserve a class mapping. YOLO annotations do not imply compatibility with YOLO model weights.

## Resume and evaluate

Always mount the same dataset and run locations. Resume only that run's complete checkpoint:

```bash
docker run --rm --gpus all --shm-size=16g \
  -v "$BEAR_WORK/data:/data:ro" -v "$BEAR_WORK/runs:/runs" \
  "$BEAR_IMAGE" python tools/train_custom.py --run /runs/my_dataset \
  --resume /runs/my_dataset/training/last.pth

docker run --rm --gpus all --shm-size=16g \
  -v "$BEAR_WORK/data:/data:ro" -v "$BEAR_WORK/runs:/runs" \
  "$BEAR_IMAGE" python tools/evaluate.py --backend torch-fp32 \
  --config /runs/my_dataset/config.yml \
  --checkpoint /runs/my_dataset/training/best_stg1.pth \
  --annotations /runs/my_dataset/val.json --images /data/my_dataset/images/val \
  --output /runs/my_dataset/evaluation.json
```

The evaluation image path above is for the YOLO layout; COCO users substitute `/data/my_dataset/val`. Select the best stage by measured AP if `best_stg2.pth` exists. For distributed runs resume with the same `torchrun` invocation and append `--resume`. Custom training has the configured epoch cap; the operational COCO patience20 monitor is not bundled into this command. Back up the full `runs/my_dataset` directory, including config, normalized labels, class mapping and optimizer/EMA/scaler checkpoints.

## Verification and limits

Before pushing, GitHub Actions builds this exact CUDA image and runs its offline CPU pipeline: class/sampling/YOLO tests, random Tiny initialization, COCO and YOLO preparation, class remapping and PyTorch predictions, plus training/evaluation CLI checks and dependency checks. Evidence and installed versions are uploaded as `container-verification` workflow artifacts. No weights are downloaded for these checks.

Hosted CI has no GPU. These image checks do not claim Docker GPU backward/DDP execution, custom-dataset convergence, TensorRT accuracy or speed. The same runtime has separate GPU training evidence in [VERIFICATION.md](VERIFICATION.md); run the GPU availability command above on your own host. TensorRT export needs the additional dependencies and validation in [EXPORT.md](EXPORT.md).

For a local build: `docker build -t bear-detector:local .`. Retained DEIMv2 non-commercial and separate DINOv3 terms apply; see LICENSE.md and THIRD_PARTY.md.
