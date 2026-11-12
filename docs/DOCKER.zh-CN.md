# NVIDIA GPU Docker微调

[English](DOCKER.md) | [简体中文](DOCKER.zh-CN.md)

镜像：`ghcr.io/xjn0412/bear-detector:finetune-cu130`，Linux x86-64。包含源码、PyTorch2.14.1、torchvision0.29.1、CUDA13.0 wheel运行库及固定微调依赖。不含数据、训练权重或TensorRT。保持主体FP16、整个解码器FP32，不要调用`model.half()`。

## 主机与访问要求

主机需安装Docker、NVIDIA Container Toolkit以及兼容CUDA13.0的NVIDIA驱动。镜像不会安装/升级主机驱动。仓库和包私有期间，先执行`docker login ghcr.io -u YOUR_GITHUB_USER`，在密码提示中输入具备`read:packages`的classic PAT；账号需有仓库/包访问权限，不要把token直接写进命令。

```bash
docker pull ghcr.io/xjn0412/bear-detector:finetune-cu130
export BEAR_IMAGE=ghcr.io/xjn0412/bear-detector:finetune-cu130
docker run --rm --gpus all "$BEAR_IMAGE" python -c \
  "import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

复现实验请使用成功[构建任务](https://github.com/xjn0412/bear-detector/actions/workflows/container.yml)记录的`sha-<完整源码commit>`标签或镜像digest；`finetune-cu130`会随更新移动。

## 持久化数据与权重

**不要把检查点只写在容器内部。** 主机挂载目录也必须真实持久化；Lium保存训练结果应使用已挂载持久Volume，或定期将检查点复制到Volume/其他服务器。`/var/lib`不会自动持久化。

```bash
export BEAR_WORK="$PWD/bear-work"  # 换成你的持久化绝对路径
mkdir -p "$BEAR_WORK/data" "$BEAR_WORK/weights" "$BEAR_WORK/runs"
gh release download v0.1.0-weights --repo xjn0412/bear-detector \
  --pattern bear-s-coco-compact.pt.gz --dir "$BEAR_WORK/weights"
gzip -dk "$BEAR_WORK/weights/bear-s-coco-compact.pt.gz"
```

按[权重页](WEIGHTS.zh-CN.md)核对SHA256。Small权重必须配`configs/S.yml`，不能配T/B。Base预训练检测权重当前待服务器恢复。随机初始化只用于代码自检。

YOLO数据放在`$BEAR_WORK/data/my_dataset/`，包含`data.yaml`、`images/train`、`images/val`、`labels/train`、`labels/val`。YAML使用`path: .`和相对路径；主机绝对路径在容器里不可见。准备、训练、恢复、验证时保持相同容器挂载路径。

## YOLO准备与训练

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

准备时自动重置分类头并适配任意正整数类别，创建新优化器/EMA/scaler。每个实验使用新的输出目录。上面为单进程单GPU；四卡改为：

```bash
docker run --rm --gpus all --shm-size=16g \
  -v "$BEAR_WORK/data:/data:ro" -v "$BEAR_WORK/runs:/runs" \
  "$BEAR_IMAGE" torchrun --standalone --nproc_per_node=4 \
  tools/train_custom.py --run /runs/my_dataset
```

总batch8需能被GPU数整除，且不超过训练图片数。16g共享内存为示例，按主机内存和worker数调整。恢复时不改batch/类别/配置；训练前确保显存空闲。

## COCO JSON

准备命令的`--data`替换为四个参数即可：

```text
--train-json /data/my_dataset/train.json --val-json /data/my_dataset/val.json
--train-images /data/my_dataset/train --val-images /data/my_dataset/val
```

两种标注进入相同训练管线，保存原类别映射。支持YOLO标注不代表可加载YOLO模型权重。

## 恢复与验证

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

验证示例为YOLO图片目录，COCO布局替换成`/data/my_dataset/val`。若有`best_stg2.pth`，按实际AP选最佳阶段。DDP恢复使用原来的torchrun命令并追加`--resume`。自定义微调按配置轮数结束，不包含线上COCO patience20监控器。备份整个run目录，包括配置、标注、类别映射及完整检查点。

## 验证范围

推送前GitHub Actions实际构建CUDA镜像，在镜像内部运行CPU离线自检：类别/采样/YOLO测试、随机Tiny初始化、COCO/YOLO准备、类别适配和PyTorch预测，以及依赖和训练/验证CLI检查。环境与结果保存为构建任务的`container-verification`附件，不下载权重。

GitHub托管runner没有GPU，不能把CPU检查当作镜像内GPU反向/DDP、自定义任务收敛、TensorRT精度或速度验证。相同源码已有单独GPU训练记录，见[验证页](VERIFICATION.zh-CN.md)；目标主机先执行上面的GPU可用性命令。TensorRT需另装依赖并按[导出指南](EXPORT.zh-CN.md)验证。

本地构建：`docker build -t bear-detector:local .`。请遵循DEIMv2非商业许可与DINOv3单独条款。
