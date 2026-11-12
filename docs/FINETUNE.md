# Fine-tuning on your dataset

[English](FINETUNE.md) | [简体中文](FINETUNE.zh-CN.md)

## Data contract

Provide COCO detection JSON files and corresponding RGB images. `images` need unique `id`, `file_name`, `width`, `height`; annotations need unique `id`, valid `image_id`, `category_id`, and `[x,y,width,height]` pixel boxes. Use finite, positive-sized boxes within the image. `categories` contain unique integer IDs and unique names; sparse IDs such as3/17/91 are supported. Training categories define the schema. Validation can lack instances of some classes but cannot rename IDs or introduce new ones. Use separate held-out images.

Preparation writes normalized zero-based annotation copies and `classes.json`, preserving original IDs and names. The decoder emits contiguous class indices. Use `classes.json` to recover dataset IDs/names. Naming a class does not create zero-shot detection capability.

## 1. Choose initialization

For transfer learning, supply a compatible trained **full detector checkpoint**. Match `configs/B.yml`, `S.yml` or `T.yml` to its backbone. Upstream DEIMv2 weights for a different architecture will fail strict loading. Trained detector downloads are not bundled in this code release.

Alternatively initialize from the public matching DINOv3 CNN backbone:

```bash
python tools/initialize.py --config configs/T.yml --pretrained --output weights/t_initial.pt
```

This uses `timm`'s `convnext_tiny.dinov3_lvd1689m` weights. Download requires network access and compliance with the DINOv3 terms. For offline use, download the matching timm `model.safetensors` separately and run:

```bash
python tools/initialize.py --config configs/T.yml \
  --backbone-weights weights/model.safetensors --output weights/t_initial.pt
```

Only the backbone is pretrained. The detector layers are fresh; it will need substantially more training than fine-tuning a trained detector and does not inherit the published COCO detector AP. Use `--random-init` only for a smoke test.

## 2. Prepare and train

Example for Tiny, with your data stored under `data/my_dataset`:

```bash
python tools/prepare_finetune.py --config configs/T.yml --checkpoint weights/t_initial.pt \
  --train-json data/my_dataset/train.json --val-json data/my_dataset/val.json \
  --train-images data/my_dataset/train --val-images data/my_dataset/val \
  --output runs/my_dataset --epochs 30 --batch 8 --lr 0.0001
python tools/train_custom.py --run runs/my_dataset
```

For a trained detector, replace the checkpoint path. Preparation resets **all** classification heads and denoising class embeddings, even when class counts match, and creates fresh optimizer/EMA/scaler state. It preserves all other compatible tensors strictly. Never use `--resume` to adapt to a different class schema.

`--epochs` must be at least2; total batch must fit the number of training images and be divisible by GPU count. Small data can overfit;30 epochs and1e-4 are starting examples, not universal recommendations. Custom defaults disable cached mosaic/mixup/copyblend and scale warm-up to available steps. Keep the FP32 decoder and numerical guards. Do not use `source/train.py` directly; the wrapper installs the required runtime.

Multiple GPUs, total batch8 across4 GPUs:

```bash
torchrun --standalone --nproc_per_node=4 tools/train_custom.py --run runs/my_dataset
```

Use `CUDA_VISIBLE_DEVICES` to select the GPUs. Leave CPU affinity disabled unless you have measured the correct topology.

## 3. Resume the same run

```bash
python tools/train_custom.py --run runs/my_dataset --resume runs/my_dataset/training/last.pth
```

Resume keeps raw weights, EMA, optimizer, scheduler, scaler and epoch. The file must belong to that run's training directory. Keep the prepared config and schema unchanged. Custom training runs for its configured epoch limit; the operational COCO patience20 controller is **not** part of this standalone fine-tuning wrapper.

Upstream full checkpoints include `last.pth`, `best_stg1.pth` and possibly `best_stg2.pth`. They contain raw model weights and an EMA module. Compare their validation AP and select the best across stages; do not assume stage2 is automatically better. `tools/common.py` explicitly selects EMA for transfer/inference/export. The operational COCO monitor's `best.pt` is a full exact best-epoch checkpoint; its inference weights are the EMA module, while raw training weights remain available for resume.

## 4. Validate

```bash
python tools/evaluate.py --backend torch-fp32 --config runs/my_dataset/config.yml \
  --checkpoint runs/my_dataset/training/best_stg1.pth \
  --annotations runs/my_dataset/val.json --images data/my_dataset/val \
  --output results/my_dataset-fp32.json
```

Use the normalized validation JSON produced by preparation. Keep the complete category list, including classes without validation instances. See the export guide for backend parity checks. Smoke tests exercise code paths; they do not establish convergence or accuracy on arbitrary real datasets.

## YOLO detection annotations

Use the usual local layout: `images/train`, `images/val`, `labels/train`, `labels/val`. A label text file matches each image stem and contains one `class cx cy width height` row per object, normalized to[0,1], with zero-based classes. Missing or empty text files mean background; add `--strict-labels` to reject missing files. Boxes must have positive size and remain inside the image. Segmentation polygons, pose, OBB, NDJSON and remote image URLs are not supported by this converter.

Example `data/my_dataset/data.yaml`:

```yaml
path: .
train: images/train
val: images/val
names:
  0: person
  1: bear
```

A relative `path` is resolved **relative to the YAML file** (not a machine-specific Ultralytics dataset setting). Splits accept a directory, image, UTF-8 image list, or list of these. In text lists, `./` entries resolve relative to that list; other relative entries resolve from the dataset root. Images need an `images/` path component; labels replace its last occurrence with `labels/`. Class IDs must be contiguous0..N−1; names may be a list or dictionary.

```bash
python tools/prepare_finetune.py --config configs/T.yml --checkpoint weights/t_initial.pt \
  --data data/my_dataset/data.yaml --output runs/my_yolo_dataset --epochs 30 --batch 8
python tools/train_custom.py --run runs/my_yolo_dataset
```

The same classifier-reset, optimizer/EMA reset, FP32 decoder, resume, export and evaluation rules apply to both input formats. Preparation stores the converted source metadata at `runs/my_yolo_dataset/yolo/dataset.json`, including missing-label counts, then writes the canonical `train.json`, `val.json` and `classes.json` beside the config. Use these normalized JSON files for evaluation. **Annotation support does not imply YOLO model-weight compatibility.**

Standalone conversion, without loading a detector:

```bash
python tools/yolo_to_coco.py --data data/my_dataset/data.yaml --output data/converted
```

Read `data/converted/dataset.json` for generated JSON/image roots. No Ultralytics dependency or automatic download scripts are executed. [YOLO detection format reference](https://docs.ultralytics.com/datasets/detect/).
