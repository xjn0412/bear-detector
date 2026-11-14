# COCO / YOLO 标注微调

[English](FINETUNE.md) | [简体中文](FINETUNE.zh-CN.md)

所有命令在仓库根目录运行。固定640输入，支持任意正整数类别数。类别增减会重建全部分类头和去噪类别嵌入，即使类别数相同也会重置；其他权重严格按结构加载。新微调从新的优化器、EMA、GradScaler开始；恢复训练则保留同一运行的完整状态。标注格式兼容不代表能够直接加载YOLO模型权重。

## 初始化

可使用匹配T/S/B/L结构的已训练完整检测器检查点。Small预训练检测权重已在[Releases](https://github.com/xjn0412/bear-detector/releases/tag/v0.1.0-weights)提供，直接下载.pt文件后准备微调。Tiny、Base、Large最终权重也均已发布；也可从公开DINOv3骨干开始：

```bash
python tools/initialize.py --config configs/T.yml --pretrained --output weights/t_initial.pt
# 离线时改为 --backbone-weights weights/model.safetensors
```

此时只有骨干预训练，检测器头是新初始化，需要充分训练。`--random-init`仅用于自检。

## COCO JSON

JSON包含 `images`、`annotations`、`categories`；框为原图像素 `[x,y,w,h]`。图像/标注ID唯一；类别ID可稀疏，类别名不能重复。训练类别定义完整模式，验证集可以缺少某类实例，但不能改变类别含义。框必须有限、宽高为正。使用互不重叠的训练/验证图片。

```bash
python tools/prepare_finetune.py --config configs/T.yml --checkpoint weights/t_initial.pt \
  --train-json data/my_dataset/train.json --val-json data/my_dataset/val.json \
  --train-images data/my_dataset/train --val-images data/my_dataset/val \
  --output runs/my_dataset --epochs 30 --batch 8 --lr 0.0001
python tools/train_custom.py --run runs/my_dataset
```

## YOLO 检测文本

目录包含 `images/train`、`images/val`、`labels/train`、`labels/val`。标签与图像同名，每行 `class cx cy w h`，坐标归一化到[0,1]，类别从0连续编号。只支持检测框，不支持分割、多边形、姿态、OBB、NDJSON或远程图片URL。超出图像边界、非有限值、非法类别会被拒绝。

`data/my_dataset/data.yaml`：

```yaml
path: .
train: images/train
val: images/val
names:
  0: person
  1: bear
```

相对 `path` 以YAML所在目录为基准；train/val可写目录、单图、图片文本列表或这些路径的列表。图片列表中以 `./` 开头的条目相对该列表文件，其他相对路径以数据集根目录为基准。图像路径须包含 `images/`，最后一个该目录分量替换为 `labels/` 后定位标签。

```bash
python tools/prepare_finetune.py --config configs/T.yml --checkpoint weights/t_initial.pt \
  --data data/my_dataset/data.yaml --output runs/my_yolo_dataset --epochs 30 --batch 8
python tools/train_custom.py --run runs/my_yolo_dataset
```

缺失或空标签表示背景；加入 `--strict-labels` 可拒绝缺失标签。准备过程把缺失计数记录在 `yolo/dataset.json` 中。也可单独转换：

```bash
python tools/yolo_to_coco.py --data data/my_dataset/data.yaml --output data/converted
```

不需要安装Ultralytics，也不会执行YAML中的下载脚本。

## 训练、恢复与验证

```bash
torchrun --standalone --nproc_per_node=4 tools/train_custom.py --run runs/my_dataset
python tools/train_custom.py --run runs/my_dataset --resume runs/my_dataset/training/last.pth
```

总batch必须能被GPU数整除且不能超过训练图片数，epochs至少2。输出目录首次准备时不能已存在。使用对应运行的完整检查点恢复，不能用resume切换类别。独立微调按指定epochs运行，并不包含线上COCO的patience20监控器。不要绕过包装器直接执行 `source/train.py`。

准备过程生成 `config.yml`、归一化 `train.json` / `val.json`、`classes.json`、`initial.pth`。评估使用准备后的验证JSON；模型输出0开始的类别索引，通过classes.json映射回原始ID/名称。

完整训练检查点保存原始权重、EMA、优化器、调度器、scaler与轮次；推理/导出明确选择EMA。独立训练可能保存 `best_stg1.pth` / `best_stg2.pth`，应比较两个阶段的验证AP后选全局最好者；不要默认第二阶段更好。微调默认关闭继承的mosaic/mixup/copyblend，以动态scale1.0起步，保留FP32解码器和有限性保护。30轮与1e-4仅是起点，不是所有数据集的最优配方。

## 已完成的小数据集示例

[African Wildlife迁移实验](../benchmarks/WILDLIFE_TRANSFER.zh-CN.md)提供Bear-B实际微调、YOLO对照、去重划分、50轮记录与可下载四类权重。
