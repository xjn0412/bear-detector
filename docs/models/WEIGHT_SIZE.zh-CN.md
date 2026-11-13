# 权重大小与参数量

[English](WEIGHT_SIZE.md) | [简体中文](WEIGHT_SIZE.zh-CN.md)

**Bear文件没有异常膨胀，但架构确实偏重。** 当前发布保存完整FP32 EMA，约4字节/参数；YOLO26公开检查点的参数以FP16存储，约2字节/参数。此外YOLO性能表是融合后部署参数，下载文件保留更多训练结构。

MB=1,000,000 bytes; M=1,000,000 parameters. Gzip bytes are download/storage size, not GPU memory. AMP training dtype does not determine checkpoint storage dtype.

## Bear releases

| Model | Full params M | Parameter bytes FP32 MB | Actual .pt MB | Gzip download MB | Excess over parameter bytes |
|---|---:|---:|---:|---:|---:|
| Bear-T | 52.04883 | 208.20 | 208.86 | 193.77 | 0.317% |
| Bear-S | 73.68339 | 294.73 | 295.46 | 274.17 | 0.247% |
| Bear-B | 112.10825 | 448.43 | 449.16 | 416.91 | 0.162% |
| Bear-L | 221.39830 | 885.59 | 886.32 | 822.79 | 0.082% |

实际审计：所有浮点state张量为FP32；包仅含model、last_epoch、metadata，没有重复EMA副本、optimizer或GradScaler。多出的0.08–0.32%来自buffer（如anchors/归一化统计）、元数据和序列化开销。gzip无损，解压不会改变张量；当前所有尺寸均保留此方案，没有重做紧凑包。

## YOLO26 public checkpoints

| Model | Published deployed params M | Checkpoint params M | Stored parameter dtype | Expected param MB | Actual .pt MB | Published GFLOPs |
|---|---:|---:|---|---:|---:|---:|
| YOLO26n | 2.4 | 2.57228 | FP16 | 5.14 | 5.54 | 5.5 |
| YOLO26s | 9.5 | 10.00978 | FP16 | 20.02 | 20.42 | 20.9 |
| YOLO26m | 20.4 | 21.89625 | FP16 | 43.79 | 44.26 | 68.4 |
| YOLO26l | 24.8 | 26.29970 | FP16 | 52.60 | 53.21 | 86.8 |
| YOLO26x | 55.7 | 58.99337 | FP16 | 117.99 | 118.67 | 194.4 |

文件大小来自官方`ultralytics/assets`的`v8.4.0` release API。完整检查点参数及dtype由ZIP的data.pkl张量元数据解析；未执行YOLO模型、未验证完整归档SHA、未重新测其AP/FLOPs/速度。权重形状和dtype无需下载全部数值即可审计。参考[官方性能表](https://docs.ultralytics.com/models/yolo26/#performance-metrics)对部署参数与检查点参数差异的说明。

## 是否比预期更大？

以YOLO26-X为例：下载结构58.99M参数，FP16文件118.67MB；Bear-B为112.11M参数，FP32文件449.16MB。参数约1.90倍，存储每参数再乘2，约3.8倍文件大小，符合预期。不能拿Bear无损gzip和YOLO未压缩.pt直接推断架构效率。

但Tiny并不小：52.05M完整参数、51.10M部署参数，规模已接近YOLO26-X的55.7M部署结构。Tiny预训练CNN为27.82M，Bear额外模块合计24.23M；其独立颈部10.40M、解码器9.90M。架构命名只表示Bear家族相对尺寸，不代表边缘轻量模型。未来减重应测试更窄/更浅颈部和更小解码器，必须重新训练验证，不能仅改名或压缩文件。

纯FP16存储可以理论上减半浮点字节，但会舍入权重，需要每个模型的精度验证；更不能对需要FP32解码器的模型直接model.half()。安全的无损safetensors等格式只能减少少量序列化开销，不能把FP32的4字节变为2字节。

[Raw measurement evidence](../../validation/benchmarks/complexity640/README.md) · [Downloads](WEIGHTS.md) · [Complexity](../benchmarks/COMPLEXITY.md).
