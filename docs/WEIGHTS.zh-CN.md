# 权重文件与较小下载包

[English](WEIGHTS.md) | [简体中文](WEIGHTS.zh-CN.md)

B/S下载包仅包含EMA模型张量与元数据，带优化器、scaler和EMA状态的完整训练检查点另行保留。文件已生成，但**尚未上传到GitHub Releases**，本页不宣称已有可用下载链接。

以下为实测十进制MB（1 MB=1,000,000字节），并非参数量：

| 模型 | FP32 EMA `.pt` | 无损 `.pt.gz` | 紧凑 `.pt` | 紧凑 `.pt.gz` |
|---|---:|---:|---:|---:|
| Base，第41轮 | 449.16 MB | 416.91 MB | 244.61 MB | 226.35 MB |
| Small，第57轮 | 295.46 MB | 274.17 MB | 167.76 MB | 155.18 MB |

- **无损压缩：** 解压后与原始文件逐字节一致，SHA256往返校验通过，保留选定检查点的实测精度。
- **紧凑候选：** 骨干、上下文模块和颈部权重以FP16存储，解码器权重仍为原始FP32，已核对完全一致。包含gzip后，B/S下载体积分别减少约49.6%/47.5%。其他权重发生舍入，不能把原始AP直接标在紧凑版本上；全量COCO精度和TensorRT一致性仍待验证。
- 两种格式都保留新任务微调所需的模型张量，但都不是可完整恢复优化器/scaler状态的训练续训检查点。

下载后先解压，再使用现有工具：

```bash
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('bear-b-coco-ema.pt.gz','rb'),open('bear-b-coco-ema.pt','wb'))"
python tools/predict.py --config configs/B.yml --checkpoint bear-b-coco-ema.pt \
  --image assets/bear-input.jpg --device cpu --output results/bear.json
```

紧凑候选替换对应文件名即可。通过仓库工具加载到正常的**FP32模型**中；FP16存储不代表直接调用`model.half()`，也不代表把解码器改成FP16。运行策略保持主体FP16 autocast、解码器FP32。将FP16张量加载回FP32参数不能恢复已舍入的信息。

压缩改变传输体积，不改变参数量或FLOPs，也不会自动提升推理速度。进一步减少运行开销需要Tiny等更小架构，或经过单独校准验证的量化。原始B/S FP32 EMA仍作为精度基准；Tiny正在训练，尚无已发布训练权重。

正式发布资产应带校验和、匹配配置与类别映射、选定轮次与实测AP，并保留上游许可条款。TensorRT引擎需在目标环境重新构建和验证，不可把旧引擎当成这些权重的导出结果。
