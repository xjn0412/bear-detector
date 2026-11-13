# 模型架构与数值策略

[English](ARCHITECTURE.md) | [简体中文](ARCHITECTURE.zh-CN.md)

ConvNeXt是CNN骨干，不是ViT；最深层上下文模块和检测解码器使用注意力。T/S/B共用检测头。

| 模块 | 设置 |
|---|---|
| ConvNeXt-T 深度 | 3/3/9/3 |
| ConvNeXt-S/B 深度 | 3/3/27/3 |
| T/S 通道 | 96/192/384/768 |
| B 通道 | 128/256/512/1024 |
| 上下文注意力 | stride32，20×20=400tokens，两块，384维、6头、SDPA、门控FFN、残差初始scale0.1 |
| P2细节支路 | 64通道与depthwise stride2卷积，下采样后注入256通道P3 |
| 颈部 | 三尺度256通道FPN/PAN式HybridEncoder，depth_mult1.37、expansion1.25、CSP2、sum融合 |
| 解码器 | 6层、256维、3尺度，采样点3/6/3、300queries，训练100去噪queries |
| 输入 | 固定640×640，RGB直接缩放与ImageNet归一化 |

没有stride4解码器记忆层，也不在高分辨率特征上做全局上下文注意力。完整参数：B112.11M、S73.68M、T52.05M（80类）。

FP16 autocast只用于骨干/上下文/颈部，**整个解码器、浮点预测、匹配与损失FP32**；主参数和优化器FP32，动态GradScaler。EMA只在实际优化器更新后推进，整轮零更新抛错，非有限损失不能静默置零。先前门控FFN溢出促使采用统一FP32解码器；FP16训练本身不保证FP16部署一致性。

`backbone.py`实现上下文/细节支路；`source/engine/deim`保留上游颈部、解码器与准则；`runtime.py`安装数值保护；`tools/common.py`定义类别与推理契约；`portable_sampling.py`只在导出时替换采样实现。相比常见YOLO密集头，该模型使用查询解码器迭代细化和top-k输出；没有进行匹配YOLO对照，不能宣称更快或更准。
