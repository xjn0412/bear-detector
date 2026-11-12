# 验证记录与范围

[English](VERIFICATION.md) | [简体中文](VERIFICATION.zh-CN.md)

发布前在独立目录验证，不修改线上COCO训练。原有6项类别/头测试、2项采样测试；新增6项YOLO转换测试，覆盖像素坐标、类别映射、文本列表/路径、背景标签与strict模式、非法框/类别/其他任务拒绝、训练验证重叠、非法类别模式。

```bash
python tools/verify_cpu.py --onnx --output results/cpu-verification.json
```

完整CPU自检覆盖随机Tiny初始化、合成三类COCO/YOLO、两种prepare入口、两种准备后CPU预测、ONNX导出及checker。另保留此前公开预训练Tiny初始化、真实图片1/3类GPU反向/优化器/检查点重载、三类两轮CLI训练的证据。此前GPU训练自检完成4次实际优化器与EMA更新；合成框的AP无意义。

B41固定COCO子集的PyTorch/TensorRT精度速度证据保留在validation目录；本次小熊展示另用B41EMA CPU FP32实际推理，原始坐标、得分、输入/检查点哈希随图提供。没有因此新增GPU基准或全COCO TensorRT认证。

第一版GitHub干净runner已通过CPU torch/torchvision与requirements安装、8项原有测试、编译检查；新增YOLO测试也加入CI。不同驱动/CUDA/操作系统仍须执行安装检查。代码自检不证明任意自定义数据都收敛，也不证明多GPU微调在所有拓扑都已验证。

## Small紧凑权重验证

第57轮EMA、全量COCO val2017 5,000张、640、batch2、PyTorch2.14.1+cu130。原始与紧凑权重严格加载到FP32模型；AMP为FP16主体加FP32解码器。所有输出有限。压缩使FP32 AP下降0.005814点，AMP下降0.009586点，均通过预设0.1点阈值。[完整指标与评测文件哈希](../validation/releases/small/S_compact_coco.json)。

发布包仅重新打包元数据，逐张量核对键、dtype与数值完全相同；manifest保留评测文件与发布文件的哈希对应。gzip解压字节一致，上传后核对GitHub资产SHA256和大小。此次与Tiny训练并行，不用于测速。紧凑权重TensorRT和自定义数据精度需另行验证。Base因源服务器故障尚未发布紧凑权重。
