# 基准协议与限制

[English](BENCHMARKS.md) | [简体中文](BENCHMARKS.zh-CN.md)

README中的全量COCO验证与部署子集对比是两组实验，不可混用AP。

全COCO val2017全部5,000张、640、无TTA、EMA：B第41轮58.9306AP，S第57轮57.8987AP，T尚未测得。B从完整第20轮、S从第30轮检查点修复恢复，之前退化的未来轮次已归档，不计入有效历史。不是从头、同配方、不中断的FP16/BF16对照，也不是架构精度上限证明。检测监督用COCO，DINOv3骨干另有外部预训练。

部署对比使用相同B41EMA、相同固定1,000图子集、预处理与top300。`validation/benchmarks/base41/coco_val1k.manifest.json`记录图像ID和哈希；按类别/目标尺寸覆盖后种子随机填充，不依赖模型得分，但不是无偏全量AP估计。

FP32子集59.4238AP，AMP59.4149AP，修复后的TensorRT59.3778AP，相对FP32下降0.0459个AP点，通过0.1子集门限。没有全量TRT一致性或S/T速度结论。

RTX5090、batch1、默认stream、无CUDA Graph、50预热/200测量；包括top300，排除主机I/O/预处理/传输，TRT另含设备拷贝。原始测量期间训练计算退出，暂停进程保留约1GiB显存，三后端条件相同；端到端墙钟时间另报。不能直接比较旧CUDA Graph结果。

原生GridSample引擎曾严重失败：混合精度7.7240AP、FP32 5.4015AP。显式双线性GatherElements导出修复恢复到59.3778，未改权重/原生训练。失败引擎不能当作可用速度结果。生成小熊上的0.9637置信度仅作展示，也不是评测样本。
