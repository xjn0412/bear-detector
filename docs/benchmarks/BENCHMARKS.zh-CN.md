# 基准协议与限制

[English](BENCHMARKS.md) | [简体中文](BENCHMARKS.zh-CN.md)

README中的全量COCO验证与部署子集对比是两组实验，不可混用AP。

全COCO val2017全部5,000张、640、无TTA、最佳EMA：Tiny第58轮56.4597AP、Small第57轮57.8987AP、全新Base第42轮59.0008AP、Large第24轮60.0728AP。T/B/L从第1轮使用完整FP32解码器，Small在第30轮后修复恢复。所有训练已结束，不能据此当成严格的FP16/BF16对照或架构精度上限。检测监督用COCO，DINOv3骨干另有外部预训练。

部署对比使用相同B41EMA、相同固定1,000图子集、预处理与top300。`validation/benchmarks/base41/coco_val1k.manifest.json`记录图像ID和哈希；按类别/目标尺寸覆盖后种子随机填充，不依赖模型得分，但不是无偏全量AP估计。

FP32子集59.4238AP，AMP59.4149AP，修复后的TensorRT59.3778AP，相对FP32下降0.0459个AP点，通过0.1子集门限。没有全量TRT一致性或S/T速度结论。

RTX5090、batch1、默认stream、无CUDA Graph、50预热/200测量；包括top300，排除主机I/O/预处理/传输，TRT另含设备拷贝。原始测量期间训练计算退出，暂停进程保留约1GiB显存，三后端条件相同；端到端墙钟时间另报。不能直接比较旧CUDA Graph结果。

原生GridSample引擎曾严重失败：混合精度7.7240AP、FP32 5.4015AP。显式双线性GatherElements导出修复恢复到59.3778，未改权重/原生训练。失败引擎不能当作可用速度结果。生成小熊上的0.9637置信度仅作展示，也不是评测样本。

## 当前结果补充（2026-11-14）

全量验证最佳EMA：T56.4597、S57.8987、B59.0008、L60.0728。训练均已结束；旧Base41的子集测速仍是历史同权重后端对比，不能代表最终Base42。四个尺寸的完整/部署参数及FLOPs现已实测，见[计算量](COMPLEXITY.zh-CN.md)；未新增GPU测速。
