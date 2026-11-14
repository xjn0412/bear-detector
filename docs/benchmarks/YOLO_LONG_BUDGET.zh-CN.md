# YOLO 默认增强配方：更长训练预算

四组实验完成。上限100个epoch，验证集20个epoch无改善则提前停止。表中保留实际训练轮数；从同一原始COCO权重和种子重新训练，使用新的100轮调度。

| Dataset / model | Original50 AP | Max100 AP | Actual epochs | Selected epoch | AP difference |
|---|---:|---:|---:|---:|---:|
| wildlife / yolo26x | 83.25 | 82.09 | 100 | 96 | -1.15 |
| wildlife / yolo11x | 77.50 | 79.66 | 100 | 98 | +2.17 |
| aquarium / yolo26x | 53.34 | 54.63 | 100 | 91 | +1.29 |
| voc2007 / yolo26x | 59.41 | 57.27 | 21 | 1 | -2.14 |

| Dataset | Bear-B50 | Bear-L50 | YOLO26x default50 | YOLO26x max100 |
|---|---:|---:|---:|---:|
| wildlife | 91.35 | 91.54 | 83.25 | 82.09 |
| aquarium | 55.49 | 58.44 | 53.34 | 54.63 |
| voc2007 | 75.46 | 76.30 | 59.41 | 57.27 |

以上为统一测试集COCO AP百分数，差值为百分点。Bear保持50轮，长预算YOLO单独列出。只使用一个种子，参数量、预训练和优化实现不同；测试集不参与选轮次或超参数。VOC difficult作为普通正样本，指标不是官方VOC AP。

[完整协议、曲线和统计](YOLO_LONG_BUDGET.md)
