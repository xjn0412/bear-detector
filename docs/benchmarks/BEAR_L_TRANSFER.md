# Bear-L transfer comparison

Status: Bear-L fine-tuning and evaluation are queued/running; final results are pending.

Bear-L starts from its original FP32 COCO EMA checkpoint. Fresh classification
heads and denoising class embeddings are initialized independently for each task.
The previous Bear-B experiment's data, augmentation, optimizer, learning-rate
schedule, seed and 50-epoch budget are retained.

| Dataset | Classes | Global training batch | Input | Epoch budget |
|---|---:|---:|---:|---:|
| African Wildlife | 4 | 32 | 640 | 50 |
| Aquarium | 7 | 16 | 640 | 50 |
| VOC2007 | 20 | 16 | 640 | 50 |

The exact audited train/validation/test splits and labels from the earlier runs
are reused. Validation chooses the EMA checkpoint; one common COCO bbox evaluator
compares Bear-L, Bear-B, YOLO26-x shared and YOLO26-x default on held-out test images.
The YOLO recipe is selected on validation, not test scores. Wildlife and Aquarium
also get paired image-bootstrap intervals. This is one training seed per model.

VOC difficult boxes remain ordinary positives, as in the earlier experiment;
the result is all-box COCO AP, not official VOC AP. Bear-L is much larger than
YOLO26-x, and the pretrained representations and native optimization differ.
No matched-size, matched-pretraining or latency claim is intended.

`tools/benchmarks/run_bear_variant_transfer.py` reproduces the fine-tuning and
strict FP32 EMA `.pt` export. `compare_bear_l_transfer.py` creates the final tables,
per-class and size metrics, confidence intervals, learning curves and hashes.
