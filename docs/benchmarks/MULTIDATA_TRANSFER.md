# Aquarium and VOC2007 transfer comparison

Status: training in progress. Final held-out test scores have not been published yet.

This experiment fine-tunes the original Bear-B COCO FP32 EMA and official
Ultralytics YOLO26-x COCO checkpoint on two more datasets. Neither starts from
the previously fine-tuned African Wildlife checkpoint.

| Dataset | Classes | Training images | Validation images | Held-out test images |
|---|---:|---:|---:|---:|
| Aquarium / ODinW | 7 | 448 | 127 | 63 |
| VOC2007 | 20 | 2,500 | 2,508 | 4,952 |

The original split assignments are preserved. Exact RGB-pixel duplicate removal
keeps test before validation before training: three VOC images are removed,
with no Aquarium duplicates. Two zero-area Aquarium annotations are removed.
Both models consume the same clipped boxes and lossless 640×640 PNG images,
created by PIL bilinear direct resizing. This fixes geometry across models and
differs from standard aspect-preserving letterbox evaluation.

VOC2007 difficult boxes are **included as ordinary positives for both models**.
VOC coordinates are converted from 1-based inclusive boxes to 0-based COCO
boxes. Results are an **all-box COCO-style transfer benchmark**, not official
VOC AP with difficult-object ignoring or the VOC2007 11-point metric.

Each dataset has three 50-epoch runs, seed 42, global batch 16, at 640:

- Bear-B: AdamW 1e-4, staged backbone learning rates 2.048e-6 to 5e-6,
  weight decay 1e-4, cosine schedule, three warm-up epochs, horizontal flip 0.5.
  The class heads and denoising class embedding are reset. FP32 master weights,
  FP16 bulk autocast, FP32 decoder and losses are retained.
- YOLO26-x shared: AdamW 1e-4, weight decay 1e-4, cosine schedule, three warm-up
  epochs, horizontal flip 0.5; mosaic, color jitter and geometric jitter disabled.
- YOLO26-x native: the same dataset, seed, batch and epoch budget with standard
  Ultralytics augmentation and automatic optimizer. Actual resolved settings
  are preserved with the experiment evidence.

Each run uses two RTX 5090 GPUs. Native trainer differences remain, including
normalization, losses, EMA, warm-up floors, gradient clipping and classifier
transfer. Bear's backbone has DINOv3 representation pretraining; official
YOLO26-x has Objects365→COCO detection training. This is not a matched-pretraining
or matched-parameter architecture ablation. Bear-B is about 1.9× larger.

Every checkpoint is selected by its own native validation AP50:95. YOLO recipe
selection also uses validation only. Final predictions use a common
`faster_coco_eval.COCOeval_faster` evaluator, 101 recall points,
`maxDets=[1,10,100]`, up to 300 predictions per image, and no test augmentation.
YOLO uses its one-to-many/NMS accuracy path. The small Aquarium test set gets
200 paired image-bootstrap samples; those intervals describe image-sampling
uncertainty conditional on the trained models, not training-seed variability.

Sources: [Aquarium dataset and CC BY 4.0 attribution](https://universe.roboflow.com/brad-dwyer/aquarium-combined),
[ODinW mirror](https://huggingface.co/GLIPModel/GLIP/tree/main/odinw_35),
[VOC dataset](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/),
[Ultralytics implementation](https://github.com/ultralytics/ultralytics).

Reproduction tools are `prepare_transfer_data.py`, `train_transfer_yolo.py`,
`run_transfer_experiment.py`, `summarize_transfer.py` and `plot_transfer_suite.py`
under `tools/benchmarks`. Training manifests retain archive SHA256 values,
removed annotations and duplicates; evaluation retains raw predictions and
per-epoch histories. Final weights use plain `.pt` files.
