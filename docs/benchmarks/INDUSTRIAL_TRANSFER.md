# Industrial defect transfer benchmark

Status: running. No final test results are claimed until the corresponding run has a completed test report.

This experiment compares the **released Bear-L architecture** with **YOLO26x** on CR7-DET, GC10-DET, NEU-DET and a derived KSDD2 bounding-box task. It does not use the proposed four-scale Bear architecture.

## Protocol

- Both models start from their published COCO detector checkpoints; all target datasets are fine-tuning datasets.
- Fixed 640×640 RGB inputs, seed42, batch8, up to100 epochs, patience20 measured on validation AP50:95. One GPU per run on an8×RTX5090 machine.
- Identical pre-resized images, class definitions and train/validation/test membership for each model. Images are converted once with PIL bilinear direct square resize; target boxes are scaled independently in x/y. This deliberately controls image geometry; it is not original-aspect-ratio evaluation.
- Exact decoded-pixel duplicates are collapsed before splitting, with the union of their unique annotations. No exact pixel hash overlaps the splits. Near-duplicate or production-sequence independence is not established.
- CR7, GC10 and NEU use custom approximately80/10/10 splits stratified by image class-presence signatures. Rare strata may remain training-only. These are reproducible internal splits, not official leaderboard results.
- KSDD2 preserves official test membership and reserves approximately10% of official training images for validation, stratified by positive/negative status.
- KSDD2 masks become one tight box per8-connected foreground component, with no area filtering. Defect-free images remain empty-label negatives. Results are detection AP on derived boxes, not official segmentation or image-level anomaly metrics.
- Classification adaptation and optimizer/EMA state are fresh. Bear resets every target-class head and denoising class embedding. YOLO follows the released framework's class-head adaptation.
- Model recipes differ: Bear uses the repository's conservative transfer recipe (flip, AdamW1e-4, DINOv3 layer-wise backbone learning rates, FP16 autocast with the complete decoder/loss in FP32); YOLO uses its standard augmentation defaults with explicit AdamW1e-3 and weight_decay0.0005. This is a system-level transfer comparison, not an architecture-only or equal-compute ablation.
- Validation selects the checkpoint. Test is used only after training finishes. Each model retains its native validation evaluator; final test scores use the same `faster-coco-eval` COCO box evaluator, IoU0.50:0.95, maxDets100. Final inference uses FP32 for both models.
- One seed is a preliminary comparison, not a statistical significance claim. Model sizes and upstream pretraining differ.

## Sources and data processing

| Dataset | Source used | Important qualification |
|---|---|---|
| CR7-DET | https://zenodo.org/records/14264153 | Creator release; original YOLO labels |
| GC10-DET | https://www.kaggle.com/datasets/alex000kim/gc10det | Mirror of the original XML dataset; duplicate folder copies collapsed; files without XML excluded and listed |
| NEU-DET | https://zenodo.org/records/16882077 | Mirror in YOLO format; custom split replaces its source split |
| KSDD2 | https://www.vicos.si/resources/kolektorsdd2/ | Official images/masks and test split; derived box annotations |

Original GC10 source: https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets

Original NEU source: https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/

KSDD2 is CC BY-NC-SA4.0. Raw datasets are not redistributed in this repository. Consult each dataset's source terms; this experiment publishes preparation code, source checksums, split membership and measured results.

## Evidence

See [`validation/benchmarks/industrial-20261116/`](../../validation/benchmarks/industrial-20261116/) for:

- `status.json`: current job status and completed epochs;
- per-dataset manifests, counts and exact split membership (`split-membership.json.gz`);
- per-model resolved protocols and validation curves;
- final held-out test reports and checkpoint hashes, once completed;
- acquisition checksums and installed-package versions.

Validation curves are not final test scores. A failed or incomplete job must remain labeled as such.

## Reproduction

Use the repository's CUDA training environment and an isolated environment with Ultralytics8.4.215. On a headless container, use `opencv-python-headless` instead of the GUI OpenCV wheel; record the actual installed environment.

Put downloaded/extracted datasets under `<base>/raw/{cr7,gc10,neu-det,ksdd2}` and both initialization checkpoints under `<base>/downloads/`. The expected checkpoint names are `bear-l-coco-mixed.pt` and `yolo26x.pt`.

```bash
python tools/benchmarks/prepare_industrial.py --base <base> --dataset cr7
python tools/benchmarks/prepare_industrial.py --base <base> --dataset gc10
python tools/benchmarks/prepare_industrial.py --base <base> --dataset neu-det
python tools/benchmarks/prepare_industrial.py --base <base> --dataset ksdd2

CUDA_VISIBLE_DEVICES=0 python tools/benchmarks/run_industrial.py --base <base> --dataset cr7 --model bear_l
CUDA_VISIBLE_DEVICES=1 python tools/benchmarks/run_industrial.py --base <base> --dataset cr7 --model yolo26x
```

`run_industrial.py` expects this repository at `<base>/repo`. Repeat the two model commands for each dataset. The optional `supervise_industrial.py` launches one model per GPU and writes local evidence snapshots; it does not contain GitHub credentials or publish automatically. Start a supervisor only once to avoid duplicate GPU jobs.

## Citation

Please cite the original dataset creators, including the CR7-DET paper *An efficient detector for detecting surface defects on cold-rolled steel strips*, the GC10 paper *Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network*, the NEU-DET work *An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features*, and Božič, Tabernik and Skočaj's *Mixed supervision for surface-defect detection: from weakly to fully supervised learning* for KSDD2.
