# Industrial defect transfer benchmark

Status: running.

**GC10 correction:** Initial GC10 runs used12 raw XML label strings and are invalid. The corrected benchmark merges `10_yaozhed` into `10_yaozhe` and excludes the entire image `img_02_425616500_00770.jpg` because one box is labeled only `d`. Both models restart from their original COCO checkpoints. Archived initial runs are diagnostics only and must not appear as valid GC10 results. A schema assertion now requires exactly7/10/6/1 classes for CR7/GC10/NEU/KSDD2.
 No final test results are claimed until the corresponding run has a completed test report.

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
- per-dataset manifests, counts and exact split membership;
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

### KSDD2 numerical recovery (2026-11-16)

Bear-L stopped during epoch15 with nonfinite box predictions. The last complete
checkpoint (epoch14) has finite model, EMA and optimizer tensors. Infinite
decoder anchor buffers are intentional invalid-location masks.

An isolated replay from that checkpoint reproduced the failure in epoch16.
The original interrupted batch/RNG was not saved, so this is a reproduced failure
of the same kind, not a claim to replay the original batch exactly. The replay
captured model state, inputs, targets and pre-forward RNG for a deterministic
comparison. First invalid operation: backbone.projections.2.0, the P5 1x1
convolution, before GroupNorm. Its inputs were finite (absolute maximum9053.694).
FP16 convolution yielded2268 nonfinite values. GroupNorm spread these to22400
values and the neck/decoder then produced nonfinite boxes. Entire-forward FP32
was finite (projection maximum65658.195, above FP16 maximum65504).

The fix runs only the P5 projection (conv/GroupNorm/SiLU) with autocast disabled
and FP32 inputs. Other backbone/neck operations retain FP16 AMP and the decoder
and loss retain FP32. Architecture, parameter names and checkpoint shapes are
unchanged. With this narrow fix the captured projection maximum was65668.672,
all boxes/losses/gradients were finite. No NaN replacement, clipping or skipped
bad batch was introduced; finite-value guards stay enabled.

`tools/benchmarks/check_p5_projection.py` provides a CUDA regression for
pre-normalization overflow, finite backward, strict checkpoint compatibility and
FP32 parity. The captured-batch forward/backward test also passed. Original logs
and full checkpoints were retained on the host. Production resumed at epoch15
from the full epoch14 checkpoint, restoring model/optimizer/scaler/EMA; historical
validation selection and patience are restored as well. The original checkpoint
does not contain RNG state, so resumed data/drop-path randomness is reseeded.
The recovery precision change is disclosed in KSDD2's protocol; other already
running jobs retain the code imported at their launch.

### General AMP safeguards supersede the P5-only recovery

The P5-only recovery completed epochs15–17, best validation AP39.200894256286
at17, but another nonfinite forward interrupted epoch18. The narrow P5 fix
therefore did not establish general AMP stability. The new precision policy
adds a YOLO-style per-layer AMP check and a complete-forward finite check on
the actual first training batch. Later forward overflow triggers a synchronized
same-batch FP32 retry, restoring buffers and CPU/CUDA RNG before retry. The
fallback is sticky in checkpoints and uses ConvNeXt activation checkpointing.
Existing finite loss guards remain. Scaled-gradient overflow skips updates
without advancing EMA. See the fine-tuning guide for controls and limitations.

Tests passed: CUDA preflight rejection and healthy AMP retention, forward-state/RNG
rollback, finite retry backward, explicit FP32, precision checkpoint round-trip,
EMA deepcopy binding, actual training-loop GradScaler skip/EMA behavior, and
two-GPU one-rank overflow followed by DDP backward and another optimizer update.
Full Bear-L batch8 tests with restored optimizer state/EMA and a synthetic
positive target passed in AMP, FP32 and injected-overflow retry modes. Peak
allocated memory was20.708GiB in AMP and23.867GiB with checkpointed FP32; the
uncheckpointed FP32 test approached30.506GiB. These are diagnostic measurements,
not dataset AP or throughput claims. Production resumes from the finite full
epoch17 checkpoint; precision reports will disclose any actual fallback.
