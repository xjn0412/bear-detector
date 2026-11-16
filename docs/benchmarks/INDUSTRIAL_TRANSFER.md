# Industrial defect transfer benchmark

Status: **8/8 runs complete**, checked 2026-11-16T09:10:34+00:00. KSDD2 Bear-L stopped at epoch37 after20 epochs without a validation improvement and selected epoch17. The preserved best checkpoint scored test AP36.6906%. Late training accuracy collapsed despite finite FP32 outputs; best-checkpoint protection prevented that trajectory from replacing the selected model.

## Completed held-out test results

All values below are **test AP50:95 in percent**, shared COCO evaluator/maxDets100 and FP32 inference. These are internal split results, not official dataset leaderboards.

| Dataset | Bear-L | YOLO26x | Difference (points) | Completed epochs Bear / YOLO | Selected epoch Bear / YOLO |
|---|---:|---:|---:|---|---|
| CR7-DET |58.52|51.80|+6.72|26 /83|6 /63|
| GC10-DET (corrected10 classes) |41.87|31.14|+10.73|26 /85|6 /65|
| NEU-DET |49.65|45.91|+3.74|31 /100|11 /100|
| KSDD2 derived boxes |36.69|11.07|+25.62|37 /100|17 /84|

Per-epoch validation curves and final test reports are in the linked evidence directory below. Validation AP and test AP are distinct. Different pretraining/recipes and one seed limit the comparison.

TensorRT rebuild and validation are complete for the selected CR7 Bear-L checkpoint: mixed208FPS/AP58.2883, or conservative FP32-backbone75.4FPS/AP58.3454, versus FP32 AP58.5209. See [precision revalidation](TENSORRT_REVALIDATION_20261116.md) for full reports, scope and the0.20-point acceptance target.


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

Observed in production: the new policy passed its startup check, then caught a
real forward overflow during epoch20, between progress reports at steps200 and220
(zero-based batch indices). It retried the same batch in FP32 and subsequent
training losses remained finite. The run selected persistent FP32 with activation
checkpointing; this precision change is recorded in the published policy report.

## Evaluation ID correction

A deployment validation warning exposed ground-truth annotation ID0, which
conflicts with the COCO evaluator unmatched sentinel. Final reports above now
reindex annotation IDs to1..N **in memory only**. Images, categories, geometry,
weights and saved predictions are unchanged. All seven completed reports were
rescored from saved predictions; source annotation hashes remain recorded.
`tools/coco_metrics.py` is shared by both evaluators, and future dataset
preparation emits positive IDs. The regression test checks a perfect prediction
against a source annotation with ID0 and confirms AP100 after reindexing.

Earlier test numbers are superseded, with original reports archived as diagnostic
evidence under `evaluation-id0-legacy/`. Historical native Bear validation curves
and checkpoint selection used the original evaluator and remain unchanged; this
rescore does not reconstruct corrected per-epoch curves or claim to remove that
selection limitation. The completed KSDD2 run retained its historical validation
metric for consistent early stopping; its final test evaluation uses the repair.

## Why Bear peaked earlier

The runs use different transfer recipes, so epoch counts do not isolate architecture.
For CR7, Bear training loss fell29.39→21.28 between epochs6 and26 while native
validation AP fell59.56→55.42. GC10 similarly fell in training loss39.06→23.09
while native AP fell38.67→36.03. These trends are consistent with overfitting.
The logged Bear LR remained9.985e-5 at epoch6 and8.857e-5 at26: the100-epoch
cosine schedule had barely decayed before early stopping. A shorter transfer
schedule or plateau-triggered LR reduction should be tested, not assumed to work.

Bear augmentation was horizontal flip only, whereas resolved YOLO settings used
mosaic1.0, scale0.5, translation0.1 and HSV jitter. YOLO weight decay was5e-4
versus Bear1e-4. Bear-L has roughly221M parameters versus YOLO26x59M. These
differences plausibly explain slower initial YOLO adaptation and less early
overfitting, but their individual effects have not been isolated in ablations.
Both models began from COCO detector checkpoints; neither was trained from scratch.

KSDD2 is a separate optimization failure: Bear's training localization errors
increased sharply as validation collapsed, and training has1877 negative images
out of2098. Imbalance is a plausible contributor, not a proven root cause.
FP32 recovery prevents representational overflow but does not guarantee sound
optimization. Best-checkpoint protection selected epoch17, whose corrected final
test AP is36.6906, rather than the collapsed final epoch. All experimental
follow-ups should use validation, with the original benchmark retained unchanged.

## Publication audit

Verified published commit 9c3b2f0df252494c12556a7d446c59cf8fa68a32 against the original completed run files:

- **8 runs,488 epoch records,976 native validation AP50/AP50:95 values** match the original Bear JSON logs and YOLO CSV logs.
- All eight final test scores were independently recomputed from saved predictions with positive annotation IDs. AP50:95/AP50/AP75 and size metrics match within1e-9 AP points.
- Selected epochs, completed epoch counts, checkpoint SHA256, source annotation hashes and prediction provenance were checked.
- The combined per-epoch table matches all eight curves. Five published compressed split manifests, including the invalid archive, decompress to the recorded memberships.
- Actual published bytes were verified against GitHub blob hashes. Compressed-file and KSDD2 YOLO JSON-formatting differences did not change data.
- Two **invalid GC10 diagnostic archives** had incomplete published curve snapshots (6/24 epochs versus22/71 recorded). They are refreshed in this update and remain explicitly invalid. Active ten-class GC10 scores were unaffected.

Historical Bear validation curves and checkpoint selection retain the annotation-ID0 evaluator limitation. They are faithful log records, not retroactively corrected validation AP. Correcting every past epoch would require unavailable past predictions/checkpoints or a new training run. Final test scores use the repaired evaluator. The audit does not establish statistical significance, identical recipes, near-duplicate independence or official leaderboard performance.

[Every epoch](../../validation/benchmarks/industrial-20261116/epoch-progress.md) · [Combined CSV](../../validation/benchmarks/industrial-20261116/all-epochs.csv) · [Machine-readable audit](../../validation/benchmarks/industrial-20261116/audit.json)
