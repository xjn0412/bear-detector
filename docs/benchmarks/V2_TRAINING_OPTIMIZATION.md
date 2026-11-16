# V2 training execution optimization

Status on 2026-11-16: the user stopped the partial second epoch to optimize immediately from the intact epoch1 checkpoint (AP50:95 **49.3219%**, AP50 **66.1290%**). CUDA EMA bitwise, finite-decision and RNG/buffer rollback checks passed. An isolated runtime profile was selected and training resumed. **Measured throughput improved 8.49% in an 80-window comparison; full-epoch speed and subsequent AP are pending.**

| Eight-GPU profile | Warmup / timed windows | Seconds/update | Images/s | Decision |
|---|---:|---:|---:|---|
| Baseline | 40 / 80 | 0.34042 | 23.50 | Reference |
| Foreach EMA + combined scaler guard + duplicate-check removal | 40 / 80 | **0.31377** | **25.50** | Selected |
| Baseline, earlier shorter comparison | 40 / 30 | 0.28336 | 28.23 | Short-window reference |
| All runtime components, earlier shorter comparison | 40 / 30 | 0.33331 | 24.00 | Rejected |

Every accepted timing window had zero overflow skips. These are separate short runs with real random augmentation, not a controlled full-convergence experiment. Baseline variability between windows is visible in the table; 8.49% is the measured paired-window result, not a guaranteed full-run improvement. Validation and checkpoint I/O are excluded. All timing updates were discarded.

The selected profile sets `BEAR_FAST_RUNTIME=1`, `BEAR_FAST_COMPONENTS=ema,scaler`, `BEAR_BACKBONE_CHECKPOINT=1`, `BEAR_COMPILE_NECK=0`. It also uses the fast training loop's duplicate-check removal, retaining FP32 conversion. Grouped output norms and grouped rollback copies remain disabled. Disabling backbone recomputation failed the output/gradient gate (maximum absolute gradient discrepancy 0.01636 in the probe) and was rejected. Four GPUs with micro-batch2 fit but reached only 20.85 images/s in a separate 20-window capacity probe; they are not selected and would change local BatchNorm statistics.

The initial checkpoint handoff had failed in `psutil.wait_procs` because container `pidfd_open` returned EINVAL, leaving training interrupted from approximately 21:40 to 23:00 Asia/Shanghai. Process waiting now polls identity-aware running/zombie state without pidfd, and termination/waiting are inside baseline recovery. Three regression tests passed. The subsequent immediate handoff, experiment continuation and automatic training relaunch were exercised successfully.

The full P2 neck, four-scale head, decoder, losses, augmentation, learning-rate schedule, FP32 safety islands, FP16 overflow handling, EMA and effective batch8 remain the same. L → B → S → T still uses the measured eight-GPU selection. Increasing batch size changes BatchNorm statistics and the optimizer trajectory, so capacity measurements cannot change the formal recipe automatically.

## Changes and acceptance

| Change | Purpose | Acceptance |
|---|---|---|
| Group EMA tensor operations with foreach | Reduce Python dispatch and small CUDA launches | Preserve multiply/multiply/add rounding order, shared-state update order, integer buffers and EMA start delay; exact CPU and CUDA checks passed; selected |
| Batch rollback buffer copies | Reduce launches while retaining retry behavior | Restore buffers and CPU/CUDA RNG; CPU/CUDA checks passed; disabled in selected profile |
| Group finite-output checks and remove duplicate checks | Reduce repeated tensor scans and synchronization | Preserve finite/Inf/NaN decisions and autograd outputs; existing checked precision policy still guards the full forward |
| Combine GradScaler rank checks into one collective | Reduce collective latency | Detect invalid scales and rank disagreement; no weakening of gradient/forward/loss checks |
| Select local CUDA device before NCCL startup | Avoid unnecessary contexts on GPU0 | No model or optimizer arithmetic change |
| Optional removal of backbone recomputation | Trade spare memory for less backward compute | Eligible only after exact GPU output/gradient comparison and successful fixed-batch throughput test |
| Optional neck compilation | Investigate launch overhead | Experimental only; never automatically selected for formal training, even if the limited parity test passes |

The live NUMA affinity probe was rejected and its original CPU masks restored: 200-window means were 0.3294s before and 0.3561s after pinning (approximately 8.1% slower). These successive windows are not a controlled profiler comparison.

## Checkpoint handoff

`tools/optimize_v2_at_checkpoint.py` waits for a complete validated epoch and a full `last.pth`, verifies the owning training queue, then stops only that queue. It runs CUDA parity checks and baseline/candidate eight-GPU timing from disposable copies of the same checkpoint. Timing uses global batch8 and configurable warmup/timed windows (selected comparison: 40/80), with validation excluded from every candidate. Timed overflow skips invalidate a result. Candidates need at least a 3% measured improvement; otherwise the baseline is retained.

The selected environment is written to `execution-profile.json`; the queue resumes L from its full checkpoint, followed by B/S/T. Optimizer, scaler, precision policy, EMA, best AP and patience are restored. Legacy checkpoints do not save RNG state, so resume is not a claim of a bitwise identical future training trajectory. An experiment failure restores the baseline queue. `optimization/CANCEL` cancels the waiting handoff; cancellation during an experiment intentionally leaves control with the operator.

The separate four-GPU micro-batch2 measurement is capacity evidence only. L's timing selection is not a measured per-model optimization result for B/S/T. Final AP and complete-epoch speed require subsequent training evidence.

## Verification evidence

- Five new CPU tests passed: EMA exactness with shared parameters, EMA cache after checkpoint loading, finite decisions/read-only autograd, buffer/RNG rollback, scaler rejection.
- Two existing accumulation/overflow CPU tests passed.
- Integrated v2 suite with the fast runtime installed: four CPU tests passed, five CUDA tests skipped because formal training occupied the GPUs.
- CUDA EMA, finite decisions and RNG/buffer rollback passed; backbone recomputation removal failed its separate gate and remained disabled.
- Selected comparison completed all 80 timed updates on both profiles with zero skips; formal L restarted from epoch1.
- Three process-wait regression tests passed. `--resume-experiment` verifies the old queue and experiment have exited before continuing from a preserved checkpoint. Compilation tests are now opt-in.

[Raw evidence](../../validation/training/v2-optimization-20261116/README.md) · [Original 4/8-GPU measurements](V2_COCO_TRAINING.md)
