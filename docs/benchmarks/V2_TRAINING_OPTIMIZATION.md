# V2 training execution optimization

Status on 2026-11-16: L epoch1 completed at AP50:95 **49.3219%**, AP50 **66.1290%**. The first handoff stopped the queue after saving its full checkpoint, then failed in `psutil.wait_procs`: container `pidfd_open` returned EINVAL. This happened before the original recovery guard, leaving training interrupted from approximately 21:40 to 23:00 Asia/Shanghai. Baseline L was relaunched from its intact full checkpoint. Process waiting now polls identity-aware running/zombie state without pidfd, and termination/waiting are covered by baseline recovery. Three regression tests passed. GPU acceptance is requeued after epoch2. **No GPU speedup or unchanged final AP has yet been measured.**

The full P2 neck, four-scale head, decoder, losses, augmentation, learning-rate schedule, FP32 safety islands, FP16 overflow handling, EMA and effective batch8 remain the same. L → B → S → T still uses the measured eight-GPU selection. Increasing batch size changes BatchNorm statistics and the optimizer trajectory, so capacity measurements cannot change the formal recipe automatically.

## Changes and acceptance

| Change | Purpose | Acceptance |
|---|---|---|
| Group EMA tensor operations with foreach | Reduce Python dispatch and small CUDA launches | Preserve multiply/multiply/add rounding order, shared-state update order, integer buffers and EMA start delay; exact CPU tests passed; CUDA gate pending |
| Batch rollback buffer copies | Reduce launches while retaining retry behavior | Restore buffers and CPU/CUDA RNG; CPU tests passed; CUDA gate pending |
| Group finite-output checks and remove duplicate checks | Reduce repeated tensor scans and synchronization | Preserve finite/Inf/NaN decisions and autograd outputs; existing checked precision policy still guards the full forward |
| Combine GradScaler rank checks into one collective | Reduce collective latency | Detect invalid scales and rank disagreement; no weakening of gradient/forward/loss checks |
| Select local CUDA device before NCCL startup | Avoid unnecessary contexts on GPU0 | No model or optimizer arithmetic change |
| Optional removal of backbone recomputation | Trade spare memory for less backward compute | Eligible only after exact GPU output/gradient comparison and successful fixed-batch throughput test |
| Optional neck compilation | Investigate launch overhead | Experimental only; never automatically selected for formal training, even if the limited parity test passes |

The live NUMA affinity probe was rejected and its original CPU masks restored: 200-window means were 0.3294s before and 0.3561s after pinning (approximately 8.1% slower). These successive windows are not a controlled profiler comparison.

## Checkpoint handoff

`tools/optimize_v2_at_checkpoint.py` waits for a complete validated epoch and a full `last.pth`, verifies the owning training queue, then stops only that queue. It runs CUDA parity checks and baseline/candidate eight-GPU timing from disposable copies of the same checkpoint. Timing uses global batch8, warmup40 and timed30 optimizer windows, with validation excluded from every candidate. Timed overflow skips invalidate a result. Candidates need at least a 3% measured improvement; otherwise the baseline is retained.

The selected environment is written to `execution-profile.json`; the queue resumes L from its full checkpoint, followed by B/S/T. Optimizer, scaler, precision policy, EMA, best AP and patience are restored. Legacy checkpoints do not save RNG state, so resume is not a claim of a bitwise identical future training trajectory. An experiment failure restores the baseline queue. `optimization/CANCEL` cancels the waiting handoff; cancellation during an experiment intentionally leaves control with the operator.

The separate four-GPU micro-batch2 measurement is capacity evidence only. L's timing selection is not a measured per-model optimization result for B/S/T. Final AP and complete-epoch speed require subsequent training evidence.

## Verification evidence

- Five new CPU tests passed: EMA exactness with shared parameters, EMA cache after checkpoint loading, finite decisions/read-only autograd, buffer/RNG rollback, scaler rejection.
- Two existing accumulation/overflow CPU tests passed.
- Integrated v2 suite with the fast runtime installed: four CPU tests passed, five CUDA tests skipped because formal training occupied the GPUs.
- No CUDA parity result or candidate throughput result existed at this snapshot. The detached handoff process was confirmed alive while L continued its first epoch.

[Raw evidence](../../validation/training/v2-optimization-20261116/README.md) · [Original 4/8-GPU measurements](V2_COCO_TRAINING.md)
