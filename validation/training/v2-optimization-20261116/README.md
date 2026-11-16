# Execution optimization evidence

2026-11-16: the partial L epoch2 was stopped at the user's request; the intact epoch1 full checkpoint was used for disposable profiling, then formal training resumed from it. Epoch1 EMA AP50:95=49.321867%, AP50=66.129045%.

- `baseline.json`, `fast.json`: selected eight-GPU comparison, 40 warmup +80 timed updates. Baseline0.340417s/update; selected EMA/scaler/duplicate-check profile0.313771s/update, measured throughput+8.49%. Zero timed skips for both. These are short-window results; full-epoch speed and AP acceptance remain pending.
- `execution-profile.json`: deployed environment flags; batch8, micro-batch1/GPU, full P2, decoder and numerical safeguards preserved.
- `execution-parity.json`: `passed` concerns the independent runtime components. CUDA EMA exactness, finite decisions and rollback passed. Backbone recomputation removal has separate false gates and was rejected.
- `combined40w30t/`: earlier shorter comparison where the full combination was slower, plus four-GPU batch2 capacity probe (20.85 images/s, not selected).
- `cpu-tests.json`: five runtime and two training-loop CPU tests.
- `integrated-cpu-tests.json`: four CPU tests passed; five CUDA tests skipped at that earlier snapshot.
- `handoff-recovery-tests.json`: three process-wait regression tests passed after a container pidfd error interrupted the first handoff.
- `affinity-probe.json`: rejected NUMA probe; original affinities restored.
- `L-epochs.csv`: completed full validation, epoch1.
- `status.json`: deployment snapshot, not a live feed.

Benchmark seed/recipe and checkpoint are fixed, but GPU/augmentation timing varies between runs. These files establish the measured window comparison, not guaranteed training-time or final-AP equivalence.

[Method and acceptance gates](../../../docs/benchmarks/V2_TRAINING_OPTIMIZATION.md)
