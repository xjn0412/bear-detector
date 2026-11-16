# Execution optimization evidence

Snapshot taken on 2026-11-16 while L epoch1 continued (7,708 effective updates, 12 initial scaled-gradient skips). The GPU handoff is waiting for its first complete validated epoch checkpoint.

- `cpu-tests.json`: five foreach/runtime tests and two training-loop CPU tests.
- `integrated-cpu-tests.json`: integrated v2 run with fast runtime installed; four passed, five CUDA tests skipped.
- `affinity-probe.json`: rejected live NUMA pinning probe; original affinities restored.
- `status.json`: handoff phase at snapshot time, not a live status feed.

GPU parity, execution candidate timings and the selected profile will be produced on the training host after checkpoint handoff. These files do not establish a speedup or final AP equivalence.

[Method and deployment gates](../../../docs/benchmarks/V2_TRAINING_OPTIMIZATION.md)
