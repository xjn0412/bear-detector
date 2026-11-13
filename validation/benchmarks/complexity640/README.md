# Fixed640 complexity and weight-storage evidence

Measured2026-11-14, reference Bear commit ea9530086fc767e69469c4873cae4d56dc677d5a, PyTorch2.14.1+cu130/Python3.11.18. CPU-only. No training/mining changes, GPU timing, TensorRT export or accuracy rerun.

- `T/S/B/L-profile.json`: actual640 deployed eval arithmetic; full/deploy parameter counts, per-module/operator totals and config hashes. Use [tools/profile_model.py](../../../tools/profile_model.py).
- `bear-storage.json`: full released .pt tensor dtype/payload and file-size audit. No optimizer/scaler or duplicate EMA.
- `yolo-storage.json`: official v8.4.0 API asset sizes and upstream digests, plus metadata-only full-checkpoint parameter/dtype counts. Complete archive digests were not independently verified; no YOLO forward executed.
- `audit_yolo_metadata.py`: inert metadata parser used for the audit. It may fetch only a checkpoint prefix, never executes the model, and is a research audit helper rather than a general checkpoint loader.

[Counting convention and exclusions](../../../docs/benchmarks/COMPLEXITY.md) · [Interpretation of weight sizes](../../../docs/models/WEIGHT_SIZE.md).
