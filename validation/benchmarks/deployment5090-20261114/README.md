# Final-model deployment validation, RTX5090,2026-11-14

Every T/S/B/L directory contains original/mixed PyTorch AMP full-COCO validation, mixed TensorRT full-COCO validation, storage/engine gates, batch1 CUDA-event timings, exact hashes and export metadata. Large prediction JSON, weights and engines remain outside Git; weights are in Releases.

- `storage.json`: conversion provenance; FP16 bulk, original FP32 decoder, gzip roundtrip.
- `reference-accuracy.json`, `mixed-accuracy.json`: paired full5000 validation at batch8.
- `tensorrt-accuracy.json`: full5000 at fixed batch1.
- `storage-validation.json`, `engine-validation.json`: separate0.1AP-point maximum-drop gates.
- `*-latency.json`: standalone GPU7 timing; no mining/training/build workload during measurements.
- `export.json`: source ONNX/engine/checkpoint provenance. Its original build-time flag is separate from subsequent `engine-validation.json`.

[Full protocol and limitations](../../../docs/benchmarks/DEPLOYMENT5090.md). Original FP32 archives and historical training AP remain unchanged. Small positive storage AP deltas are numerical changes, not training improvements. No guarantee is made for custom datasets or newly rebuilt engines.
