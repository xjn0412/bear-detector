# Benchmark protocol and limitations

[English](BENCHMARKS.md) | [简体中文](BENCHMARKS.zh-CN.md)

The README contains two separate comparisons: full-COCO PyTorch validation of selected checkpoints, and same-checkpoint backend parity on a fixed1,000-image subset. Do not combine their AP numbers.

## Full COCO

Final selected EMA validation: Tiny58=56.4597, Small57=57.8987, fresh Base42=59.0008, Large24=60.0728. All use full5,000-image val2017 at640 without TTA. Tiny/Base/Large used the full FP32 decoder from epoch1; Small was repaired/resumed after30. T/B stopped with10-epoch patience, Large with20; all training is complete. These are not matched fresh FP16/BF16 ablations. DINOv3 backbone pretraining used external data; detector supervision used COCO train2017.

Parameter and FLOPs measurements now cover all four release configs. [Exact scope, per-module counts and reproduction](COMPLEXITY.md). Full checkpoint and deployment counts are distinguished, as YOLO's performance tables report fused/deployed counts.

## Deployment subset

`validation/benchmarks/base41/RESULTS.md` and adjacent JSON files record the measured Base41 comparison. Manifest: `validation/benchmarks/base41/coco_val1k.manifest.json` contains selected image IDs/source hash/subset hash. Selection greedily covers category/size combinations then uses seeded random fill; it is model-score independent but not an unbiased estimate of full COCO AP.

All three backends use the exact same immutable EMA checkpoint hash, same images, PIL bilinear direct resize, normalization, coordinate conversion and top300. TensorRT drops0.0459077 AP points versus FP32 on this subset. That historical subset alone implies no final-model validation; the new full-COCO checks and final T/S/B/L timings are linked below.

RTX5090 GPU7, batch1, eager enqueue/default CUDA stream, no CUDA Graph,50warm-ups/200iterations. Training compute was stopped during the original measurement. A paused process retained about1GiB GPU memory but did no compute, identically across backends. Timings include top300; exclude hostIO/preprocessing/transfers. TensorRT includes device-to-device input copies. Image-to-CPU-output wall timings are recorded separately and exclude JSON serialization/COCO evaluation.

Old nativeGridSample TensorRT builds produced7.7240AP(mixed) and5.4015AP(FP32) on the subset; they are unsafe. The release exporter uses equivalent explicitbilinear sampling, whose final engine achieved59.3778460AP. Only the final portable-sampling timing is advertised.

No matched YOLO latency benchmark has been run. Bear FLOPs are now measured; the external YOLO FLOPs remain author-reported with a different profiler. Backbone parameters alone understate the full decoder/neck cost. Do not reuse speed-proxy Tiny timings as trained Tiny results.

## Current final-model benchmark / 当前最终模型基准

[Final T/S/B/L full-COCO deployment validation, isolated RTX5090 timings and mixed storage](DEPLOYMENT5090.md) · [中文](DEPLOYMENT5090.zh-CN.md). Historical Base41 subset results above are retained for provenance, not used as final-model timings.
