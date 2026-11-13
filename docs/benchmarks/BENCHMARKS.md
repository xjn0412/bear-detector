# Benchmark protocol and limitations

[English](BENCHMARKS.md) | [简体中文](BENCHMARKS.zh-CN.md)

The README contains two separate comparisons: full-COCO PyTorch validation of selected checkpoints, and same-checkpoint backend parity on a fixed1,000-image subset. Do not combine their AP numbers.

## Full COCO

Base epoch41 EMA:58.9305642 mAP50:95. Small epoch57 EMA:57.8986801. Both use all5,000 val2017 images at640, without test-time augmentation. Tiny detector AP is unmeasured. Parameter counts were recomputed from the release configuration on CPU; include the full model for80 classes.

The training history contains numerical repairs/resumes: B completed20 then resumed with whole-decoderFP32; S completed30 then resumed likewise. Prior degraded later epochs are archived, not included in active history. Ongoing100-epoch schedules have patience20. A flat late curve is not proof of an architecture's absolute accuracy ceiling. These are not matched fresh FP16/BF16 runs. External DINOv3 backbone pretraining was used; only detector supervision is COCOtrain2017.

## Deployment subset

`validation/benchmarks/base41/RESULTS.md` and adjacent JSON files record the measured Base41 comparison. Manifest: `validation/benchmarks/base41/coco_val1k.manifest.json` contains selected image IDs/source hash/subset hash. Selection greedily covers category/size combinations then uses seeded random fill; it is model-score independent but not an unbiased estimate of full COCO AP.

All three backends use the exact same immutable EMA checkpoint hash, same images, PIL bilinear direct resize, normalization, coordinate conversion and top300. TensorRT drops0.0459077 AP points versus FP32 on this subset. No full-COCO TensorRT parity approval and no Small/Tiny deployment speed/accuracy claim is implied.

RTX5090 GPU7, batch1, eager enqueue/default CUDA stream, no CUDA Graph,50warm-ups/200iterations. Training compute was stopped during the original measurement. A paused process retained about1GiB GPU memory but did no compute, identically across backends. Timings include top300; exclude hostIO/preprocessing/transfers. TensorRT includes device-to-device input copies. Image-to-CPU-output wall timings are recorded separately and exclude JSON serialization/COCO evaluation.

Old nativeGridSample TensorRT builds produced7.7240AP(mixed) and5.4015AP(FP32) on the subset; they are unsafe. The release exporter uses equivalent explicitbilinear sampling, whose final engine achieved59.3778460AP. Only the final portable-sampling timing is advertised.

No matched YOLO benchmark or verified FLOPs comparison is included. Backbone parameters alone understate the full decoder/neck cost. Do not reuse speed-proxy Tiny timings as trained Tiny results.
