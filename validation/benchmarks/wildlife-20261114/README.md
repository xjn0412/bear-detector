# African Wildlife transfer evidence · 2026-11-14

[Full report](../../../docs/benchmarks/WILDLIFE_TRANSFER.md) · [中文](../../../docs/benchmarks/WILDLIFE_TRANSFER.zh-CN.md) · [Plain .pt downloads and raw evidence](https://github.com/xjn0412/bear-detector/releases/tag/wildlife-transfer-20261114)

All five runs completed 50 epochs, seed42, global batch32 at640 on RTX5090 GPUs. Train/validation/test contain1,017/222/224 images after removing41 exact duplicates. Validation chooses checkpoints; the held-out test uses the same COCO bbox evaluator for every model. These results are separate from COCO detector benchmarking.

| File | Meaning |
|---|---|
| [summary.json](summary.json) | Common test AP, class AP, paired image-bootstrap intervals; percentages |
| [curves.json](curves.json) | All50 epochs for all5 runs; native validation metrics, not a common evaluator curve |
| [selection.json](selection.json) | Validation-selected epoch and native validation score |
| [dataset-manifest.json](dataset-manifest.json) | Source archive SHA, split/box counts and every removed duplicate |
| [environment.json](environment.json) | Versions, model size, training GPU allocation, initial YOLO weight hashes, data hashes and size strata |
| [bear-config.yml](bear-config.yml) | Exact Bear experiment configuration with portable `/work/wildlife` prefix |
| [bear-initialization.json](bear-initialization.json) | Original COCO source hash, reset class heads and fresh optimizer/EMA/scaler policy |
| `yolo*-args.yml` | Native resolved training configuration for each YOLO run |
| `*-evaluation.json` | Raw backend evaluation reports and evaluated checkpoint hashes |
| [release-assets.json](release-assets.json) | All11 artifact hashes matched GitHub upload digests |
| [weights-manifest.json](weights-manifest.json) | Released weight byte sizes/hashes and exact Bear EMA preservation |

`AP_small: null` in the summary means undefined because the resized test split contains zero COCO-small objects. Raw evaluator reports retain the scaled undefined sentinel `-100`. There are27 medium and344 large objects. Bootstrap intervals condition on the fixed trained checkpoints; they do not measure training-seed uncertainty.

The release includes `wildlife-evaluation-evidence.zip`, containing standard COCO prediction JSON, ground-truth annotations, raw epoch CSV/logs, configurations, split audit and summary. Images and checkpoints are not embedded in that evidence archive. After extraction, `tools/benchmarks/summarize_wildlife.py --root PATH --bootstrap 200` recomputes the common metrics and intervals without a GPU. Download source images through the separately provided preparation script when reproducing training.

Bear-B's raw evaluated full training checkpoint SHA differs from the downloadable EMA-only file SHA because optimizer/scaler and other training state are removed. The exported model tensors were compared exactly, including dtype, and the released file passed strict model loading. Both hashes are retained; the release contains original FP32 tensors rather than a rounded compact conversion.
