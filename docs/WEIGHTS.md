# Weight downloads

[English](WEIGHTS.md) | [简体中文](WEIGHTS.zh-CN.md)

[Small release, manifests and SHA256SUMS](https://github.com/xjn0412/bear-detector/releases/tag/v0.1.0-weights). The repository and release assets are public; browser/curl downloads need no login. Both packages contain selected epoch57 EMA tensors and metadata, without optimizer/scaler resume states. They support inference and fresh fine-tuning.

| Small package | Download size | Full COCO FP32 AP50:95 | Full COCO AMP AP50:95 |
|---|---:|---:|---:|
| [Compact](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-compact.pt.gz) | 155.18 MB | 57.89035 | 57.87800 |
| [Lossless reference](https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-ema.pt.gz) | 274.17 MB | 57.89617 | 57.88758 |

Decimal MB, 1 MB = 1,000,000 bytes. The original uncompressed Small FP32 checkpoint is 295.46 MB. Compact download is 47.5% smaller than that file, or 43.4% smaller than its lossless gzip. Compression changes neither parameter count nor FLOPs, and does not itself improve inference speed.

## Accuracy and storage policy

Full COCO val2017, all5,000 images, 640, batch2. Compact lost **0.00581 AP point in FP32** and **0.00959 in AMP**, passing the preset maximum0.1-point loss in each mode. Outputs were finite. [All metrics and evaluated hashes](../validation/releases/small/S_compact_coco.json). The reference rerun differs by0.00251 AP point from the historical training validation (57.89868); the paired results above are the appropriate compression comparison.

Compact stores FP16 backbone/context/neck tensors, with the original decoder tensors preserved in FP32. Load into a normal **FP32 model** through the repository tools; do not call `model.half()`. AMP inference keeps the entire decoder FP32. Loading FP16 storage into FP32 parameters does not undo rounding. The lossless package preserves the original file byte for byte.

The release adds validation metadata to the evaluated compact checkpoint; tensor keys/dtypes/values were verified identical. [Manifest](../validation/releases/small/S_release_manifest.json) records evaluated and released hashes. Gzip round trips and uploaded GitHub asset SHA256 digests were verified. **Compact TensorRT accuracy is not validated.** Build a fresh engine and validate it; older engines contain different weights. Custom tasks also need their own accuracy check. Validation ran alongside training, without timing claims.

## Download and use

Use the public release links above, or download without authentication:

```bash
curl -fLO https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/bear-s-coco-compact.pt.gz
curl -fLO https://github.com/xjn0412/bear-detector/releases/download/v0.1.0-weights/SHA256SUMS
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('bear-s-coco-compact.pt.gz','rb'),open('bear-s-coco-compact.pt','wb'))"
python tools/predict.py --config configs/S.yml --checkpoint bear-s-coco-compact.pt \
  --classes configs/coco-classes.json --image assets/bear-input.jpg \
  --device cuda --amp --output results/bear.json
```

Verify the archive SHA256 before decompression (compact SHA256 below). To use the lossless reference, substitute `bear-s-coco-ema.pt(.gz)`. CPU uses `--device cpu` without `--amp`. Installation, COCO/YOLO arbitrary-class fine-tuning and export are covered by the other guides.

```
90b17d14e3dc50ec59a0df846b85f203957c29ee4e3396e96ccfe26d9b308b14  bear-s-coco-compact.pt.gz
e2b7a9565a2ea9e646e070dc6d1f59b70a675f390dd2958f38218f465cb9710d  bear-s-coco-ema.pt.gz
```

Reproduce the paired accuracy check with a local COCO val2017 installation:

```bash
python tools/validate_compact.py --config configs/S.yml \
  --reference bear-s-coco-ema.pt --candidate bear-s-coco-compact.pt \
  --annotations data/coco/annotations/instances_val2017.json \
  --images data/coco/val2017 --output results/compact-comparison.json
```

The earlier Base checkpoint is currently inaccessible after a source-server failure. Base retraining is in progress (setup/restart stage); no untested Base asset is published. Tiny is training and has no released trained checkpoint yet. Retained DEIMv2 non-commercial and separate DINOv3 license terms apply.
