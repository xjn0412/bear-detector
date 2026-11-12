# Verification record

[English](VERIFICATION.md) | [简体中文](VERIFICATION.zh-CN.md)

Verified on 2026-11-12 before the initial repository content commit. Checks ran on an isolated copy; active COCO training code and recipe were not changed. The CPU checks explicitly hid CUDA devices. Dependencies came from the recorded installed environment, not a newly downloaded clean wheel installation; installing on a different OS/CUDA combination still requires the installation checks.

## Current repository checks

- Six class/schema/head tests, including class counts1/3/80/123, sparse/reordered IDs, classifier reset with equal class counts, non-class shape rejection, invalid/NaN boxes and category validation.
- Two portable sampling tests, including image edges/out-of-range coordinates.
- CLI parsing for initialization, synthetic dataset generation, preparation, training, prediction, export, evaluation, subset creation and comparison.
- Full CPU pipeline: random Tiny detector initialization → synthetic three-class COCO JSON → preparation → class mapping → native PyTorch CPU prediction → portable ONNX export and ONNX checker.
- Public pretrained Tiny initialization through timm downloaded/loaded successfully, without private weight paths.
- Parameters recomputed for all three model configurations.
- Accuracy comparison command reproduced the committed fixed-subset result gate.

Run the offline checks yourself:

```bash
python -m pip install -r requirements-export.txt
python tools/verify_cpu.py --onnx --output results/cpu-verification.json
python tools/compare_results.py --reference validation/B_torch-fp32.json \
  --candidate validation/portable_B_tensorrt.json --max-drop 0.1 \
  --output results/recorded-parity.json
```

A CPU smoke pass validates wiring, checkpoint compatibility and export construction, not trained accuracy. The generated images/boxes are synthetic and its untrained predictions have no benchmark meaning.

## Earlier measured GPU evidence retained with this code

`validation/` contains the real-image1/3-class full-criterion backward/optimizer/checkpoint-reload test, Tiny3-class two-epoch full training CLI smoke (four actual optimizer and EMA updates, scaler saved), data-preparation checks, Base41 checkpoint provenance, fixed subset manifest, environment and same-checkpoint PyTorch/TensorRT accuracy/timing reports. Those GPU measurements were not rerun while preparing the repository, to avoid competing with the active COCO training. The model/export path is the same validated implementation.

The end-to-end training smoke used toy boxes and proves execution only. Multi-GPU training has run on the operational COCO B/S jobs; the standalone arbitrary-class fine-tuning example was checked on oneGPU, not every topology/dataset. No independent real custom-dataset convergence result, full-COCO TensorRT parity result, or trainedTiny accuracy result is claimed.

## Scope of reproducibility

The repository contains no COCO images, foundation weights or TensorRT engines. Small EMA and compact detector checkpoints are available as release assets. `--pretrained` is an alternative bootstrap route. Rebuild engines and repeat held-out accuracy checks for your own trained checkpoint, dataset and deployment machine. A private repository requires GitHub access before cloning.

## Bear Detector / YOLO update

Six YOLO tests cover class/coordinate conversion, directory/image/text-list paths, background and strict labels, invalid/NaN/out-of-range/task rejection, split overlap and class schema errors. `verify_cpu.py` now checks both COCO and direct YOLO-YAML preparation paths and CPU prediction. The generated bear illustration uses actual Base41 EMA CPU FP32 output; JSON retains input/checkpoint hashes and unedited model box coordinates. No active COCO run is changed.

## Small compact release validation

Full COCO val2017, 5,000 images, 640, batch2, PyTorch2.14.1+cu130, selected epoch57 EMA. Both reference and compact loaded strictly into FP32 models; AMP used FP16 bulk plus an FP32 decoder. All predictions were finite. Compact AP changed by -0.005814 points in FP32 and -0.009586 in AMP, passing the preset maximum0.1-point loss in each mode. [Full metrics and evaluated checkpoint hashes](../validation/S_compact_coco.json).

The release repacks metadata only; tensor keys, dtypes and values were checked identical to the evaluated candidate. The release manifest links evaluated and released hashes. Gzip round trips were byte-exact, and GitHub asset SHA256 digests and sizes were checked after upload. This was an accuracy check alongside Tiny training, without speed claims. TensorRT compact accuracy and custom-dataset accuracy require separate validation; Base compact remains unpublished pending source-server recovery.
