# Verification record

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

The public repository contains no COCO images, foundation weights, trained detector checkpoints or TensorRT engines. `--pretrained` is the usable bootstrap route for new users; supply a compatible trained checkpoint for detector transfer learning. Rebuild engines and repeat held-out accuracy checks for your own trained checkpoint, dataset and deployment machine. A private repository requires GitHub access before cloning.
