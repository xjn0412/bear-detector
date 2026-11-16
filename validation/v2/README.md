# V2 implementation verification

Measured2026-11-16. These checks use random initialization and synthetic inputs. **No v2 COCO AP, transfer benchmark or production latency is claimed.**

| Check | Result | Evidence |
|---|---|---|
| Full v2-L arithmetic, actual640 deployed forward |783.9922 GFLOPs;224.5296M full /223.1572M deployed params |[L-profile.json](L-profile.json) |
| RTX5090 v2-L forward/backward and AdamW |Two finite, nonzero optimizer updates; loss80.2737→78.5073 |[gpu-smoke.json](gpu-smoke.json) |
| PyTorch AMP inference |Normal/zero/tiny/large inputs finite; first-input per-scale relative L2 below0.001 |[gpu-smoke.json](gpu-smoke.json) |
| Failure-path regression tests |9 passed: full-neck/anchor geometry, all-scale gradients, FP32 projection range, tiny gradients, RNG/buffer rollback, training/inference fallback, invalid inputs, scaled-gradient update blocking |[test_v2.py](../../tests/test_v2.py) |
| Weight-storage regression tests |4 passed, including v2 FP32 islands and rejection of newly zeroed values |[test_compact.py](../../tests/test_compact.py) |
| Existing class/schema regression |6 passed |[test_classes.py](../../tests/test_classes.py) |
| V2-T ONNX export and checker |Passed, fixed batch1×3×640×640 |[T-export.json](T-export.json) |
| ONNX Runtime versus PyTorch FP32 |Two inputs passed; maximum box error0.0000763 pixels, score error0.000000127 |[onnx-check.json](onnx-check.json) |
| Supported training CLI and checkpoint |V2-T completed2 synthetic epochs:8 batches,6 optimizer/EMA updates,2 scaled-overflow skips; loss scale1024→256; complete state saved |[cli-smoke.json](cli-smoke.json) |
| TensorRT engine build / AP parity |Not verified; TensorRT is absent from this validation environment |Exporter enforces v2 FP32 scopes but engine validation remains required |

The v2-L GPU smoke uses PyTorch2.14.1+cu130, batch1, FP32 master weights, checked FP16 CNN/neck, FP32 projections/context/decoder, AdamW and dynamic GradScaler. Peak allocated memory for the verification process is27.88GiB; this includes test instrumentation and is not an isolated training-memory benchmark. Default total batch8 assumes8 GPUs; do not assume batch4 per32GB GPU is viable.

An earlier checkpointing-enabled smoke attempt observed nonfinite **scaled gradients** despite a finite forward and stopped before an update. The verifier now exercises loss-scale reduction and unchanged-parameter checks if that recurs; the final two-update run completed at scale1024 with finite gradients. The supported CLI also completed8 synthetic training batches, safely skipped2 scaled-overflow updates, and saved6 EMA updates with final scale256. A separate injected-gradient-overflow regression explicitly verifies that the optimizer is skipped and the scale drops1024→512. These checks used single-GPU training; an eight-GPU DDP run and long-training stability remain unverified. The numerical policy remains conservative and supports explicit FP32.

Full-resolution maps are160²/80²/40²/20², with34,000 decoder-memory locations. Pooled context contains1,600 tokens. Profile breakdown: backbone/projections/context575.7235G, full neck200.4560G, decoder7.8127G. Context10.0270G is already included in the backbone column. Counting scope matches the [v1 convention](../../docs/benchmarks/COMPLEXITY.md).

Source checkpoint and exported ONNX hashes are recorded in the JSON reports. Random checkpoints/engines are not published as useful detector weights. The ONNX export is a static graph and contains no Python runtime fallback.

Reproduction commands and the supported training launcher are in [ARCHITECTURE_V2.md](../../docs/models/ARCHITECTURE_V2.md).
