# Measured B epoch41 EMA deployment check

Fixed 1,000-image COCO val2017 subset, 640×640, same checkpoint and preprocessing, RTX 5090 GPU7. These are **subset AP** values, not official full COCO results.

| Backend | AP50:95 | AP50 | GPU mean ms | FPS | Image-to-CPU-output mean ms |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | 59.4238 | 77.0142 | 19.4644 | 51.38 | 32.64 |
| PyTorch FP16 backbone/context/neck + FP32 decoder | 59.4149 | 77.0066 | 19.1562 | 52.20 | 31.46 |
| TensorRT FP16 backbone/context/neck + FP32 decoder, portable bilinear sampling | 59.3778 | 76.9858 | 4.0203 | 248.74 | 16.10 |

TRT versus FP32 AP delta: **−0.0459 AP points**. This passes the selected 0.1-point subset regression threshold. Full COCO numerical/accuracy parity remains unvalidated. The full-COCO AP of this selected B epoch41 checkpoint was 58.9306 before export.

GPU timing is batch1 eager enqueue with no CUDA Graph, 50 warm-ups and 200 timed iterations. It includes top-300 postprocessing and excludes image decode, resize/normalization and host transfers. TRT additionally includes device-to-device input copies and uses the default CUDA stream; these are measured application-path timings, not maximum optimized throughput. All training contexts were exited during timing; the miner remained SIGSTOP with ~1GiB resident GPU memory and performed no mining compute. The same residency applied to all three backends. Do not compare these values directly with older CUDA Graph timings.

## Repairs validated

1. Native ONNX GridSample engines failed accuracy: mixed-precision TRT 7.7240 AP; FP32 TRT 5.4015 AP. Finite outputs were insufficient.
2. ONNX Runtime matched PyTorch CPU on three real images. Intermediate TRT checks isolated a large sampling discrepancy at the second decoder layer while preceding backbone/neck/layer0/refinement/offset tensors matched closely.
3. Export-only explicit bilinear zero-padded sampling using GatherElements restored subset AP to 59.3778 without altering weights or native PyTorch training. Edge/out-of-bounds sampling unit tests passed.
4. Class heads/postprocessing tested for 1,3,80,123 classes; sparse ID mapping, category order, schema rejection, same-count classifier reset and coordinate scaling passed.
5. Real-image B-model full-criterion backward, optimizer update and checkpoint save/model reload checks passed for 1 and 3 classes.
6. Tiny 3-class full CLI preparation/train/validation/checkpoint test completed two epochs/four optimizer updates with four EMA updates and scaler state saved. The toy dataset used synthetic boxes on real images; its AP is not a model accuracy result. The initial default scaler skipped all four updates; the custom fine-tuning default now starts at scale1.0 and refuses an entire epoch without an actual optimizer update.

Architecture and numerical policies of ongoing B/S/T training were preserved. Server1 training resumed after the requested timing sessions; server2 was not paused.
