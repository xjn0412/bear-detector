"""Check an exported v2 graph against FP32 PyTorch; not a dataset AP test."""
import argparse
import json
from pathlib import Path
from common import Inference, load_model, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/V2-T.yml')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--onnx', required=True)
    p.add_argument('--output', default='results/v2-onnx-check.json')
    a = p.parse_args()
    import numpy as np
    import onnxruntime as ort
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.mha.set_fastpath_enabled(False)
    cfg, core = load_model(a.config, a.checkpoint, deploy=True)
    model = Inference.create(core, cfg.postprocessor, amp=False).eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(a.onnx, sess_options=options, providers=['CPUExecutionProvider'])
    records = []
    for label, image in [('zeros', torch.zeros(1, 3, 640, 640)),
                         ('normal', torch.randn(1, 3, 640, 640))]:
        sizes = torch.tensor([[640, 640]], dtype=torch.int64)
        with torch.inference_mode():
            expected = [x.numpy() for x in model(image, sizes)]
        actual = session.run(None, {'images': image.numpy(), 'orig_target_sizes': sizes.numpy()})
        assert all(np.isfinite(x).all() for x in actual)
        assert np.array_equal(expected[0], actual[0]), 'class/order mismatch'
        np.testing.assert_allclose(actual[1], expected[1], atol=.1, rtol=1e-4)
        np.testing.assert_allclose(actual[2], expected[2], atol=1e-4, rtol=1e-3)
        records.append(dict(input=label, box_max_absolute_error=float(np.max(np.abs(actual[1] - expected[1]))),
                            score_max_absolute_error=float(np.max(np.abs(actual[2] - expected[2])))))
    report = dict(passed=True, config=a.config, config_sha256=digest(a.config),
                  checkpoint_sha256=digest(a.checkpoint), onnx_sha256=digest(a.onnx),
                  onnxruntime=ort.__version__, records=records,
                  scope='Two FP32 CPU inputs, random v2-T checkpoint; no AP or TensorRT claim')
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
