"""Eager PyTorch inference guards; exported engines require separate validation."""
import warnings
import math

import torch
from precision import invalid_outputs


class CheckedInference(torch.nn.Module):
    def __init__(self, model, post, amp=False, checked=True):
        super().__init__()
        self.core = model
        self.post = post.deploy()
        self.checked = checked
        self.amp_enabled = bool(amp)
        self._amp_validated = False
        self.precision_report = dict(requested='fp16_amp' if amp else 'fp32',
                                     selected='fp16_amp' if amp else 'fp32',
                                     fallback_count=0, checked=checked)
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('Checked inference requires FP32 master weights; use autocast, not model.half()')

    def _once(self, images, sizes, amp):
        with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=amp):
            features = self.core.encoder(self.core.backbone(images))
        if self.checked and invalid_outputs(features):
            raise FloatingPointError('Nonfinite inference features')
        if self.checked and amp and not self._amp_validated:
            # Finite values can still have lost small signals. Compare before
            # top-k makes query order discontinuous, once for this model.
            with torch.autocast(device_type=images.device.type, enabled=False):
                reference = self.core.encoder(self.core.backbone(images.float()))
            if invalid_outputs(reference):
                raise FloatingPointError('Nonfinite FP32 inference reference features')
            errors = []
            for actual, expected in zip(features, reference):
                delta = (actual.double() - expected.double()).norm()
                errors.append(float(delta / expected.double().norm().clamp_min(1e-30)))
            self.precision_report['feature_relative_l2_errors'] = errors
            if any(not math.isfinite(error) or error > .05 for error in errors):
                raise FloatingPointError('AMP/FP32 inference feature error exceeds 5%')
            self._amp_validated = True
        with torch.autocast(device_type=images.device.type, enabled=False):
            outputs = self.core.decoder([f.float() for f in features])
            if self.checked and invalid_outputs(outputs):
                raise FloatingPointError('Nonfinite inference decoder outputs')
            predictions = self.post(outputs, sizes)
        if self.checked and invalid_outputs(predictions):
            raise FloatingPointError('Nonfinite inference predictions')
        return predictions

    @torch.no_grad()
    def forward(self, images, orig_target_sizes):
        if self.training or self.core.training:
            raise RuntimeError('Inference wrapper must remain in eval mode')
        if not self.checked:
            # Static export path: no Python finite checks or fallback are traced.
            return self._once(images, orig_target_sizes, self.amp_enabled)
        if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
            raise RuntimeError('Export with checked=False and validate the resulting engine separately')
        if self.amp_enabled and images.device.type != 'cuda':
            raise ValueError('AMP inference requires CUDA; use amp=False on CPU')
        if invalid_outputs((images, orig_target_sizes)):
            raise FloatingPointError('Nonfinite inference inputs')
        if (orig_target_sizes <= 0).any():
            raise ValueError('Image dimensions must be positive')
        try:
            return self._once(images.float(), orig_target_sizes, self.amp_enabled)
        except FloatingPointError as error:
            if not self.amp_enabled:
                raise
            # Eval mode disables DropPath/dropout and BN updates; retry has no
            # training RNG/buffer mutation to undo. Retain FP32 for later frames.
            self.amp_enabled = False
            self.precision_report.update(selected='fp32', fallback_count=1,
                                         fallback_reason=str(error))
            warnings.warn('Inference AMP numerical check failed; retrying the same input in FP32',
                          RuntimeWarning, stacklevel=2)
            return self._once(images.float(), orig_target_sizes, False)
