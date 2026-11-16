"""Eager PyTorch inference guards; exported engines require separate validation."""
import warnings

import torch
from precision import invalid_outputs


class CheckedInference(torch.nn.Module):
    def __init__(self, model, post, amp=False, checked=True):
        super().__init__()
        self.core = model
        self.post = post.deploy()
        self.checked = checked
        self.amp_enabled = bool(amp)
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
            warnings.warn('Inference AMP produced nonfinite values; retrying the same input in FP32',
                          RuntimeWarning, stacklevel=2)
            return self._once(images.float(), orig_target_sizes, False)
