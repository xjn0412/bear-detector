"""Checked FP16 AMP with a persistent, same-batch FP32 fallback.

The model keeps FP32 master parameters. Install before DDP wrapping so a
failed forward is retried before DDP prepares the returned autograd graph.
"""
import json
import os
import types
from pathlib import Path
from contextlib import contextmanager

import torch
from torch import nn


def distributed_bad(bad, device):
    flag = torch.tensor(int(bad), device=device, dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


def floating_tensors(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from floating_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from floating_tensors(item)


def invalid_outputs(value):
    flags = [~torch.isfinite(t).all() for t in floating_tensors(value)]
    return bool(torch.stack(flags).any().item()) if flags else False


class ForwardSnapshot:
    """Rollback forward-only mutation; never copy or change trainable weights."""
    def __init__(self, model, device):
        self.device = device
        self.cpu_rng = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
        self.buffers = [(b, b.detach().clone()) for b in model.buffers()]

    @torch.no_grad()
    def restore(self):
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state(self.cuda_rng, self.device)
        for buffer, saved in self.buffers:
            buffer.copy_(saved)


class PrecisionPolicy:
    def __init__(self, requested='auto'):
        if requested not in ('auto', 'fp32'):
            raise ValueError('Precision must be auto or fp32')
        self.requested = requested
        self.amp_enabled = requested == 'auto'
        self.checked = False
        self.events = []
        self._checkpointing_enabled = False

    def record(self, event, **details):
        entry = dict(event=event, **details)
        self.events.append(entry)
        print('BEAR_PRECISION ' + json.dumps(entry), flush=True)
        destination = os.getenv('BEAR_PRECISION_LOG')
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if destination and (not distributed or torch.distributed.get_rank() == 0):
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(json.dumps(self.state_dict(), indent=2))
            temporary.replace(path)

    def state_dict(self):
        return dict(version=1, requested=self.requested,
                    amp_enabled=self.amp_enabled, checked=self.checked,
                    events=self.events)

    def load_state_dict(self, state):
        # An explicit FP32 request wins. A previous fallback stays disabled.
        self.amp_enabled = self.requested == 'auto' and state['amp_enabled']
        self.events = list(state.get('events', []))
        self.checked = False  # Recheck resumed weights on the current device.

    def preflight(self, model, forward, args, kwargs, device):
        if self.checked:
            return
        if not self.amp_enabled or device.type != 'cuda':
            self.amp_enabled = False
            self.checked = True
            self.record('fp32_selected', reason='requested, resumed fallback, or non-CUDA device')
            return
        snapshot = ForwardSnapshot(model, device)
        failed = []
        handles = []
        # Match YOLO's per-layer AMP/FP32 comparison, using Bear's actual
        # weights and first training batch. Exclude explicit FP32 islands.
        protected = [n for n, m in model.named_modules()
                     if getattr(m, '_keep_fp32', False)]

        def compare(name):
            def hook(layer, inputs, reference):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    candidate = layer.forward(inputs[0])
                delta = (reference.double() - candidate.double()).norm()
                # A fixed 1e-6 floor can silently accept total loss of a
                # small signal. FP64 norms also avoid FP32 norm overflow.
                limit = 0.05 * reference.double().norm() + 1e-12
                if not bool(torch.isfinite(delta) & (delta <= limit)):
                    failed.append(name)
            return hook

        try:
            for name, layer in model.named_modules():
                if not name.startswith(('backbone.', 'encoder.')):
                    continue
                if any(name == p or name.startswith(p + '.') for p in protected):
                    continue
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    handles.append(layer.register_forward_hook(compare(name)))
            with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
                reference = forward(*args, **kwargs)
            if distributed_bad(invalid_outputs(reference), device):
                raise FloatingPointError('FP32 AMP-check reference is nonfinite; inspect model/data')
            del reference
        finally:
            for handle in handles:
                handle.remove()
            snapshot.restore()
        # Also exercise the complete AMP path (attention, residuals, etc.).
        try:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                candidate = forward(*args, **kwargs)
            bad = invalid_outputs(candidate) or bool(failed)
            del candidate
            bad = distributed_bad(bad, device)
        finally:
            snapshot.restore()
        self.amp_enabled = not bad
        self.checked = True
        self.record('amp_check_failed' if bad else 'amp_check_passed',
                    failed_layers=failed[:20], failed_layer_count=len(failed),
                    selected='fp32' if bad else 'fp16_amp')

    def forward(self, model, forward, args, kwargs):
        device = next(model.parameters()).device
        if not model.training:
            # Validation keeps the existing FP32 contract.
            with torch.autocast(device_type=device.type, enabled=False):
                return forward(*args, **kwargs)
        self.preflight(model, forward, args, kwargs, device)
        if os.getenv('BEAR_COMPILE_NECK','0')=='1' and not getattr(model,'_bear_neck_compiled',False):
            # Compile after the AMP preflight has removed its temporary hooks.
            # Wrapping forward preserves every parameter/state-dict key.
            model.encoder.forward = torch.compile(model.encoder.forward, mode='default')
            model._bear_neck_compiled = True
        if not self.amp_enabled:
            self.enable_fp32_checkpointing(model)
        snapshot = ForwardSnapshot(model, device) if self.amp_enabled else None
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=self.amp_enabled):
            outputs = forward(*args, **kwargs)
        if not distributed_bad(invalid_outputs(outputs), device):
            return outputs
        del outputs  # Release the failed graph before allocating the retry.
        if not self.amp_enabled:
            raise FloatingPointError('Nonfinite FP32 model outputs; update blocked; inspect model/data')
        snapshot.restore()
        self.amp_enabled = False
        self.record('forward_overflow_fallback', selected='fp32',
                    action='restore RNG/buffers and retry same batch before backward')
        self.enable_fp32_checkpointing(model)
        with torch.autocast(device_type=device.type, enabled=False):
            outputs = forward(*args, **kwargs)
        if distributed_bad(invalid_outputs(outputs), device):
            snapshot.restore()
            raise FloatingPointError('Same-batch FP32 retry is nonfinite; update blocked')
        return outputs

    def enable_fp32_checkpointing(self, model):
        # FP32 activations can otherwise exhaust a GPU sized for AMP training.
        # timm checkpoints ConvNeXt blocks and preserves stochastic-depth RNG.
        body = getattr(getattr(model, 'backbone', None), 'body', None)
        if not self._checkpointing_enabled and hasattr(body, 'set_grad_checkpointing'):
            body.set_grad_checkpointing(True)
            self._checkpointing_enabled = True
            self.record('fp32_activation_checkpointing', backbone='ConvNeXt')


def install_precision_policy(model, requested='auto'):
    if hasattr(model, '_bear_precision'):
        return model
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('AMP requires FP32 master parameters; do not call model.half()')
    # Store bound methods on the instance, not inside a closure. deepcopy used
    # by ModelEMA must bind both methods to the copied model and copied policy.
    model._bear_original_forward = model.forward
    model._bear_precision = PrecisionPolicy(requested)
    model.forward = types.MethodType(checked_forward, model)
    return model


def checked_forward(self, *args, **kwargs):
    return self._bear_precision.forward(self, self._bear_original_forward, args, kwargs)


def finite_gradients(optimizer, device):
    flags = [~torch.isfinite(p.grad).all() for group in optimizer.param_groups
             for p in group['params'] if p.grad is not None]
    bad = bool(torch.stack(flags).any().item()) if flags else True
    if distributed_bad(bad, device):
        raise FloatingPointError('Nonfinite or missing FP32 gradients; optimizer update blocked')
