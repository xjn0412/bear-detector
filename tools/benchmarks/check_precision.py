"""CUDA checks for AMP preflight, rollback, checkpoint state and DDP fallback."""
import copy
import os
import sys
from pathlib import Path

import torch
from torch import nn
sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                str(Path(__file__).resolve().parents[2] / 'source')]
from precision import install_precision_policy


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(32, 32, 1, bias=False),
                                      nn.BatchNorm2d(32), nn.Dropout(0.25))
        with torch.no_grad():
            self.backbone[0].weight.fill_(1.)

    def forward(self, x, targets=None):
        value = self.backbone(x).flatten(1)[:, :4]
        return {'pred_boxes': value, 'pred_logits': value[:, :1]}


def assert_buffers_equal(a, b):
    for x, y in zip(a.buffers(), b.buffers()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def single():
    device = torch.device('cuda', 0)
    x = torch.full((2, 32, 2, 2), 3000., device=device)
    # First batch catches unsafe AMP; the check must not advance BN or RNG.
    model = Toy().to(device).train()
    expected = copy.deepcopy(model)
    install_precision_policy(model)
    before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state()
    model._bear_precision.preflight(model, lambda x: Toy.forward(model, x), (x,), {}, device)
    assert not model._bear_precision.amp_enabled
    assert_buffers_equal(model, expected)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(cuda_before, torch.cuda.get_rng_state())
    assert torch.isfinite(model(x)['pred_boxes']).all()

    # Simulate a late overflow after a successful startup check.
    model = Toy().to(device).train()
    expected = copy.deepcopy(model)
    install_precision_policy(model)
    model._bear_precision.checked = True
    rng = torch.cuda.get_rng_state()
    actual = model(x)
    actual_rng = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(rng)
    reference = expected(x)
    assert torch.equal(actual_rng, torch.cuda.get_rng_state())
    torch.testing.assert_close(actual['pred_boxes'], reference['pred_boxes'], rtol=0, atol=0)
    assert_buffers_equal(model, expected)
    assert model.backbone[1].num_batches_tracked.item() == 1
    actual['pred_boxes'].square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert not model._bear_precision.amp_enabled
    state = model._bear_precision.state_dict()
    resumed = install_precision_policy(Toy().to(device))
    resumed._bear_precision.load_state_dict(state)
    assert not resumed._bear_precision.amp_enabled and not resumed._bear_precision.checked
    assert set(resumed.state_dict()) == set(expected.state_dict())

    # Healthy weights retain AMP; explicit FP32 never runs the AMP path.
    healthy = install_precision_policy(Toy().to(device))
    healthy(torch.randn_like(x))
    assert healthy._bear_precision.amp_enabled
    explicit = install_precision_policy(Toy().to(device), 'fp32')
    assert torch.isfinite(explicit(x)['pred_boxes']).all()
    assert not explicit._bear_precision.amp_enabled
    ema = copy.deepcopy(explicit).eval()
    assert ema._bear_original_forward.__self__ is ema
    assert ema.forward.__self__ is ema
    assert ema._bear_precision is not explicit._bear_precision
    assert set(ema.state_dict()) == set(explicit.state_dict())
    try:
        explicit(torch.full_like(x, float('nan')))
    except FloatingPointError:
        pass
    else:
        raise AssertionError('Invalid FP32 outputs must stop training')
    print('PASS: preflight, rollback/RNG, finite backward, sticky resume, healthy AMP, FP32 failure guard', flush=True)


def training_loop():
    import backbone, compat, runtime
    runtime.install_training_loop()
    from engine.solver.det_engine import train_one_epoch
    from engine.solver._solver import BaseSolver

    class Criterion(nn.Module):
        def forward(self, outputs, targets, **kwargs):
            return {'loss_bbox': outputs['pred_boxes'].float().square().mean()}

    class EMA:
        updates = 0
        def update(self, model):
            self.updates += 1

    for overflow in (True, False):
        model = install_precision_policy(Toy().cuda(), 'auto' if overflow else 'fp32')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer._observed_steps = 0
        def stepped(opt, args, kwargs):
            opt._observed_steps += 1
        optimizer.register_step_post_hook(stepped)
        if overflow:
            next(model.parameters()).register_hook(lambda grad: torch.full_like(grad, float('inf')))
        ema = EMA()
        scaler = torch.amp.GradScaler('cuda', init_scale=128.) if overflow else None
        batches = [(torch.randn(2, 32, 2, 2), [{'labels': torch.tensor([0])}])]
        train_one_epoch(False, None, model, Criterion(), batches, optimizer,
                        torch.device('cuda'), 0, max_norm=0.1, scaler=scaler, ema=ema)
        assert optimizer._observed_steps == (0 if overflow else 1)
        assert ema.updates == optimizer._observed_steps
        if overflow:
            assert scaler.get_scale() == 64.
        solver = BaseSolver.__new__(BaseSolver)
        solver.model, solver.last_epoch = model, 0
        state = solver.state_dict()
        assert 'precision_policy' in state
        restored = BaseSolver.__new__(BaseSolver)
        restored.model, restored.last_epoch = install_precision_policy(Toy().cuda()), -1
        restored.load_state_dict(state)
        assert restored.model._bear_precision.amp_enabled == model._bear_precision.amp_enabled
    print('PASS: actual trainer skips overflowing gradients without EMA update; FP32 updates; precision checkpoint roundtrip', flush=True)


def ddp():
    import torch.distributed as dist
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    try:
        torch.manual_seed(42)
        model = install_precision_policy(Toy().cuda(rank))
        wrapped = nn.parallel.DistributedDataParallel(model, device_ids=[rank],
                    broadcast_buffers=False, find_unused_parameters=True)
        with torch.no_grad():
            wrapped(torch.randn(2, 32, 2, 2, device=rank))
        assert model._bear_precision.amp_enabled
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
        x = torch.full((2, 32, 2, 2), 3000. if rank == 1 else 1., device=rank)
        out = wrapped(x)
        out['pred_boxes'].square().mean().backward()
        assert not model._bear_precision.amp_enabled
        assert model.backbone[1].num_batches_tracked.item() == 2
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        optimizer.zero_grad()
        wrapped(torch.ones_like(x))['pred_boxes'].square().mean().backward()
        optimizer.step()
        dist.barrier()
        if rank == 0:
            print('PASS: one-rank overflow synchronizes FP32 fallback; DDP backward and next update succeed', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    if 'LOCAL_RANK' in os.environ:
        ddp()
    else:
        single()
        training_loop()
