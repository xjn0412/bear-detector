"""Exact EMA/rollback checks and adversarial finite-check equivalence."""
import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import common
import torch
from torch import nn
import precision
from engine.optim.ema import ModelEMA
import fast_runtime


class State(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, 16))
        self.shared = nn.Parameter(torch.randn(4))
        self.alias = self.shared
        self.register_buffer('anchor', torch.tensor([1., float('inf')]))
        self.register_buffer('empty', torch.empty(0))
        self.register_buffer('counter', torch.tensor(3))
        self.register_buffer('mask', torch.tensor([True, False]))


class Fast(unittest.TestCase):
    def test_ema_matches_reference_bit_for_bit_with_shared_state(self):
        torch.manual_seed(7)
        source = State()
        reference = ModelEMA(source)
        candidate = copy.deepcopy(reference)
        reference.start = candidate.start = 2
        for step in range(20):
            with torch.no_grad():
                for parameter in source.parameters():
                    parameter.add_(torch.randn_like(parameter) * .01)
            reference.update(source)
            fast_runtime.ema_update(candidate, source)
            for key, expected in reference.module.state_dict().items():
                self.assertTrue(torch.equal(expected, candidate.module.state_dict()[key]), (step, key))
            self.assertEqual(reference.updates, candidate.updates)
            self.assertEqual(reference.before_start, candidate.before_start)

    def test_ema_cache_after_checkpoint_load(self):
        source = State()
        reference, candidate = ModelEMA(source), ModelEMA(source)
        candidate.load_state_dict(reference.state_dict())
        fast_runtime.ema_update(candidate, source)
        candidate.load_state_dict(reference.state_dict())
        with torch.no_grad():
            source.weight.mul_(.5)
        reference.update(source)
        fast_runtime.ema_update(candidate, source)
        for key, expected in reference.module.state_dict().items():
            self.assertTrue(torch.equal(expected, candidate.module.state_dict()[key]), key)

    def test_finite_check_preserves_all_decisions_and_does_not_modify_graph(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            for value in [0., 1e-30, 1e30, float('inf'), -float('inf'), float('nan')]:
                tensor = torch.tensor([value, 0.], dtype=dtype, requires_grad=True)
                original_version = tensor._version
                tree = {'main': tensor, 'other': [torch.empty(0), torch.ones(2, dtype=torch.int64)]}
                self.assertEqual(precision.invalid_outputs(tree), fast_runtime.invalid_outputs(tree), (dtype, value))
                self.assertEqual(tensor._version, original_version)
        x = torch.randn(4, requires_grad=True)
        self.assertFalse(fast_runtime.invalid_outputs(x))
        x.square().sum().backward()
        torch.testing.assert_close(x.grad, 2*x, rtol=0, atol=0)

    def test_snapshot_restores_mixed_buffers_and_rng_exactly(self):
        module = State()
        original = {k: x.clone() for k,x in module.named_buffers()}
        snapshot = fast_runtime.ForwardSnapshot(module, torch.device('cpu'))
        expected_random = torch.rand(8)
        with torch.no_grad():
            module.anchor.zero_()
            module.counter.add_(10)
            module.mask.logical_not_()
        snapshot.restore()
        self.assertTrue(torch.equal(expected_random, torch.rand(8)))
        for key, value in module.named_buffers():
            self.assertTrue(torch.equal(value, original[key]), key)

    def test_scale_guard_rejects_invalid_scales(self):
        class Scaler:
            def __init__(self, scale): self.scale=scale
            def get_scale(self): return self.scale
        for value in [0., 1e-12, float('inf'), float('nan')]:
            with self.assertRaises(FloatingPointError):
                fast_runtime.scaler_guard(Scaler(value))
        fast_runtime.scaler_guard(Scaler(.25))


if __name__ == '__main__':
    unittest.main()
