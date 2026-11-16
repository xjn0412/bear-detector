"""Differential accumulation and skipped-update/EMA checks on CPU."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import common
import torch
from torch import nn
from torch.utils.data import DataLoader
from v2_training import train_epoch


class InjectInf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient * float('inf')


class Model(nn.Module):
    def __init__(self, inject=False):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        nn.init.constant_(self.conv.weight, .5)
        self.inject = inject

    def forward(self, x, targets=None):
        value = self.conv(x)
        if self.inject:
            self.inject = False
            value = InjectInf.apply(value)
        return {'pred_boxes': value, 'pred_logits': value}


class Criterion(nn.Module):
    def forward(self, outputs, targets, **kwargs):
        expected = torch.stack([t['value'] for t in targets]).reshape(-1, 1, 1, 1)
        return {'loss_bbox': (outputs['pred_boxes'] - expected).square().mean()}


class EMA:
    def __init__(self):
        self.updates = 0

    def update(self, model):
        self.updates += 1


def loader(batch):
    items = [(torch.full((1, 1, 1), float(i)), {'value': torch.tensor(float(i)/3)}) for i in range(1, 5)]
    return DataLoader(items, batch_size=batch, collate_fn=lambda x: (torch.stack([a for a,b in x]), [b for a,b in x]))


class Training(unittest.TestCase):
    def run_epoch(self, batch, accumulation, inject=False):
        torch.set_num_threads(1)
        model, ema = Model(inject), EMA()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        scaler = torch.amp.GradScaler('cpu', init_scale=1024.)
        with patch.dict(os.environ, {'BEAR_ACCUMULATE': str(accumulation), 'BEAR_BENCHMARK': '0'}), patch('v2_training.synchronize'):
            stats = train_epoch(False, None, model, Criterion(), loader(batch), optimizer,
                                torch.device('cpu'), 0, scaler=scaler, ema=ema, print_freq=100)
        return model.conv.weight.detach(), ema, scaler, stats

    def test_accumulation_matches_direct_effective_batch(self):
        direct, ema_direct, _, stats = self.run_epoch(2, 1)
        accumulated, ema_accumulated, _, accumulated_stats = self.run_epoch(1, 2)
        torch.testing.assert_close(direct, accumulated)
        self.assertEqual(ema_direct.updates, 2)
        self.assertEqual(ema_accumulated.updates, 2)
        self.assertEqual(accumulated_stats['optimizer_updates'], 2)

    def test_gradient_overflow_skips_optimizer_and_ema(self):
        value, ema, scaler, stats = self.run_epoch(1, 2, inject=True)
        self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(stats['skipped_updates'], 1)
        self.assertEqual(stats['optimizer_updates'], 1)
        self.assertEqual(ema.updates, 1)
        self.assertEqual(scaler.get_scale(), 512.)


if __name__ == '__main__':
    unittest.main()
