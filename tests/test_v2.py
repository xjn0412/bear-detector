"""Four-scale geometry, numerical islands and failure-path regression tests."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from common import config

import torch
from torch import nn
import backbone
from backbone_v2 import SafeProjection, FourScaleContext
from inference_precision import CheckedInference
from precision import PrecisionPolicy, ForwardSnapshot, install_precision_policy


class ToyPost(nn.Module):
    def deploy(self):
        return self

    def forward(self, outputs, sizes):
        return outputs['pred_logits'], outputs['pred_boxes']


class ToyCore(nn.Module):
    def __init__(self, corrupt=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.corrupt = corrupt

    def backbone(self, x):
        # An AMP-only corruption tests both overflow and finite signal loss.
        enabled = torch.is_autocast_enabled(x.device.type)
        return x * (self.corrupt if enabled else 1.) * self.weight

    def encoder(self, x):
        return [x]

    def decoder(self, feats):
        return {'pred_logits': feats[0].mean((2, 3)),
                'pred_boxes': feats[0].mean((2, 3))}


class V2(unittest.TestCase):
    def test_full_neck_and_anchor_geometry(self):
        torch.set_num_threads(2)
        cfg = config('configs/V2-T.yml')
        model = cfg.model.eval()
        self.assertEqual(len(model.encoder.fpn_blocks), 3)
        self.assertEqual(len(model.encoder.pan_blocks), 3)
        self.assertEqual(model.decoder.feat_strides, [4, 8, 16, 32])
        self.assertEqual(model.decoder.decoder.layers[0].cross_attn.num_points_list, [4, 4, 2, 2])
        with torch.no_grad():
            anchors, valid = model.decoder._generate_anchors()
            wh = anchors.sigmoid()[0, [12880, 28840, 32820, 33810], 2:]
        torch.testing.assert_close(wh, torch.tensor([[.025]*2, [.05]*2, [.1]*2, [.2]*2]))
        self.assertEqual(anchors.shape[1], 34000)
        self.assertEqual(valid.shape[1], 34000)

    def test_context_preserves_gradients_to_all_scales(self):
        torch.set_num_threads(2)
        context = FourScaleContext([32]*4, dim=32, heads=4, pool_size=2)
        features = [torch.randn(1, 32, s, s, requires_grad=True) for s in [16, 8, 4, 2]]
        outputs = context(features)
        sum((x * torch.randn_like(x)).sum() for x in outputs).backward()
        for feature, output in zip(features, outputs):
            self.assertEqual(output.dtype, torch.float32)
            self.assertTrue(torch.isfinite(feature.grad).all())
            self.assertGreater(feature.grad.abs().sum().item(), 0)

    def test_rng_and_buffers_rollback(self):
        model = nn.BatchNorm2d(3).train()
        snapshot = ForwardSnapshot(model, torch.device('cpu'))
        expected_rng = torch.get_rng_state().clone()
        model(torch.randn(2, 3, 4, 4))
        snapshot.restore()
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertEqual(model.num_batches_tracked.item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_fp32_projection_handles_fp16_overflow_range(self):
        project = SafeProjection(32, 32).cuda()
        x = torch.full((1, 32, 8, 8), 1e6, device='cuda', requires_grad=True)
        with torch.autocast('cuda', dtype=torch.float16):
            output = project(x)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_tiny_projection_gradient_survives_outer_amp(self):
        project = SafeProjection(32, 32).cuda()
        x = torch.randn(1, 32, 4, 4, device='cuda', requires_grad=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = (project(x) * torch.randn_like(x)).sum() * 1e-12
        loss.backward()
        self.assertGreater(x.grad.abs().max().item(), 0)
        self.assertLess(x.grad.abs().max().item(), torch.finfo(torch.float16).smallest_normal)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_inference_fallback_overflow_and_finite_signal_loss(self):
        for corruption in [float('inf'), 0.]:
            wrapper = CheckedInference(ToyCore(corruption).cuda().eval(), ToyPost(), amp=True).eval()
            result = wrapper(torch.ones(1, 3, 4, 4, device='cuda'), torch.tensor([[4, 4]], device='cuda'))
            self.assertTrue(all(torch.isfinite(x).all() for x in result))
            self.assertFalse(wrapper.amp_enabled)
            self.assertEqual(wrapper.precision_report['fallback_count'], 1)
            self.assertTrue(all(torch.equal(x, torch.ones_like(x)) for x in result))

    def test_nonfinite_fp32_input_is_rejected(self):
        wrapper = CheckedInference(ToyCore().eval(), ToyPost()).eval()
        with self.assertRaises(FloatingPointError):
            wrapper(torch.full((1, 3, 4, 4), float('nan')), torch.tensor([[4, 4]]))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_training_overflow_retries_without_double_buffer_update(self):
        class TrainCore(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones((), device='cuda'))
                self.register_buffer('calls', torch.zeros((), device='cuda'))

            def forward(self, x):
                self.calls.add_(1)
                result = x * self.weight
                if torch.is_autocast_enabled('cuda'):
                    result = result * float('inf')
                return {'pred_boxes': result, 'pred_logits': result}

        model = install_precision_policy(TrainCore().train())
        model._bear_precision.checked = True  # specifically exercise late overflow
        output = model(torch.ones((), device='cuda'))
        self.assertEqual(model.calls.item(), 1)
        self.assertFalse(model._bear_precision.amp_enabled)
        self.assertEqual(output['pred_boxes'].item(), 1.)
        output['pred_boxes'].backward()
        self.assertEqual(model.weight.grad.item(), 1.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_scaled_gradient_overflow_blocks_optimizer_update(self):
        parameter = nn.Parameter(torch.ones((), device='cuda'))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
        scaler.scale(parameter.square()).backward()
        parameter.grad.fill_(float('inf'))
        scaler.unscale_(optimizer)
        before = parameter.detach().clone()
        scaler.step(optimizer)
        scaler.update()
        self.assertTrue(torch.equal(parameter, before))
        self.assertEqual(scaler.get_scale(), 512.)


if __name__ == '__main__':
    unittest.main()
