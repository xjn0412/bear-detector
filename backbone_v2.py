"""Bear v2: full-resolution P2-P5 with pooled multiscale FP32 context.

Imported by backbone.py so every existing training/export entry point registers
the v2 modules. No global attention is performed on the 34,000 detection cells.
"""
from pathlib import Path

import timm
import torch
from torch import nn
from torch.nn import functional as F

from engine.core import register
from engine.deim.deim_decoder import DEIMTransformer
from engine.deim.hybrid_encoder import HybridEncoder


class SafeProjection(nn.Sequential):
    """FP32 convolution AND normalization: casting after overflow is too late."""
    _keep_fp32 = True

    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 1, bias=False),
                         nn.GroupNorm(32, cout), nn.SiLU())

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float())


class PooledContextBlock(nn.Module):
    """Pre-LN SDPA and SwiGLU; all arithmetic remains FP32 under outer AMP."""
    _keep_fp32 = True

    def __init__(self, dim=256, heads=8):
        super().__init__()
        if dim % heads:
            raise ValueError('Context width must be divisible by head count')
        self.dim, self.heads = dim, heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.ffn_in, self.ffn_out = nn.Linear(dim, 4 * dim), nn.Linear(2 * dim, dim)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            b, n, d = x.shape
            qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
            x = x + self.proj(attended.transpose(1, 2).reshape(b, n, d))
            value, gate = self.ffn_in(self.norm2(x)).chunk(2, dim=-1)
            return x + self.ffn_out(value * F.silu(gate))


class FourScaleContext(nn.Module):
    _keep_fp32 = True

    def __init__(self, channels, dim=256, layers=2, heads=8, pool_size=20):
        super().__init__()
        if len(channels) != 4 or pool_size < 1 or dim % 32 or layers < 1:
            raise ValueError('Require four scales, positive pool/layers, and width divisible by 32')
        self.pool_size = pool_size
        self.projections = nn.ModuleList([SafeProjection(c, dim) for c in channels])
        self.blocks = nn.Sequential(*[PooledContextBlock(dim, heads) for _ in range(layers)])
        self.scale_embed = nn.Parameter(torch.empty(4, 1, dim))
        nn.init.normal_(self.scale_embed, std=.02)
        position = HybridEncoder.build_2d_sincos_position_embedding(pool_size, pool_size, dim)
        self.register_buffer('position', position)
        self.norm = nn.LayerNorm(dim)

    def forward(self, features):
        with torch.autocast(device_type=features[0].device.type, enabled=False):
            tokens = []
            for i, (feature, project) in enumerate(zip(features, self.projections)):
                pooled = F.adaptive_avg_pool2d(feature.float(), self.pool_size)
                t = project(pooled).flatten(2).transpose(1, 2)
                tokens.append(t + self.position + self.scale_embed[i])
            context = self.norm(self.blocks(torch.cat(tokens, dim=1)))
            n = self.pool_size ** 2
            return tuple(t.transpose(1, 2).reshape(t.shape[0], -1, self.pool_size, self.pool_size)
                         for t in context.split(n, dim=1))


@register()
class FourScaleHybrid(nn.Module):
    """Keep all four ConvNeXt feature maps; replace v1's C5-only context."""
    def __init__(self, variant='convnext_large', weights_path='', hidden_dim=256,
                 attention_layers=2, attention_heads=8, pool_size=20, gradient_checkpointing=True):
        super().__init__()
        if weights_path and not Path(weights_path).is_file():
            raise FileNotFoundError(weights_path)
        if variant not in ('convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'):
            raise ValueError('Unsupported ConvNeXt variant')
        self.variant = variant
        self.body = timm.create_model(variant + '.dinov3_lvd1689m', pretrained=False,
                                     num_classes=0, checkpoint_path=weights_path or None,
                                     drop_path_rate=.1)
        self.body.head = nn.Identity()
        self.body.set_grad_checkpointing(gradient_checkpointing)
        channels = [int(info['num_chs']) for info in self.body.feature_info]
        self.projections = nn.ModuleList([SafeProjection(c, hidden_dim) for c in channels])
        self.context = FourScaleContext(channels, hidden_dim, attention_layers,
                                        attention_heads, pool_size)
        self.context_scale = nn.Parameter(torch.full((4, hidden_dim), .1))

    def forward(self, x):
        if x.shape[-2:] != (640, 640):
            raise ValueError('Exact 640x640 input required')
        features = self.body.forward_intermediates(x, indices=[0, 1, 2, 3], norm=True,
                                                   output_fmt='NCHW', intermediates_only=True)
        context = self.context(features)
        # Preserve small context residuals instead of rounding the sum to FP16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return tuple(project(f) + self.context_scale[i].view(1, -1, 1, 1) *
                         F.interpolate(c, size=f.shape[-2:], mode='bilinear', align_corners=False)
                         for i, (f, c, project) in enumerate(zip(features, context, self.projections)))


@register()
class FourScaleTransformer(DEIMTransformer):
    """FP32 decoder in every entry point, including direct model and export."""
    _keep_fp32 = True

    def forward(self, feats, targets=None):
        for layer in self.decoder.layers:
            layer._bear_v2 = True
        with torch.autocast(device_type=feats[0].device.type, enabled=False):
            return super().forward([f.float() for f in feats], targets)

    def _generate_anchors(self, spatial_shapes=None, grid_size=.025,
                          dtype=torch.float32, device='cpu'):
        # Adding P2 must not double the existing P3/P4/P5 anchor sizes.
        return super()._generate_anchors(spatial_shapes, grid_size, dtype, device)
