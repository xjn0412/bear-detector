"""Convolutional pyramid with self-attention ONLY at stride32 (400 tokens)."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import timm
from timm.layers import DropPath
from engine.core import register
from engine.backbone.hgnetv2 import HGNetv2


def conv_norm(cin,cout,kernel=1,stride=1,groups=1):
    return nn.Sequential(nn.Conv2d(cin,cout,kernel,stride,kernel//2,
        groups=groups,bias=False),nn.GroupNorm(32,cout),nn.SiLU())


class DeepStageBlock(nn.Module):
    def __init__(self,dim=384,heads=6,drop_path=.05):
        super().__init__()
        assert dim%heads==0
        self.dim=dim;self.heads=heads;self.head_dim=dim//heads
        self.position=nn.Conv2d(dim,dim,3,padding=1,groups=dim)
        self.norm1=nn.LayerNorm(dim);self.norm2=nn.LayerNorm(dim)
        self.qkv=nn.Linear(dim,dim*3);self.proj=nn.Linear(dim,dim)
        # SwiGLU hidden width2d, without attention on the high-resolution maps.
        self.ffn_in=nn.Linear(dim,dim*4);self.ffn_out=nn.Linear(dim*2,dim)
        self.drop_path=DropPath(drop_path)
        self.last_tokens=None

    def forward(self,x):
        b,c,h,w=x.shape
        if (h,w)!=(20,20):raise ValueError('Backbone attention must stay at stride32 of640')
        x=x+self.position(x)
        tokens=x.flatten(2).transpose(1,2);self.last_tokens=h*w
        qkv=self.qkv(self.norm1(tokens)).reshape(b,h*w,3,self.heads,self.head_dim)
        q,k,v=qkv.permute(2,0,3,1,4).unbind(0)
        attn=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
        tokens=tokens+self.drop_path(self.proj(attn.transpose(1,2).reshape(b,h*w,c)))
        value,gate=self.ffn_in(self.norm2(tokens)).chunk(2,dim=-1)
        tokens=tokens+self.drop_path(self.ffn_out(value*F.silu(gate)))
        return tokens.transpose(1,2).reshape(b,c,h,w)


class DeepestContext(nn.Module):
    def __init__(self,channels,dim=384,layers=2):
        super().__init__()
        self.down=conv_norm(channels,dim)
        self.blocks=nn.Sequential(*[DeepStageBlock(dim,6) for _ in range(layers)])
        self.up=nn.Conv2d(dim,channels,1,bias=False)
        # Preserve the pretrained CNN path while new context layers learn.
        self.scale=nn.Parameter(torch.full((channels,),.1))

    def forward(self,x):return x+self.scale.view(1,-1,1,1)*self.up(self.blocks(self.down(x)))


@register()
class DeepStageHybrid(nn.Module):
    def __init__(self,variant='convnext_small',weights_path='',hidden_dim=256,
                 attention_dim=384,attention_layers=2,detail_channels=64):
        super().__init__()
        self.variant=variant
        if variant.startswith('convnext_'):
            if weights_path and not Path(weights_path).is_file():raise FileNotFoundError(weights_path)
            self.body=timm.create_model(variant+'.dinov3_lvd1689m',pretrained=False,
                num_classes=0,checkpoint_path=weights_path or None,drop_path_rate=.1)
            # The classification pooling/norm head is outside dense feature extraction.
            self.body.head=nn.Identity()
            channels=[int(info['num_chs']) for info in self.body.feature_info]
        elif variant=='hgnet_b4':
            # Speed/shape control ONLY, not an accuracy-trained candidate.
            self.body=HGNetv2(name='B4',pretrained=False,return_idx=[0,1,2,3],
                freeze_at=-1,freeze_norm=False)
            channels=[128,512,1024,2048]
        else:raise ValueError(variant)
        self.context=DeepestContext(channels[-1],attention_dim,attention_layers)
        self.projections=nn.ModuleList([conv_norm(c,hidden_dim) for c in channels[1:]])
        # P2 detail contributes to P3 through cheap convolutions. No P2 attention
        # and no stride4 decoder memory map are introduced.
        self.detail=nn.Sequential(conv_norm(channels[0],detail_channels),
            conv_norm(detail_channels,detail_channels,3,2,detail_channels),
            conv_norm(detail_channels,hidden_dim))
        self.detail_scale=nn.Parameter(torch.tensor(.1))

    def forward(self,x):
        if x.shape[-2:]!=(640,640):raise ValueError('Exact640x640 input required')
        if self.variant.startswith('convnext_'):
            features=self.body.forward_intermediates(x,indices=[0,1,2,3],norm=True,
                output_fmt='NCHW',intermediates_only=True)
        else:features=self.body(x)
        c2,c3,c4,c5=features;c5=self.context(c5)
        p3,p4,p5=[proj(f) for proj,f in zip(self.projections,[c3,c4,c5])]
        p3=p3+self.detail_scale*self.detail(c2)
        return p3,p4,p5
