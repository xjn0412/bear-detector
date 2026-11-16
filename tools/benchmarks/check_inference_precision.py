"""CUDA regression for inference overflow recovery and the static export path."""
import sys
from pathlib import Path
import copy
import warnings
import torch
from torch import nn
sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[2]/'tools')]
from common import Inference

class Decoder(nn.Module):
    def forward(self, features):
        values=features[0].mean((2,3))[:,:4]
        return dict(pred_boxes=values.sigmoid(),pred_logits=values[:,:1]/96000.)

class Encoder(nn.Module):
    def forward(self,x):return (x,)

class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Conv2d(32,32,1,bias=False)
        self.encoder=Encoder()
        self.decoder=Decoder()
        with torch.no_grad():self.backbone.weight.fill_(1.)

class Post(nn.Module):
    def deploy(self):return self
    def forward(self,outputs,sizes):
        boxes=outputs['pred_boxes'][:,None,:]
        return torch.zeros_like(boxes[:,:,0],dtype=torch.int64),boxes,outputs['pred_logits'].sigmoid()

def main():
    core=Core().cuda().eval()
    reference=Inference.create(copy.deepcopy(core),Post(),amp=False).cuda()
    checked=Inference.create(core,Post(),amp=True).cuda()
    x=torch.full((2,32,2,2),3000.,device='cuda')
    sizes=torch.tensor([[640,640],[640,640]],device='cuda')
    calls=[]
    core.backbone.register_forward_hook(lambda m,args,out:calls.append(out.dtype))
    with warnings.catch_warnings(record=True) as caught:
        actual=checked(x,sizes)
    expected=reference(x,sizes)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert calls==[torch.float16,torch.float32]
    assert not checked.amp_enabled and checked.precision_report['fallback_count']==1
    assert len(caught)==1
    checked(x,sizes)
    assert calls[-1]==torch.float32 and len(calls)==3
    assert all(not y.requires_grad for y in actual)
    for module in [checked,reference]:
        try:module(torch.full_like(x,float('nan')),sizes)
        except FloatingPointError:pass
        else:raise AssertionError('Invalid input must raise before inference')
    healthy=Inference.create(Core().cuda(),Post(),amp=True)
    healthy(torch.randn_like(x),sizes)
    assert healthy.amp_enabled and healthy.precision_report['fallback_count']==0

    # The eager fallback must not accidentally become a constant traced branch.
    static=Inference.create(Core(),Post(),checked=False)
    cpu_x=torch.randn(2,32,2,2)
    cpu_sizes=sizes.cpu()
    traced=torch.jit.trace(static,(cpu_x,cpu_sizes))
    for a,b in zip(traced(cpu_x,cpu_sizes),static(cpu_x,cpu_sizes)):
        torch.testing.assert_close(a,b)
    dynamic=Inference.create(Core(),Post(),checked=True)
    try:torch.jit.trace(dynamic,(cpu_x,cpu_sizes))
    except RuntimeError as e:assert 'checked=False' in str(e)
    else:raise AssertionError('Checked Python fallback cannot be embedded in a static trace')
    print('PASS: same-image FP32 retry, persistent fallback, healthy AMP, finite guards, no gradients, static export separation')

if __name__=='__main__':main()
