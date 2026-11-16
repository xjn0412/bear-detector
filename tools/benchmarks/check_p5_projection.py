"""CUDA regression: projection overflow must be prevented before normalization."""
import sys
from pathlib import Path
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[2]/'source')]
from backbone import conv_norm, FP32Projection

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('This regression requires CUDA FP16 autocast')
    torch.manual_seed(42)
    old=conv_norm(32,32).cuda()
    with torch.no_grad():
        old[0].weight.fill_(1)
    fixed=FP32Projection(*conv_norm(32,32)).cuda()
    fixed.load_state_dict(old.state_dict(),strict=True)
    x=torch.full((2,32,2,2),3000.,device='cuda',requires_grad=True)
    with torch.autocast('cuda',dtype=torch.float16):
        broken=old(x)
        result=fixed(x)
    assert not torch.isfinite(broken).all(), 'Fixture did not reproduce FP16 overflow'
    assert torch.isfinite(result).all(), 'FP32 projection must remain finite'
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in fixed.parameters())
    y=torch.randn_like(x).detach()
    with torch.autocast('cuda',dtype=torch.float16):
        actual=fixed(y)
    reference=fixed(y)
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    print('PASS: overflow fixture, finite backward, unchanged state keys, FP32 parity')

if __name__=='__main__':
    main()
