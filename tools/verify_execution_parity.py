"""CUDA checks for foreach math, backbone recomputation and optional compilation."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from common import config, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--compile',action='store_true')
    a=p.parse_args()
    import torch
    import fast_runtime
    from engine.optim.ema import ModelEMA
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    report=dict(checkpoint_sha256=digest(a.checkpoint),torch=torch.__version__,tests={})
    # CUDA rounding and shared-state handling, not just CPU equivalence.
    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.randn(128,128,device='cuda'))
            self.alias=self.weight
            self.register_buffer('anchor',torch.tensor([float('inf'),1.],device='cuda'))
    source=Shared();reference=ModelEMA(source);candidate=copy.deepcopy(reference)
    for step in range(20):
        with torch.no_grad():source.weight.add_(.001)
        reference.update(source);fast_runtime.ema_update(candidate,source)
        for key,value in reference.module.state_dict().items():
            assert torch.equal(value,candidate.module.state_dict()[key]),('EMA',key)
    report['tests']['cuda_ema_bitwise']=True
    values=[torch.tensor([1e30,0.],device='cuda',requires_grad=True),torch.tensor([float('nan')],device='cuda')]
    import precision
    for value in values:
        version=value._version
        assert precision.invalid_outputs(value)==fast_runtime.invalid_outputs(value)
        assert value._version==version
    report['tests']['cuda_finite_decisions']=True
    snap=fast_runtime.ForwardSnapshot(source,torch.device('cuda'))
    rng=torch.rand(4,device='cuda');source.anchor.zero_();snap.restore()
    assert torch.equal(rng,torch.rand(4,device='cuda'))
    assert torch.isinf(source.anchor[0])
    report['tests']['cuda_rollback']=True
    report['passed']=True
    report['passed_scope']='runtime components; backbone recomputation has separate gates'
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2))
    del reference,candidate,source,values,snap
    torch.cuda.empty_cache()
    cfg=config(a.config);model=cfg.model
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)['model']
    model.load_state_dict(state,strict=True);del state
    image=torch.randn(1,3,640,640)
    expected=None;checkpoint_bitwise=True;checkpoint_close=True;gradient_max_error=0.
    for checkpointed in [True,False]:
        branch=copy.deepcopy(model.backbone).cuda().to(memory_format=torch.channels_last).train()
        branch.body.set_grad_checkpointing(checkpointed)
        torch.manual_seed(77);torch.cuda.manual_seed_all(77)
        x=image.cuda().to(memory_format=torch.channels_last).requires_grad_()
        with torch.autocast('cuda',dtype=torch.float16):outputs=branch(x)
        loss=sum(t.float().square().mean() for t in outputs)
        loss.backward()
        actual=dict(outputs=[t.detach().cpu() for t in outputs],
                    gradients={n:p.grad.detach().cpu() for n,p in branch.named_parameters() if p.grad is not None})
        assert all(torch.isfinite(t).all() for t in actual['gradients'].values())
        if expected is None:expected=actual
        else:
            for actual_tensor,expected_tensor in zip(actual['outputs'],expected['outputs']):
                checkpoint_bitwise &= torch.equal(actual_tensor,expected_tensor)
                checkpoint_close &= torch.allclose(actual_tensor,expected_tensor,rtol=1e-5,atol=1e-6)
            assert actual['gradients'].keys()==expected['gradients'].keys()
            for name,expected_tensor in expected['gradients'].items():
                actual_tensor=actual['gradients'][name]
                checkpoint_bitwise &= torch.equal(actual_tensor,expected_tensor)
                gradient_max_error=max(gradient_max_error,float((actual_tensor-expected_tensor).abs().max()))
                checkpoint_close &= torch.allclose(actual_tensor,expected_tensor,rtol=1e-4,atol=1e-7)
        del branch,x,outputs,loss,actual
        torch.cuda.empty_cache()
    report['tests']['backbone_checkpoint_allclose']=bool(checkpoint_close)
    report['tests']['backbone_checkpoint_bitwise']=bool(checkpoint_bitwise)
    report['backbone_gradient_max_absolute_error']=gradient_max_error
    del expected
    report['passed']=True
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2))
    if a.compile:
        try:
            reference=copy.deepcopy(model.encoder).cuda().train()
            compiled=copy.deepcopy(model.encoder).cuda().train()
            compiled.forward=torch.compile(compiled.forward,mode='default')
            torch.manual_seed(78)
            inputs=[torch.randn(1,256,s,s,device='cuda').to(memory_format=torch.channels_last) for s in [160,80,40,20]]
            def run(module):
                x=[t.detach().clone().requires_grad_() for t in inputs]
                with torch.autocast('cuda',dtype=torch.float16):out=module(x)
                sum(t.float().square().mean() for t in out).backward()
                return [t.detach().cpu() for t in out],{n:p.grad.detach().cpu() for n,p in module.named_parameters() if p.grad is not None}
            ref_out,ref_grad=run(reference);out,grad=run(compiled)
            exact=all(torch.equal(x,y) for x,y in zip(out,ref_out)) and all(torch.equal(grad[n],v) for n,v in ref_grad.items())
            report['tests']['compiled_neck_bitwise']=exact
            report['compile_deployable']=exact
        except Exception as error:
            report['compile_deployable']=False;report['compile_error']=str(error)[:1000]
    report['passed']=True
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
