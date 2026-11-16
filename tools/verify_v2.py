"""Real 640 CUDA train/infer smoke check. Random weights imply no AP claim."""
import argparse
import json
from pathlib import Path
from common import config, Inference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/V2-L.yml')
    p.add_argument('--output', default='results/v2-gpu-smoke.json')
    args = p.parse_args()
    import torch
    import runtime
    from precision import install_precision_policy, invalid_outputs, ForwardSnapshot
    from common import digest
    runtime.install_training_loop()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = config(args.config)
    core = cfg.model.cuda()
    policy_model = install_precision_policy(core, 'auto')
    optimizer = torch.optim.AdamW(core.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    criterion = cfg.criterion.cuda()
    image = torch.randn(1, 3, 640, 640, device='cuda')
    target = [{'labels': torch.tensor([0, 1], device='cuda'),
               'boxes': torch.tensor([[.25, .25, .05, .05], [.6, .6, .3, .4]], device='cuda'),
               'image_id': torch.tensor([1], device='cuda'),
               'orig_size': torch.tensor([640, 640], device='cuda')}]
    records = []
    core.train()
    skipped = []
    step = 0
    for attempt in range(12):
        snapshot = ForwardSnapshot(core, image.device)
        optimizer.zero_grad(set_to_none=True)
        output = policy_model(image, targets=target)
        assert not invalid_outputs(output)
        with torch.autocast('cuda', enabled=False):
            losses = criterion(output, target, epoch=0, step=step, global_step=step, epoch_step=2)
            loss = sum(losses.values())
        assert torch.isfinite(loss) and loss.item() > 0
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [(n, p.grad) for n, p in core.named_parameters() if p.grad is not None]
        assert gradients
        bad = [n for n, g in gradients if not torch.isfinite(g).all()]
        if bad:
            before = core.backbone.context_scale.detach().clone()
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            assert torch.equal(before, core.backbone.context_scale), 'overflow changed parameters'
            assert scaler.get_scale() < scale, 'overflow did not lower loss scale'
            skipped.append(dict(attempt=attempt, scale_before=scale, scale_after=scaler.get_scale(), nonfinite_gradients=bad[:10]))
            snapshot.restore()
            print('SKIPPED_SCALED_OVERFLOW', skipped[-1], flush=True)
            continue
        for prefix in ['backbone.body.', 'backbone.context.', 'encoder.fpn_blocks.2.', 'encoder.pan_blocks.0.', 'decoder.']:
            assert any(n.startswith(prefix) and g.abs().sum().item() > 0 for n, g in gradients), prefix
        norm = torch.nn.utils.clip_grad_norm_(core.parameters(), .1)
        before = core.backbone.context_scale.detach().clone()
        scaler.step(optimizer)
        scaler.update()
        assert not torch.equal(before, core.backbone.context_scale), 'optimizer did not update context gates'
        records.append(dict(step=step, loss=float(loss.detach()), grad_norm=float(norm), scale=scaler.get_scale()))
        print('TRAIN_STEP', records[-1], flush=True)
        step += 1
        if step == 2:
            break
    assert step == 2, 'could not complete two finite optimizer updates'
    del output, losses, loss, gradients, optimizer, criterion
    core.eval()
    wrapper = Inference.create(core, cfg.postprocessor, amp=True).cuda().eval()
    sizes = torch.tensor([[640, 640]], device='cuda')
    cases = []
    with torch.no_grad():
        for label, sample in [('normal', image), ('zeros', torch.zeros_like(image)), ('tiny', image * 1e-8), ('large', image * 1000)]:
            prediction = wrapper(sample, sizes)
            assert not invalid_outputs(prediction)
            cases.append(label)
    report = dict(passed=True, config=args.config, config_sha256=digest(args.config), torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  training=records, skipped_scaled_overflow=skipped, training_precision=core._bear_precision.state_dict(),
                  inference_cases=cases, inference_precision=wrapper.precision_report,
                  peak_memory_bytes=torch.cuda.max_memory_allocated(),
                  weights='seed42 random initialization; two synthetic updates',
                  limitations='Smoke check only; no dataset AP, convergence, TensorRT parity or universal numeric guarantee')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
