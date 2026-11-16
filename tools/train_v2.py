"""Supported v2 COCO launcher: checked AMP, FP32 losses and observed updates."""
import argparse
import os
from pathlib import Path
import runpy
import sys
from common import ROOT, config, weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/V2-L.yml')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', help='V2 initialization/detector checkpoint; fresh optimizer')
    group.add_argument('--resume', help='Full training checkpoint, including optimizer/scaler/EMA')
    p.add_argument('--amp', choices=['auto', 'fp32'], default='auto')
    a = p.parse_args()
    os.environ['BEAR_PRECISION'] = a.amp
    cfg = config(a.config)
    if cfg.yaml_cfg['DEIM']['backbone'] != 'FourScaleHybrid':
        p.error('Use a v2 configuration')
    os.environ['BEAR_PRECISION_LOG'] = str(Path(cfg.yaml_cfg['output_dir']) / 'precision-policy.json')
    import torch
    import runtime
    from engine.core import YAMLConfig
    runtime.install_training_loop()
    # Install the same actual-step audit as the custom-data launcher.
    original = YAMLConfig.optimizer

    def observed(cfg):
        opt = original.fget(cfg)
        if not hasattr(opt, '_observed_steps'):
            opt._observed_steps = 0
            def stepped(opt, args, kwargs):
                if getattr(opt, 'found_inf', None) is None or opt.found_inf.item() == 0:
                    opt._observed_steps += 1
            opt.register_step_post_hook(stepped)
        return opt
    YAMLConfig.optimizer = property(observed)
    import engine.solver.det_engine as engine
    import engine.solver.det_solver as solver
    original_epoch = engine.train_one_epoch

    def checked_epoch(*args, **kwargs):
        optimizer = kwargs.get('optimizer', args[5] if len(args) > 5 else None)
        before = optimizer._observed_steps
        result = original_epoch(*args, **kwargs)
        if optimizer._observed_steps == before:
            raise FloatingPointError('Entire epoch had zero optimizer updates')
        return result
    engine.train_one_epoch = solver.train_one_epoch = checked_epoch

    checkpoint = Path(a.resume or a.checkpoint).resolve()
    state, package = weights(checkpoint)
    # Strict checking rejects incompatible v1 detector weights before training.
    cfg.model.load_state_dict(state, strict=True)
    if a.resume:
        required = {'optimizer', 'ema'} | ({'scaler'} if a.amp == 'auto' else set())
        missing = required - set(package)
        if missing:
            raise ValueError('Full resume checkpoint missing: ' + str(sorted(missing)))
    elif any(key in package for key in ['optimizer', 'lr_scheduler', 'scaler']):
        raise ValueError('Use --resume for a full training checkpoint')
    del state, package, cfg
    sys.argv = ['train.py', '-c', a.config, '-r', str(checkpoint), '--seed', '42']
    if a.amp == 'auto':
        sys.argv.append('--use-amp')
    runpy.run_path(str(ROOT / 'source/train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
