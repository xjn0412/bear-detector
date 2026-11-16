"""Measure actual COCO DDP training and full-val wall time at fixed effective batch."""
import argparse
import json
import math
import os
from pathlib import Path
import time
from common import config, weights, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--accumulate', type=int, required=True)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--restore-training-state', action='store_true')
    p.add_argument('--skip-validation', action='store_true')
    a = p.parse_args()
    os.environ.update(BEAR_ACCUMULATE=str(a.accumulate), BEAR_BENCHMARK='1',
                      BEAR_BENCHMARK_WARMUP=str(a.warmup), BEAR_BENCHMARK_STEPS=str(a.steps))
    import torch
    import runtime
    from engine.misc import dist_utils
    runtime.install_training_loop()
    from v2_training import train_epoch, synchronize
    if 'LOCAL_RANK' in os.environ:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist_utils.setup_distributed(seed=42)
    rank, world = dist_utils.get_rank(), dist_utils.get_world_size()
    device = torch.device('cuda', rank)
    torch.set_num_threads(4)
    cfg = config(a.config)
    state, package = weights(a.checkpoint)
    if a.restore_training_state:
        state = package['model']
    cfg.model.load_state_dict(state, strict=True)
    del state
    model = dist_utils.warp_model(cfg.model.to(device), find_unused_parameters=True)
    optimizer = cfg.optimizer
    for group in optimizer.param_groups:
        group['lr'] = min(group['lr'], 1e-5)
    criterion = cfg.criterion.to(device)
    ema = cfg.ema.to(device)
    scaler = cfg.scaler
    if a.restore_training_state:
        if 'precision_policy' in package:cfg.model._bear_precision.load_state_dict(package['precision_policy'])
        optimizer.load_state_dict(package['optimizer'])
        scaler.load_state_dict(package['scaler'])
        ema.load_state_dict(package['ema'])
    del package
    for group in optimizer.param_groups:
        group['lr']=min(group['lr'],1e-5)
    loader = dist_utils.warp_loader(cfg.train_dataloader, shuffle=True)
    loader.set_epoch(5)
    loader.sampler.set_epoch(5)
    torch.cuda.reset_peak_memory_stats()
    stats = train_epoch(False, None, model, criterion, loader, optimizer, device, 5,
                        max_norm=.1, scaler=scaler, ema=ema, print_freq=10)
    # Measure the same complete 5,000-image held-out evaluation for both counts.
    val = dist_utils.warp_loader(cfg.val_dataloader, shuffle=False)
    import engine.solver.det_solver as solver
    synchronize()
    start = time.perf_counter()
    if not a.skip_validation:
        solver.evaluate(ema.module, criterion, cfg.postprocessor.to(device), val, cfg.evaluator, device)
    synchronize()
    validation_seconds = time.perf_counter() - start
    effective = cfg.yaml_cfg['train_dataloader']['total_batch_size'] * a.accumulate
    local = torch.tensor([stats['timed_seconds'], validation_seconds,
                          stats['peak_memory_bytes']], device=device, dtype=torch.float64)
    torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.MAX)
    seconds_per_window = float(local[0]) / stats['measured_windows']
    validation_seconds = float(local[1])
    epoch_train_seconds = math.ceil(118287 / effective) * seconds_per_window
    measured_val_images = len(val.dataset)
    estimated_full_val_seconds = validation_seconds * 5000 / measured_val_images
    if rank == 0:
        report = dict(passed=True, config=a.config, config_sha256=digest(a.config),
                      checkpoint_sha256=digest(a.checkpoint), gpus=world,
                      gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      micro_batch_per_gpu=cfg.yaml_cfg['train_dataloader']['total_batch_size']//world,
                      effective_global_batch=effective, accumulation=a.accumulate,
                      warmup_windows=a.warmup, measured_windows=a.steps,
                      stats=stats, seconds_per_optimizer_window=seconds_per_window,
                      images_per_second=effective/seconds_per_window,
                      estimated_training_seconds_per_epoch=epoch_train_seconds,
                      measured_validation_images=measured_val_images, measured_validation_seconds=validation_seconds,
                      estimated_full_val_seconds=estimated_full_val_seconds,
                      estimated_epoch_seconds_excluding_checkpoint=epoch_train_seconds+estimated_full_val_seconds,
                      peak_memory_bytes_max_rank=int(local[2]),
                      selected_precision=cfg.model._bear_precision.state_dict(),
                      execution_environment={k:os.getenv(k,'') for k in ['BEAR_FAST_RUNTIME','BEAR_BACKBONE_CHECKPOINT','BEAR_COMPILE_NECK']},
                      validation_performed=not a.skip_validation, accuracy_claim=False, scope='COCO timing subset with real augmentation/forward/backward/AdamW/EMA; extrapolate118287 train and5000 val images; checkpoint time excluded')
        if stats['timed_skipped']:
            report['passed'] = False
            report['reason'] = 'Scaled-overflow skips in timed window; do not use this result to select throughput'
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + '\n')
        print('BENCHMARK_RESULT', json.dumps(report), flush=True)
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
