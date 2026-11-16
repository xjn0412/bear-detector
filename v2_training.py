"""V2 accumulation, measured throughput, EMA checkpoints and patience control."""
from contextlib import nullcontext
import csv
import json
import math
import os
from pathlib import Path
import time

import torch
from engine.misc import dist_utils
from precision import distributed_bad


def synchronize():
    torch.cuda.synchronize()
    if dist_utils.is_dist_available_and_initialized():
        torch.distributed.barrier()


def train_epoch(self_lr_scheduler, lr_scheduler, model, criterion, data_loader,
                optimizer, device, epoch, max_norm=0, **kwargs):
    from runtime import finite_outputs, finite_flag, scaler_guard
    model.train()
    criterion.train()
    scaler, ema = kwargs.get('scaler'), kwargs.get('ema')
    accumulation = int(os.getenv('BEAR_ACCUMULATE', '1'))
    if accumulation < 1:
        raise ValueError('Accumulation must be positive')
    micro_count = len(data_loader)
    windows = math.ceil(micro_count / accumulation)
    benchmark = os.getenv('BEAR_BENCHMARK', '0') == '1'
    warmup = int(os.getenv('BEAR_BENCHMARK_WARMUP', '10')) if benchmark else 0
    measured = int(os.getenv('BEAR_BENCHMARK_STEPS', '20')) if benchmark else windows
    completed = successful = skipped = timed_successful = timed_skipped = 0
    loss_sum = 0.
    start = time.perf_counter()
    timed_start = None
    optimizer.zero_grad(set_to_none=True)
    for i, (samples, targets) in enumerate(data_loader):
        if benchmark and completed == warmup and i % accumulation == 0:
            synchronize()
            timed_start = time.perf_counter()
        end_window = (i + 1) % accumulation == 0 or i + 1 == micro_count
        window_size = min(accumulation, micro_count - (i // accumulation) * accumulation)
        samples = samples.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        sync = model.no_sync() if hasattr(model, 'no_sync') and not end_window else nullcontext()
        with sync:
            outputs = finite_outputs(model(samples, targets=targets))
            with torch.autocast(device_type='cuda', enabled=False):
                losses = criterion(outputs, targets, epoch=epoch, step=i,
                                   global_step=epoch * micro_count + i, epoch_step=micro_count)
                loss = sum(losses.values())
                finite_flag(loss, 'loss')
            loss_sum += float(loss.detach())
            if scaler is not None:
                scaler_guard(scaler)
                scaler.scale(loss / window_size).backward()
            else:
                (loss / window_size).backward()
        if not end_window:
            continue
        if scaler is not None:
            scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm if max_norm > 0 else float('inf'))
        bad = distributed_bad(not bool(torch.isfinite(norm)), device)
        if bad:
            if scaler is None:
                raise FloatingPointError('Nonfinite FP32 gradients; optimizer update blocked')
            scale = scaler.get_scale()
            # All ranks skip together; reset native backoff growth tracking.
            state = scaler.state_dict()
            state.update(scale=scale * scaler.get_backoff_factor(), _growth_tracker=0)
            scaler.update(new_scale=state['scale'])
            scaler.load_state_dict(state)
            print('AMP_SKIPPED_UPDATE', epoch, i, scale, scaler.get_scale(), flush=True)
            skipped += 1
            if benchmark and completed >= warmup:
                timed_skipped += 1
        else:
            if self_lr_scheduler:
                lr_scheduler.step(epoch * windows + completed, optimizer)
            before = getattr(optimizer, '_observed_steps', None)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            did_step = before is None or optimizer._observed_steps > before
            if not did_step:
                raise FloatingPointError('Finite-gradient optimizer update was unexpectedly skipped')
            successful += 1
            if benchmark and completed >= warmup:
                timed_successful += 1
            if ema is not None:
                ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        completed += 1
        if completed % int(kwargs.get('print_freq', 20)) == 0:
            print(f'V2_TRAIN epoch={epoch+1} micro={i+1}/{micro_count} updates={successful} '
                  f'skipped={skipped} loss={float(loss.detach()):.5f} lr={optimizer.param_groups[0]["lr"]:.8g}', flush=True)
        if benchmark and completed >= warmup + measured:
            break
    synchronize()
    elapsed = time.perf_counter() - start
    if successful == 0:
        raise FloatingPointError('Entire epoch had zero optimizer updates')
    report = dict(loss=loss_sum / (i + 1), optimizer_updates=successful, skipped_updates=skipped,
                  micro_batches=i + 1, seconds=elapsed,
                  final_loss_scale=scaler.get_scale() if scaler is not None else None)
    if benchmark:
        if completed < warmup + measured:
            raise RuntimeError('Benchmark loader was too short')
        report.update(timed_seconds=time.perf_counter() - timed_start,
                      measured_windows=measured, timed_successful=timed_successful,
                      timed_skipped=timed_skipped, accumulation=accumulation,
                      peak_memory_bytes=torch.cuda.max_memory_allocated())
    return report


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def fit(self):
    from engine.optim.lr_scheduler import FlatCosineLRScheduler
    import engine.solver.det_solver as solver
    self.train()
    cfg = self.cfg
    accumulation = int(os.getenv('BEAR_ACCUMULATE', '1'))
    windows = math.ceil(len(self.train_dataloader) / accumulation)
    scheduler = FlatCosineLRScheduler(self.optimizer, cfg.lr_gamma, windows,
                                    total_epochs=cfg.epoches, warmup_iter=cfg.warmup_iter,
                                    flat_epochs=cfg.flat_epoch, no_aug_epochs=cfg.no_aug_epoch)
    self.lr_scheduler = scheduler
    control = getattr(self, '_v2_control', dict(best_ap=-1., best_epoch=0, stale=0))
    patience = int(os.getenv('BEAR_PATIENCE', '20'))
    if patience and control['stale'] >= patience:
        print('V2_ALREADY_EARLY_STOPPED', control, flush=True)
        return
    for epoch in range(self.last_epoch + 1, cfg.epoches):
        self.train_dataloader.set_epoch(epoch)
        if dist_utils.is_dist_available_and_initialized():
            self.train_dataloader.sampler.set_epoch(epoch)
        started = time.perf_counter()
        training = train_epoch(True, scheduler, self.model, self.criterion,
                               self.train_dataloader, self.optimizer, self.device, epoch,
                               max_norm=cfg.clip_max_norm, print_freq=cfg.print_freq,
                               scaler=self.scaler, ema=self.ema)
        validation_start = time.perf_counter()
        module = self.ema.module if self.ema is not None else self.model
        evaluation, _ = solver.evaluate(module, self.criterion, self.postprocessor,
                                        self.val_dataloader, self.evaluator, self.device)
        validation_seconds = time.perf_counter() - validation_start
        ap, ap50 = evaluation['coco_eval_bbox'][0], evaluation['coco_eval_bbox'][1]
        if not math.isfinite(ap):
            raise FloatingPointError('Nonfinite validation AP')
        improved = ap > control['best_ap']
        control = dict(best_ap=ap if improved else control['best_ap'],
                       best_epoch=epoch+1 if improved else control['best_epoch'],
                       stale=0 if improved else control['stale']+1)
        self._v2_control = control
        self.last_epoch = epoch
        state = self.state_dict()
        if dist_utils.is_main_process():
            out = self.output_dir
            row = dict(epoch=epoch+1, map50_95=ap*100, map50=ap50*100,
                       best_map50_95=control['best_ap']*100, best_epoch=control['best_epoch'],
                       patience_used=control['stale'], training_seconds=training['seconds'],
                       training_loss=training['loss'], loss_scale=training['final_loss_scale'],
                       gpus=dist_utils.get_world_size(),
                       effective_global_batch=cfg.yaml_cfg['train_dataloader']['total_batch_size']*accumulation,
                       validation_seconds=validation_seconds,
                       optimizer_updates=training['optimizer_updates'], skipped_updates=training['skipped_updates'],
                       elapsed_before_checkpoint_seconds=time.perf_counter()-started)
            atomic_save(state, out / 'last.pth')
            if improved:
                atomic_save(state, out / 'best-full.pth')
                atomic_save({'model': module.state_dict(), 'last_epoch': epoch,
                             'metadata': dict(architecture='Bear Detector v2', ema=True,
                                              map50_95=ap*100, map50=ap50*100)}, out / 'best.pt')
            (out / 'status.json').write_text(json.dumps(row, indent=2) + '\n')
            with (out / 'epochs.jsonl').open('a') as output:
                output.write(json.dumps(row) + '\n')
            csv_path = out / 'epochs.csv'
            exists = csv_path.exists()
            with csv_path.open('a', newline='') as output:
                writer = csv.DictWriter(output, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            print('V2_EPOCH', json.dumps(row), flush=True)
        synchronize()
        if patience and control['stale'] >= patience:
            print('V2_EARLY_STOP', epoch+1, control, flush=True)
            break


def install():
    from engine.solver.det_solver import DetSolver
    from engine.solver._solver import BaseSolver
    if getattr(DetSolver, '_bear_v2_fit', False):
        return
    original_state, original_load = BaseSolver.state_dict, BaseSolver.load_state_dict
    def state_dict(self):
        state = original_state(self)
        state['v2_training_control'] = getattr(self, '_v2_control', {})
        return state
    def load_state_dict(self, state):
        original_load(self, state)
        if state.get('v2_training_control'):
            self._v2_control = state['v2_training_control']
    BaseSolver.state_dict, BaseSolver.load_state_dict = state_dict, load_state_dict
    DetSolver.fit = fit
    DetSolver._bear_v2_fit = True
