"""Batch1 eager CUDA-event latency: GPU inputs through top300; no CUDA Graph.

Preprocessing/host transfers excluded. TensorRT includes device-buffer copies.
Each command must run alone on a GPU with no mining/training/build workload.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
from common import digest
from deployment_backend import load_backend, image_tensor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['torch-fp32', 'torch-amp', 'tensorrt'], required=True)
    for k in ['config', 'checkpoint', 'image', 'output']:
        p.add_argument('--' + k, required=True)
    p.add_argument('--engine')
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--iterations', type=int, default=300)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    if a.warmup < 1 or a.iterations < 1 or a.repeats < 1:
        p.error('warmup/iterations/repeats must be positive')
    if a.backend == 'tensorrt' and not a.engine:
        p.error('--engine required')
    import torch
    import numpy as np
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.mha.set_fastpath_enabled(False)
    forward, nc, extra = load_backend(a.backend, a.config, a.checkpoint, a.engine)
    x, sz = image_tensor(a.image)
    x, sz = x.unsqueeze(0).cuda(), sz.unsqueeze(0).cuda()
    rounds = []
    samples = []
    with torch.inference_mode():
        for _ in range(a.warmup):
            out = forward(x, sz)
        torch.cuda.synchronize()
        if not all(torch.isfinite(t).all() for t in out):
            raise FloatingPointError('Nonfinite benchmark output')
        torch.cuda.reset_peak_memory_stats()
        for _ in range(a.repeats):
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(a.iterations)]
            for start, end in events:
                start.record()
                out = forward(x, sz)
                end.record()
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in events]
            rounds.append(statistics.mean(times))
            samples.extend(times)
    smi = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    process_mib = None
    for line in smi.splitlines():
        values = line.split(',')
        if values[0].strip() == str(os.getpid()):
            process_mib = int(values[1].strip())
    report = dict(backend=a.backend, checkpoint_sha256=digest(a.checkpoint), image_sha256=digest(a.image),
        input=[1, 3, 640, 640], classes=nc, mean_ms=statistics.mean(samples),
        median_ms=statistics.median(samples), p95_ms=float(np.percentile(samples, 95)),
        FPS=1000 / statistics.mean(samples), round_mean_ms=rounds,
        round_mean_cv_pct=100 * statistics.pstdev(rounds) / statistics.mean(rounds),
        warmup=a.warmup, iterations_per_round=a.iterations, repeats=a.repeats,
        torch_peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
        process_gpu_memory_MiB=process_mib, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        timing_scope=__doc__, **extra)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
