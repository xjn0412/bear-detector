"""Measure reversible NUMA affinity on the running job, without restarting it."""
import argparse
import json
import os
from pathlib import Path
import re
import time
import psutil


def expand(spec):
    out = []
    for part in spec.strip().split(','):
        if '-' in part:
            first, last = map(int, part.split('-'))
            out.extend(range(first, last + 1))
        else:
            out.append(int(part))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir', required=True)
    p.add_argument('--windows', type=int, default=200)
    a = p.parse_args()
    work = Path(a.workdir)
    log = work / 'logs/train-L.log'
    output = work / 'optimization'
    output.mkdir(exist_ok=True)
    def latest():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 20000))
            lines = stream.read().decode(errors='replace').splitlines()
        hits = [re.search(r'epoch=(\d+) micro=(\d+)/', x) for x in lines if x.startswith('V2_TRAIN')]
        return tuple(map(int, hits[-1].groups())) if hits else None
    def measure():
        previous = latest()
        deadline = time.monotonic() + 300
        while latest() == previous:
            if time.monotonic() > deadline:
                raise TimeoutError('Training progress did not advance')
            time.sleep(.1)
        epoch, start_step = latest()
        started = time.monotonic()
        while True:
            current_epoch, step = latest()
            if current_epoch != epoch:
                raise RuntimeError('Epoch boundary invalidated this timing window')
            if step - start_step >= a.windows:
                return dict(epoch=epoch, start=start_step, end=step,
                            seconds=time.monotonic()-started,
                            seconds_per_window=(time.monotonic()-started)/(step-start_step))
            if time.monotonic() > deadline:
                raise TimeoutError('Training progress stopped')
            time.sleep(.1)
    report = dict(scope='Live full-COCO L training, unchanged optimizer/model/data order; two non-overlapping windows are not a controlled GPU profiler')
    report['before'] = measure()
    agent = psutil.Process(int((work / 'active-child.pid').read_text()))
    ranks = [child for child in agent.children() if 'LOCAL_RANK' in child.environ()]
    if len(ranks) != 8:
        raise RuntimeError('Expected exactly the current eight DDP workers')
    allowed = set(os.sched_getaffinity(0))
    saved = {}
    try:
        for rank_process in ranks:
            rank = int(rank_process.environ()['LOCAL_RANK'])
            node, slot = rank // 4, rank % 4
            cpus = [c for c in expand(Path(f'/sys/devices/system/node/node{node}/cpulist').read_text()) if c in allowed][slot::4]
            if not cpus:
                raise RuntimeError('No allowed NUMA-local CPUs')
            for process in [rank_process, *rank_process.children(recursive=True)]:
                for thread in process.threads():
                    try:
                        saved[thread.id] = sorted(os.sched_getaffinity(thread.id))
                        os.sched_setaffinity(thread.id, cpus)
                    except ProcessLookupError:
                        pass
        (output / 'original-affinity.json').write_text(json.dumps(saved))
        report['after'] = measure()
        report['speedup'] = report['before']['seconds_per_window']/report['after']['seconds_per_window']
        report['kept'] = report['speedup'] >= 1.03
        if not report['kept']:
            for tid, cpus in saved.items():
                try:
                    os.sched_setaffinity(tid, cpus)
                except ProcessLookupError:
                    pass
        (output / 'affinity-probe.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    except Exception:
        for tid, cpus in saved.items():
            try:
                os.sched_setaffinity(tid, cpus)
            except ProcessLookupError:
                pass
        raise


if __name__ == '__main__':
    main()
