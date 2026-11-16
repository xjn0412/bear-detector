"""Run the unchanged Bear trainer with validation AP patience and resumable state."""
import inspect
import runpy
import sys
from pathlib import Path

repo=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(repo),str(repo/'source'),str(repo/'tools')]
import backbone,compat
import engine.solver.det_solver as solver

source=inspect.getsource(solver.DetSolver.fit)
source=source.replace('        start_time = time.time()', """        benchmark_best = float("-inf")
        benchmark_stale = 0
        history_path = self.output_dir / "log.txt"
        if history_path.exists():
            for line in history_path.read_text().splitlines():
                previous = json.loads(line)
                if previous["epoch"] > self.last_epoch:
                    raise ValueError("History is ahead of resume checkpoint")
                previous_ap = previous["test_coco_eval_bbox"][0]
                if previous_ap > benchmark_best:
                    benchmark_best, benchmark_stale = previous_ap, 0
                    historical_best_epoch = previous["epoch"]
                else:
                    benchmark_stale += 1
            if benchmark_best > float("-inf"):
                top1 = benchmark_best
                best_stat = {"epoch": historical_best_epoch, "coco_eval_bbox": benchmark_best}
                best_stat_print = best_stat.copy()
                print("BENCHMARK_RESUME_HISTORY", self.last_epoch + 1, benchmark_best, benchmark_stale, flush=True)
        start_time = time.time()""")
marker='        total_time = time.time() - start_time'
assert source.count(marker)==1
source=source.replace(marker, '''            benchmark_ap = float(test_stats['coco_eval_bbox'][0])
            if benchmark_ap > benchmark_best:
                benchmark_best, benchmark_stale = benchmark_ap, 0
            else:
                benchmark_stale += 1
            if benchmark_stale >= 20:
                print('BENCHMARK_EARLY_STOP patience=20', epoch + 1, flush=True)
                break

'''+marker)
namespace={}
exec('if True:\n'+source,solver.__dict__,namespace)
solver.DetSolver.fit=namespace['fit']
runpy.run_path(str(repo/'tools/train_custom.py'),run_name='__main__')
