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
source=source.replace('        start_time = time.time()', '        benchmark_best = float("-inf")\n        benchmark_stale = 0\n        start_time = time.time()')
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
