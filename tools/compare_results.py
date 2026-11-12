"""Gate a candidate backend against the same checkpoint/subset FP32 reference."""
import argparse,json,sys,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);p.add_argument('--max-drop',type=float,default=.1);a=p.parse_args()
r=json.loads(Path(a.reference).read_text());c=json.loads(Path(a.candidate).read_text())
for k in ['checkpoint_sha256','annotations_sha256','images','input','class_count']:
 if r[k]!=c[k]:raise ValueError('Incomparable results: '+k)
if a.max_drop<0 or not all(math.isfinite(v['AP']) for v in [r,c]):raise ValueError('Invalid AP or tolerance')
delta=c['AP']-r['AP'];passed=delta>=-a.max_drop
report={'passed':passed,'AP_delta':delta,'max_allowed_drop_AP_points':a.max_drop,'reference':r['backend'],'candidate':c['backend'],'images':c['images'],'checkpoint_sha256':c['checkpoint_sha256'],'annotations_sha256':c['annotations_sha256'],'scope':'This fixed subset and checkpoint only; not full COCO or all custom datasets','full_coco_accuracy_validated':False}
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report));sys.exit(0 if passed else 1)
