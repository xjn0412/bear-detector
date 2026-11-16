"""Load COCO ground truth with positive annotation IDs for match bookkeeping."""
import json
from faster_coco_eval import COCO

def ground_truth(path):
    data=json.loads(__import__('pathlib').Path(path).read_text())
    original=[a['id'] for a in data['annotations']]
    if len(set(original))!=len(original):raise ValueError('Duplicate annotation IDs')
    data['annotations']=[dict(a,id=i+1) for i,a in enumerate(data['annotations'])]
    data.setdefault('info',{})
    gt=COCO();gt.dataset=data;gt.createIndex()
    return gt
