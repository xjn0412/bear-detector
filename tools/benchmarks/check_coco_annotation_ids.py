"""Regression: annotation ID0 must not erase a perfect COCO match."""
import json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from coco_metrics import ground_truth
from common import normalize,categories
from faster_coco_eval import COCOeval_faster

def main():
    data=dict(images=[dict(id=0,width=20,height=20,file_name='unused.png')],
              categories=[dict(id=0,name='defect')],
              annotations=[dict(id=0,image_id=0,category_id=0,bbox=[2,2,10,10],area=100,iscrowd=0)])
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'annotations.json';source=json.dumps(data);path.write_text(source)
        gt=ground_truth(path)
        assert path.read_text()==source
        assert gt.dataset['annotations'][0]==dict(data['annotations'][0],id=1)
        result=[dict(image_id=0,category_id=0,bbox=[2,2,10,10],score=.9)]
        ev=COCOeval_faster(gt,gt.loadRes(result),'bbox')
        ev.evaluate();ev.accumulate();ev.summarize()
        assert abs(ev.stats[0]-1)<1e-10
        normalized=normalize(data,categories(data))
        assert normalized['annotations'][0]['id']==1
        assert data['annotations'][0]['id']==0
        bad=dict(data,annotations=data['annotations']*2);path.write_text(json.dumps(bad))
        try:ground_truth(path)
        except ValueError:pass
        else:raise AssertionError('Duplicate IDs must be rejected')
    print('PASS: source annotation ID0 scores perfect AP after reindex; file/geometry unchanged; duplicate IDs rejected')

if __name__=='__main__':main()
