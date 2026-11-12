import json
from pathlib import Path
import sys
import tempfile
import unittest
from PIL import Image
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from yolo_to_coco import convert, schema


class Yolo(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        for split in ['train','val']:
            (self.root/'images'/split).mkdir(parents=True)
            (self.root/'labels'/split).mkdir(parents=True)
            Image.new('RGB',(200,100)).save(self.root/'images'/split/'a.jpg')
            (self.root/'labels'/split/'a.txt').write_text('1 0.5 0.5 0.4 0.2\n')
        self.data={'path':'.','train':'images/train','val':'images/val','names':{0:'cat',1:'bear'}}
    def tearDown(self):self.tmp.cleanup()
    def run_convert(self):
        path=self.root/'data.yaml';path.write_text(yaml.safe_dump(self.data))
        return convert(path,self.root/'converted')
    def test_pixel_boxes_and_class_mapping(self):
        result=self.run_convert()
        d=json.loads(Path(result['train_json']).read_text())
        self.assertEqual(d['annotations'][0]['bbox'],[60.,40.,80.,20.])
        self.assertEqual(d['annotations'][0]['category_id'],1)
        self.assertEqual(d['categories'],[{'id':0,'name':'cat'},{'id':1,'name':'bear'}])
        self.assertTrue((Path(result['train_images'])/d['images'][0]['file_name']).exists())
    def test_lists_and_relative_text_files(self):
        (self.root/'train.txt').write_text('./images/train/a.jpg\n')
        self.data['train']='train.txt';self.data['val']=['images/val/a.jpg']
        result=self.run_convert();self.assertEqual(len(json.loads(Path(result['val_json']).read_text())['images']),1)
    def test_background_and_strict_labels(self):
        (self.root/'labels/train/a.txt').unlink()
        result=self.run_convert();self.assertEqual(result['missing_labels_as_background']['train'],1)
        with self.assertRaises(FileNotFoundError):convert(self.root/'data.yaml',self.root/'strict',True)
    def test_invalid_boxes_classes_and_other_tasks(self):
        path=self.root/'data.yaml';path.write_text(yaml.safe_dump(self.data))
        for n,line in enumerate(['2 0.5 0.5 0.4 0.2','1 nan 0.5 0.4 0.2','1 0.1 0.5 0.8 0.2','1 0.5 0.5 0.2 0.2 0.3']):
            (self.root/'labels/train/a.txt').write_text(line)
            with self.assertRaises(ValueError):convert(path,self.root/f'bad{n}')
    def test_overlap_rejected(self):
        self.data['val']='images/train'
        with self.assertRaises(ValueError):self.run_convert()
    def test_class_schema_rejected(self):
        for names in [{1:'bear'}, {0.5:'bear'}, ['bear','bear'], []]:
            with self.assertRaises(ValueError):schema(names)


if __name__=='__main__':unittest.main()
