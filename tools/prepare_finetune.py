"""Create an isolated fine-tuning run from COCO-format data and detector weights."""
import argparse,json,copy
from pathlib import Path
from common import ROOT,categories,normalize,transfer,weights,config,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
 for n in ['train-json','val-json','train-images','val-images']:p.add_argument('--'+n)
 p.add_argument('--output',required=True)
 p.add_argument('--data',help='YOLO detection dataset YAML; alternative to the four COCO arguments')
 p.add_argument('--strict-labels',action='store_true',help='YOLO: reject missing label files rather than treating as background')
 p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch',type=int,default=8);p.add_argument('--lr',type=float,default=1e-4)
 a=p.parse_args()
 if a.epochs<2 or a.batch<1 or a.lr<=0:raise ValueError('Invalid training settings')
 coco_args=[a.train_json,a.val_json,a.train_images,a.val_images]
 if a.data and any(coco_args):p.error('Use --data OR the four COCO arguments')
 if not a.data and not all(coco_args):p.error('Provide --data or all four COCO arguments')
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 if a.data:
  from yolo_to_coco import convert
  converted=convert(a.data,out/'yolo',a.strict_labels)
  for k in ['train_json','val_json','train_images','val_images']:setattr(a,k,converted[k])
 tr=json.loads(Path(a.train_json).read_text());va=json.loads(Path(a.val_json).read_text());schema=categories(tr)
 for name,data,images in [('train',tr,a.train_images),('val',va,a.val_images)]:
  data=normalize(data,schema)
  if not data['images']:raise ValueError('Empty '+name+' split')
  for im in data['images']:
   if not (Path(images)/im['file_name']).is_file():raise FileNotFoundError(im['file_name'])
  (out/(name+'.json')).write_text(json.dumps(data))
 if {x['file_name'] for x in tr['images']}&{x['file_name'] for x in va['images']} and Path(a.train_images).resolve()==Path(a.val_images).resolve():
  raise ValueError('Train/validation images overlap')
 import yaml,torch
 d=config(a.config).yaml_cfg;d=copy.deepcopy(d)
 d.update(num_classes=len(schema),remap_mscoco_category=False,output_dir=str(out/'training'),epoches=a.epochs,device='cuda')
 for n,images in [('train',a.train_images),('val',a.val_images)]:
  loader=d[n+'_dataloader'];loader['total_batch_size']=a.batch
  loader['dataset'].update(ann_file=str(out/(n+'.json')),img_folder=str(Path(images).resolve()))
 # Simpler custom-data recipe: no cached mosaic/mixup schedule inherited from COCO.
 ds=d['train_dataloader']['dataset'];ops=ds['transforms']['ops']
 ds['transforms']={'type':'Compose','ops':[x for x in ops if x['type'] not in ['Mosaic','RandomIoUCrop','RandomZoomOut']]}
 coll=d['train_dataloader']['collate_fn']
 coll.update(mixup_prob=0.,copyblend_prob=0.,stop_epoch=a.epochs,base_size_repeat=None)
 if len(tr['images'])<a.batch:raise ValueError('Batch exceeds training images; reduce --batch')
 d['warmup_iter']=max(1,min(500,(len(tr['images'])//a.batch)*a.epochs//10))
 d['scaler']['init_scale']=1.0
 d['optimizer']['lr']=a.lr;d['flat_epoch']=0;d['no_aug_epoch']=min(5,a.epochs//4)
 d['DEIMCriterion']['matcher']['matcher_change_epoch']=a.epochs
 (out/'config.yml').write_text(yaml.safe_dump(d,sort_keys=False))
 torch.manual_seed(42);cfg=config(out/'config.yml');m=cfg.model
 src,ck=weights(a.checkpoint);adapted,reset=transfer(m.state_dict(),src)
 m.load_state_dict(adapted,strict=True)
 torch.save({'model':m.state_dict(),'last_epoch':-1},out/'initial.pth')
 mapping={'num_classes':len(schema),'classes':[{'index':i,'category_id':c['id'],'name':c['name']} for i,c in enumerate(schema)]}
 (out/'classes.json').write_text(json.dumps(mapping,indent=2))
 (out/'initialization.json').write_text(json.dumps({'source_checkpoint':str(Path(a.checkpoint).resolve()),'source_sha256':digest(a.checkpoint),'reset_parameters':reset,'optimizer_ema_scaler_reset':True,'classes':mapping,'seed':42},indent=2))
 print('Prepared',out,'classes=',len(schema),'Fresh optimizer/EMA/scaler; no prior class rows reused.')
if __name__=='__main__':main()
