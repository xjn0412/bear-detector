import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from common import categories,normalize,transfer,is_class_parameter
import torch
class Classes(unittest.TestCase):
 def test_invalid_annotation_rejected(self):
  d={'categories':[{'id':3,'name':'a'}],'images':[{'id':1}],'annotations':[{'id':1,'image_id':1,'category_id':3,'bbox':[0,0,float('nan'),2]}]}
  with self.assertRaises(ValueError):normalize(d,categories(d))
  d['annotations'][0]['bbox']=[0,0,0,2]
  with self.assertRaises(ValueError):normalize(d,categories(d))
  with self.assertRaises(ValueError):categories({'categories':[{'id':'3','name':'a'}]})
 def test_sparse_categories_and_empty_validation_class(self):
  schema=[{'id':3,'name':'a'},{'id':77,'name':'b'}]
  d={'categories':schema[::-1],'images':[{'id':1}],'annotations':[{'id':1,'image_id':1,'category_id':77,'bbox':[0,0,1,1]}]}
  out=normalize(d,categories(d));self.assertEqual(out['annotations'][0]['category_id'],1)
  self.assertEqual([c['id'] for c in out['categories']],[0,1])
  d['categories']=schema[:1];self.assertEqual(len(normalize(d,schema)['categories']),2)
 def test_unknown_rejected(self):
  with self.assertRaises(ValueError):normalize({'categories':[{'id':7,'name':'wrong'}],'images':[],'annotations':[]},[{'id':7,'name':'a'}])
 def test_same_count_different_classes_reset(self):
  src={'backbone.x':torch.ones(2),'decoder.dec_score_head.0.weight':torch.ones(3,2)}
  dst={'backbone.x':torch.zeros(2),'decoder.dec_score_head.0.weight':torch.zeros(3,2)}
  state,reset=transfer(dst,src);self.assertTrue(torch.equal(state['backbone.x'],src['backbone.x']));self.assertEqual(state[reset[0]].sum(),0)
 def test_unsupported_shape_rejected(self):
  with self.assertRaises(ValueError):transfer({'backbone.x':torch.zeros(3)},{'backbone.x':torch.zeros(2)})
 def test_heads_and_postprocess(self):
  from engine.deim.deim_decoder import DEIMTransformer
  from engine.deim.postprocessor import PostProcessor
  for k in [1,3,80,123]:
   dec=DEIMTransformer(num_classes=k,feat_channels=[256]*3,hidden_dim=256,num_layers=2,num_queries=10,num_denoising=10,eval_spatial_size=[640,640])
   self.assertEqual(dec.enc_score_head.out_features,k)
   self.assertTrue(all(x.out_features==k for x in dec.dec_score_head))
   self.assertEqual(dec.denoising_class_embed.num_embeddings,k+1)
   post=PostProcessor(num_classes=k,num_top_queries=300).deploy()
   labels,boxes,scores=post({'pred_logits':torch.zeros(1,10,k),'pred_boxes':torch.tensor([.5,.5,.2,.4]).repeat(1,10,1)},torch.tensor([[100,200]]))
   self.assertLess(int(labels.max()),k);self.assertEqual(scores.shape[1],min(300,10*k))
   self.assertTrue(torch.allclose(boxes[0,0],torch.tensor([40.,60.,60.,140.])))
if __name__=='__main__':unittest.main()
