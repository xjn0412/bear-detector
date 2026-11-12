import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from portable_sampling import grid_sample
import torch
class Sampling(unittest.TestCase):
 def test_bilinear_zeros_edges_and_outside(self):
  torch.manual_seed(42)
  x=torch.randn(2,3,5,7);g=torch.rand(2,11,4,2)*4-2
  g[:,0,:,0]=torch.tensor([-1.,1.,-3.,3.])
  a=torch.nn.functional.grid_sample(x,g,align_corners=False)
  b=grid_sample(x,g,align_corners=False)
  torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)
 def test_reject_unsupported_modes(self):
  with self.assertRaises(ValueError):grid_sample(torch.ones(1,1,2,2),torch.zeros(1,1,1,2),mode='nearest')
if __name__=='__main__':unittest.main()
