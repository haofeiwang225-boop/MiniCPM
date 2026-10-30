import unittest,tempfile
from pathlib import Path
import torch
import global84_temporal_rank_demand as t
from test_minicpm_single_layer_rank_gain import TinyLM
class Tests(unittest.TestCase):
 def test_decomposition(self):
  a=torch.eye(8)[:2];b=torch.eye(8)[:,:2];g=torch.randn(8,8,dtype=torch.float64)
  v=t.m.random_frame(t.m.basis(a.T),4,42)
  r,_,_=t.metrics(g,a,b,v);self.assertAlmostEqual(r['CT']+r['ER'],1)
 def test_zero_and_deficient(self):
  z=torch.zeros(8,2);r,u,v=t.metrics(torch.zeros(8,8),z.T,z,torch.eye(8)[:,:4])
  self.assertTrue(r['zero_gradient']);self.assertEqual(r['CT'],0);self.assertIsNone(t.overlap(u,u,1))
 def test_overlap(self):
  v=torch.eye(8,dtype=torch.float64);self.assertEqual(t.overlap(v[:,:4],v[:,:4],4),1)
  self.assertEqual(t.overlap(v[:,:4],v[:,4:],4),0)
 def test_turnover(self):
  self.assertEqual(t.turnover({'x':4,'y':4},{'x':0,'y':8})['rank_turnover'],.5)
 def test_snapshot(self):
  model=TinyLM();layers={'target':model.target};before=t.snapshot(layers)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'adapter.pt';torch.save(before,p);h=t.w.shared.sha256(p)
   loaded=torch.load(p,weights_only=True)
   for k in before['target']:self.assertTrue(torch.equal(loaded['target'][k],getattr(model.target,k)))
   self.assertEqual(h,t.w.shared.sha256(p))
 def test_spearman(self):
  self.assertAlmostEqual(t.spearman([1,2,3],[3,2,1]),-1)
if __name__=='__main__':unittest.main()
