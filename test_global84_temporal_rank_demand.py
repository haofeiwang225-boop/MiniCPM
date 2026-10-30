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

class AuditTests(unittest.TestCase):
 def test_known_spectrum_and_energy(self):
  g=torch.diag(torch.tensor([4.,3.,2.,1.,0.,0.],dtype=torch.float64));a=torch.zeros(1,6);b=torch.zeros(6,1)
  r,_,_=t.metrics(g,a,b,torch.eye(6,dtype=torch.float64)[:,:4])
  self.assertEqual(r['singular_values'],[4.,3.,2.,1.])
  torch.testing.assert_close(torch.tensor(r['M_full']),r['ER']*torch.tensor(r['Q_residual']))
  self.assertEqual(r['M'],r['M_full'])
 def test_orthogonalized_overlaps(self):
  x=torch.eye(8,dtype=torch.float64)[:,:4]
  self.assertAlmostEqual(t.overlap(x,x*3,4),1)
  self.assertIsNone(t.overlap(x,x[:,[0,0,0,0]],4))
 def test_normalization_is_scale_invariant(self):
  g=torch.eye(8,dtype=torch.float64);a=torch.zeros(1,8);b=torch.zeros(8,1)
  frames={f:g[:,:4] for f in t.m.FAMILIES}
  one=t.normalized_scores(g,[g]*4,[1]*4,a,b,frames)
  two=t.normalized_scores(g*7,[g*3]*4,[1]*4,a,b,frames)
  for kind in one:
   for key in one[kind]:torch.testing.assert_close(torch.tensor(one[kind][key]),torch.tensor(two[kind][key]))
 def test_decay_not_learning(self):
  model=TinyLM();layers={'x':model.target};initial=t.snapshot(layers)
  stats={'x':{k:dict(max_gradient_norm=0.,nonzero_gradient_steps=0) for k in initial['x']}}
  for p in (model.target.weight_a,model.target.weight_b):p.grad=torch.zeros_like(p)
  t.activity_update(layers,stats)
  with torch.no_grad():model.target.weight_a.mul_(.99)
  r=t.activity_finish(layers,initial,stats)
  self.assertFalse(r['x']['weight_a']['effective_update_observed'])
 def test_three_seed_metrics(self):
  row=dict(CT=.2,ER=.8,M_full=[.1]*4,Q_residual=[.125]*4,
   block_consistency={r:[.5]*6 for r in t.w.ROLES},analysis=dict(candidates={f:dict(audit_U=[2.]*4) for f in t.m.FAMILIES}))
  r=t.aggregate_metrics({s:{(0,'x'):row} for s in (42,43,44)})[0]
  self.assertEqual(r['CT']['n'],3);self.assertAlmostEqual(r['ER']['std'],0.)
  with self.assertRaises(ValueError):t.aggregate_metrics({42:{(0,'x'):row}})
