import unittest,torch
import global84_temporal_from_zero as z
class Tests(unittest.TestCase):
 def test_stages(self):self.assertEqual(z.STAGES,(0,100,400,800,1200,1600,2000))
 def test_budget(self):
  rows=[dict(module=f'{i}.{kind}',shape=[d,2048],effective_rank=float(i+1)) for i in range(42) for kind,d in [('q_proj',2048),('v_proj',256)]]
  ranks,total=z.allocator(rows);self.assertLessEqual(total,480256);self.assertEqual(total,sum(ranks[r['module']]*sum(r['shape']) for r in rows))
 def test_linear_continuity(self):
  p=torch.nn.Parameter(torch.ones(1));o=torch.optim.AdamW([p],lr=5e-5);s=torch.optim.lr_scheduler.LambdaLR(o,lambda k:max(0,1-k/2000))
  for i in range(2000):p.sum().backward();o.step();s.step();o.zero_grad()
  self.assertEqual(s.last_epoch,2000);self.assertEqual(o.param_groups[0]['lr'],0.);self.assertEqual(int(o.state[p]['step']),2000)
 def test_probe_preserves_following_training(self):
  import copy
  from test_minicpm_single_layer_rank_gain import TinyLM,examples
  torch.manual_seed(17);original=TinyLM();original.target.lora_dropout=torch.nn.Dropout(.1)
  outputs=[]
  for probe in (False,True):
   model=copy.deepcopy(original);torch.manual_seed(42)
   for p in model.parameters():p.requires_grad_(False)
   factors=[model.target.weight_a,model.target.weight_b]
   for p in factors:p.requires_grad_(True)
   optimizer=torch.optim.AdamW(factors,lr=5e-5);losses=[]
   for step,example in enumerate(examples()):
    model.train();loss=model(**z.w.shared.batch(example,'cpu')).loss;loss.backward();optimizer.step();optimizer.zero_grad();losses.append(float(loss.detach()))
    if probe and step==0:z.w.core.collect_weight_gradients(model,{'target':model.target},[z.w.shared.batch(example,'cpu')])
   outputs.append((copy.deepcopy(model.state_dict()),losses,copy.deepcopy(optimizer.state_dict())))
  self.assertEqual(outputs[0][1],outputs[1][1])
  for k,v in outputs[0][0].items():self.assertTrue(torch.equal(v,outputs[1][0][k]))
  for key,state in outputs[0][2]['state'].items():
   for k,v in state.items():self.assertTrue(torch.equal(v,outputs[1][2]['state'][key][k]))
if __name__=='__main__':unittest.main()
