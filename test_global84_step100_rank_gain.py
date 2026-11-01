import unittest,copy,torch
import global84_step100_rank_gain as g
from test_minicpm_single_layer_rank_gain import TinyLM,examples
class Tests(unittest.TestCase):
 def test_dropout_zero_output_and_rng(self):
  for rank in (1,3):
   model=TinyLM();model.target.lora_dropout=torch.nn.Dropout(.1);old=model.target
   wrapper=g.SharedDropoutExpansion(old,rank,'target',42)
   for training in (True,False):
    old.train(training);wrapper.train(training);x=torch.randn(2,5,12)
    torch.manual_seed(11);before=old(x);rng=torch.get_rng_state()
    torch.manual_seed(11);after=wrapper(x)
    self.assertTrue(torch.equal(before,after));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
 def test_prefix_directions(self):
  model=TinyLM();a=g.SharedDropoutExpansion(model.target,1,'x',42);b=g.SharedDropoutExpansion(model.target,4,'x',42)
  self.assertTrue(torch.equal(a.add_a,b.add_a[:1]))
 def test_two_steps(self):
  model=TinyLM();old=model.target;state=copy.deepcopy(old.state_dict());scale=old.lora_scaler
  added=g.install(model,{'target':old},{'target':2},42);ps=g.factors(model,{'target':old},added)
  self.assertEqual(len(ps),4);o=torch.optim.AdamW(ps,lr=5e-5)
  for step,example in enumerate(examples()):
   o.zero_grad();model(**g.w.shared.batch(example,'cpu')).loss.backward()
   self.assertGreater(float(added['target'].add_b.grad.norm()),0)
   if step:self.assertGreater(float(added['target'].add_a.grad.norm()),0)
   o.step()
  self.assertEqual(old.lora_scaler,scale);self.assertTrue(torch.equal(old.weight,state['weight']))
 def test_exact_budget_and_order(self):
  rows=[dict(module=f'model.layers.{i}.self_attn.{k}',cost_per_added_rank=c) for i in range(42) for k,c in [('q_proj',4096),('v_proj',2304)]]
  a=g.m.random_allocation(rows,42);self.assertEqual(g.validate_budget(a,rows),25600)
  self.assertEqual(g.order(42),g.order(42));self.assertEqual(len(set(g.order(42))),300)
  a[rows[0]['module']]+=1
  with self.assertRaises(ValueError):g.validate_budget(a,rows)

class EvaluationTests(unittest.TestCase):
 def test_masked_token_weighted_evaluation(self):
  model=TinyLM();data=examples();result=g.w.shared.evaluate(model,data,'cpu')
  sums=[];tokens=[]
  for ex in data:
   batch=g.w.shared.batch(ex,'cpu');logits=model(**batch).logits
   loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),batch['labels'][:,1:].reshape(-1),ignore_index=-100,reduction='sum')
   sums.append(float(loss));tokens.append(int((batch['labels'][:,1:]!=-100).sum()))
  self.assertAlmostEqual(result['token_weighted_nll'],sum(sums)/sum(tokens),places=5)
 def test_multiple_modules_parameter_identity(self):
  model=torch.nn.Module();model.q=TinyLM().target;model.v=TinyLM().target
  layers={'q':model.q,'v':model.v};before={n:(l.weight_a.clone(),l.weight_b.clone(),id(l.weight_a),l.lora_scaler) for n,l in layers.items()}
  added=g.install(model,layers,{'q':3,'v':1},42);params=g.factors(model,layers,added)
  self.assertEqual(len(params),8);self.assertEqual(len({id(p) for p in params}),8)
  for n,l in layers.items():
   a,b,identity,scale=before[n];self.assertTrue(torch.equal(a,l.weight_a));self.assertTrue(torch.equal(b,l.weight_b));self.assertEqual(identity,id(l.weight_a));self.assertEqual(scale,l.lora_scaler)
 def test_statistics_exact_constant_paired_effect(self):
  import tempfile,json
  from pathlib import Path
  from global84_step100_rank_statistics import analyze
  with tempfile.TemporaryDirectory() as d:
   paths=[]
   for seed in (42,43,44):
    p=Path(d)/str(seed);p.mkdir();paths.append(p)
    (p/'report.json').write_text(json.dumps(dict(status='PASS',mode='train',seed=seed,inputs={},code_sha256={})))
    for i,arm in enumerate(g.ARMS):
     (p/arm).mkdir();rows=[dict(source=dict(family='GSM' if j<200 else 'MATH',original_question_sha256=str(j)),effective_tokens=j%3+1,nll_sum=(j%3+1)*(2+i*.1)) for j in range(400)]
     (p/arm/'result.json').write_text(json.dumps(dict(status='PASS',evaluations={'300':{'samples':rows}})))
   r=analyze(paths)['results']['overall']['comparisons']['current_C-random']
   self.assertAlmostEqual(r['mean'],.2);self.assertAlmostEqual(r['ci95'][0],.2);self.assertAlmostEqual(r['ci95'][1],.2)

if __name__=='__main__':unittest.main()
