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

class ConsistencyTests(unittest.TestCase):
 def test_unequal_lengths_equal_sample_gradient(self):
  from test_minicpm_single_layer_rank_gain import TinyLM,examples
  model=TinyLM();batches=[z.w.shared.batch(x,'cpu') for x in examples()]
  batches[0]['labels'][:,:-1]=-100
  expected=[]
  for batch in batches:
   model.eval();model.target.weight.requires_grad_(True)
   loss=model(**batch).loss;g,=torch.autograd.grad(loss,model.target.weight);expected.append(g)
  actual,profile=z.collect_equal_gradients(model,{'target':model.target},batches)
  torch.testing.assert_close(actual['target'],torch.stack(expected).mean(0))
  self.assertEqual(profile['forward_calls'],len(batches));self.assertEqual(profile['sample_weights'],[.5,.5])
 def test_block_weight_binding_is_private(self):
  from unittest.mock import patch
  original=z.w.block_tokens
  def fake(*args):
   return w.block_tokens(None,None),w.analyze_module.__globals__['block_tokens'](None,None),w.core.collect_weight_gradients
  with patch.object(z.t,'stage',fake):
   first,second,collector=z.equal_stage(None,None,None,None,0,42,None,{})
   self.assertEqual(first,[8]*4);self.assertEqual(second,[8]*4)
   self.assertIs(collector,z.collect_equal_gradients)
  self.assertIs(z.w.block_tokens,original)
 def test_history_mismatch(self):
  import json
  hist=json.loads(z.HISTORY.read_text());z.check_history(hist)
  for key in ('lora_rank','lora_alpha','dynamic_length','gradient_checkpointing','direction_correction'):
   altered=dict(hist);altered[key]='wrong'
   with self.assertRaisesRegex(ValueError,key):z.check_history(altered)
 def test_initial_validation(self):
  from types import SimpleNamespace
  layers={f'model.layers.{i}.self_attn.{kind}':SimpleNamespace(lora_rank=2,in_features=8,out_features=4,weight_a=torch.ones(2,8),weight_b=torch.zeros(4,2)) for i in range(42) for kind in ('q_proj','v_proj')}
  ranks={n:2 for n in layers};total=84*24
  self.assertEqual(z.validate_initial(layers,ranks,total),total)
  with self.assertRaises(ValueError):z.validate_initial(layers,ranks,total+1)
  layers[next(iter(layers))].weight_b[0,0]=1
  with self.assertRaisesRegex(ValueError,'zero-B'):z.validate_initial(layers,ranks,total)

class FinalProtocolTests(unittest.TestCase):
 def test_full_pool_order_invariance(self):
  import json,tempfile,hashlib,random
  from pathlib import Path
  rows=[dict(type=f+'_test',original_question=f+'_'+str(i),query=str(i),response='answer') for f in ('GSM','MATH') for i in range(70)]
  rows.append(dict(rows[5],response='alternative'))
  banned={hashlib.sha256(b'GSM_0').hexdigest()}
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'source.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
   samples,metadata,info=z.sample_probe_pool(p,banned)
   random.Random(9).shuffle(rows);p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
   other,meta2,info2=z.sample_probe_pool(p,banned)
   self.assertEqual(samples,other)
   self.assertEqual([x['original_question_sha256'] for x in metadata],[x['original_question_sha256'] for x in meta2])
   self.assertEqual(info['candidate_group_counts'],{'GSM':69,'MATH':70});self.assertEqual(info['scanned_lines'],141)
   self.assertEqual(len({x['original_question_sha256'] for x in metadata}),128)
   self.assertFalse(banned & {x['original_question_sha256'] for x in metadata})
 def test_stage_zero_analysis_independent_of_training_seed(self):
  from unittest.mock import patch
  from test_global84_experiment import inventory
  torch.manual_seed(77)
  rows=[dict(r,d_in=12,d_out=8) for r in inventory()]
  a=torch.randn(2,12);b=torch.zeros(8,2)
  payload=dict(layers={r['module']:dict(state=dict(weight_a=a,weight_b=b)) for r in rows})
  gradients={role:[torch.randn(8,12,dtype=torch.float64) for _ in range(4)] for role in z.w.ROLES}
  plan=dict(common=dict(tokens={r:list(range(1,33)) for r in z.w.ROLES}))
  def fake_stage(model,layers,plan,data,step,seed,out,previous):
   frames={};options={'C':{},'DE':{}}
   for i,row in enumerate(rows):
    frame,record=w.analyze_module(None,None,plan,payload,i,row,seed,'cpu')
    frames[row['module']]=frame
    for kind in options:options[kind][row['module']]=record[kind+'_prefix_options']
   return frames,options,{kind:z.m.exact_stratified_allocation(x,rows)[0] for kind,x in options.items()}
  def blocks(folder,artifacts,role,index,shape,device):return gradients[role]
  with patch.object(z.t,'stage',fake_stage),patch.object(z.w,'load_gradient_blocks',side_effect=blocks):
   results=[]
   for training_seed in (42,43,44):
    torch.manual_seed(training_seed)
    results.append(z.equal_stage(None,None,plan,None,0,training_seed,None,{}))
  for other in results[1:]:
   self.assertEqual(results[0][1:],other[1:])
   for n,frames in results[0][0].items():
    for family,v in frames.items():self.assertTrue(torch.equal(v,other[0][n][family]))
