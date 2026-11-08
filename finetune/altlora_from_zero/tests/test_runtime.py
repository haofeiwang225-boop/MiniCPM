import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from altlora.adapters import DenseAdapter,adapter_state,factor_pairs
from altlora.optimizer import OfficialAltLoRA,OfficialSchedule,_Pairs
from altlora.source import official_class
from altlora_from_zero.initialization import attach_pattern,restore_initial
from altlora_from_zero.state import (set_step_rng,rng_fingerprint,preserved,save_checkpoint,load_checkpoint,rng_state)
from altlora_from_zero.protocol import expected_side,file_hash,training_order
from altlora_from_zero.diagnostics import coverage_rows,update_rows
from altlora_from_zero.benchmark import load_official_samples,paired,wilson


def equal_tree(test,a,b):
    if torch.is_tensor(a):test.assertTrue(torch.equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:equal_tree(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal_tree(test,x,y)
    else:test.assertEqual(a,b)


def make_model():
    return nn.Sequential(DenseAdapter(nn.Linear(4,4,bias=False),rank=2,dropout=.1))


class RuntimeTests(unittest.TestCase):
    def test_three_initial_adapters_and_base_output(self):
        torch.manual_seed(42);base=nn.Linear(4,4,bias=False);x=torch.randn(3,4)
        first=DenseAdapter(copy.deepcopy(base),2,dropout=.1);initial=adapter_state({'x':first})
        for _ in range(3):
            layer=DenseAdapter(copy.deepcopy(base),2,dropout=.1)
            restore_initial({'x':layer},initial)
            self.assertTrue(torch.equal(layer.weight_a,initial['x']['A']))
            self.assertTrue(torch.equal(layer.weight_b,initial['x']['B']))
            for training in (True,False):
                layer.train(training);self.assertTrue(torch.equal(layer(x),base(x)))

    def test_pair_dropout_and_measurement_rng(self):
        model=make_model();model[0].weight_b.data.fill_(.2);x=torch.ones(3,4)
        outputs=[];fingerprints=[]
        for _ in range(3):
            set_step_rng(42,7)
            with preserved(model):
                model.eval();torch.rand(100);np.random.rand(12)
            fingerprints.append(rng_fingerprint(rng_state()));outputs.append(model(x))
        self.assertEqual(len(set(fingerprints)),1)
        self.assertTrue(all(torch.equal(outputs[0],y) for y in outputs))
        self.assertTrue(model.training)

    def test_real_official_schedule_2000(self):
        a=nn.Parameter(torch.zeros(1,3));b=nn.Parameter(torch.zeros(4,1))
        for arm,mode in [('vanilla','official-main'),('altlora_false','official-main'),('altlora_true','official-trainer')]:
            schedule=OfficialSchedule([('x',a,b)],mode);counts={'A':0,'B':0}
            for step in range(1,2001):
                self.assertEqual(schedule.prepare(),expected_side(arm,step))
                counts['A']+=int(a.requires_grad);counts['B']+=int(b.requires_grad)
            self.assertEqual(counts,{'A':1009,'B':1000} if arm=='altlora_true' else {'A':2000,'B':2000})

    def test_direct_official_equivalence_mixed_ranks_and_frozen_momentum(self):
        for mode in ('official-main','official-trainer'):
            torch.manual_seed(9)
            pairs=[(str(r),nn.Parameter(torch.randn(r,4)*.1),nn.Parameter(torch.randn(3,r)*.1)) for r in (1,2)]
            copies=[(n,nn.Parameter(a.detach().clone()),nn.Parameter(b.detach().clone())) for n,a,b in pairs]
            bridge=OfficialAltLoRA(pairs,lr=5e-5,weight_decay=.01)
            raw=[official_class()(_Pairs([pair]),rank=pair[1].shape[0],lr=5e-5,betas=(.9,.999),eps=1e-4,weight_decay=.01,reg=1e-5,correct_bias=True) for pair in copies]
            schedule=OfficialSchedule(pairs,mode)
            for step in range(1,43):
                side=schedule.prepare();before=[]
                for (_,a,b),(_,aa,bb) in zip(pairs,copies):
                    for name,p,q in [('A',a,aa),('B',b,bb)]:
                        q.requires_grad_(p.requires_grad)
                        grad=torch.full_like(p,step*.001)
                        p.grad=grad.clone() if p.requires_grad else None;q.grad=grad.clone() if q.requires_grad else None
                        if not p.requires_grad:before.append((p,p.detach().clone(),copy.deepcopy(bridge.state.get(p,{}))))
                bridge.step()
                with contextlib.redirect_stdout(io.StringIO()):
                    for opt in raw:opt.step()
                for (n,a,b),(_,aa,bb),opt in zip(pairs,copies,raw):
                    for p,q in [(a,aa),(b,bb)]:
                        self.assertTrue(torch.equal(p,q));equal_tree(self,bridge.state.get(p,{}),opt.state.get(q,{}))
                for p,value,state in before:
                    self.assertTrue(torch.equal(p,value));equal_tree(self,bridge.state.get(p,{}),state)
                bridge.zero_grad(set_to_none=True)
                for opt in raw:opt.zero_grad(set_to_none=True)
            self.assertEqual(len(bridge.delegates),2)

    def test_checkpoint_exact_next_step(self):
        for arm in ('vanilla','altlora_false','altlora_true'):
            torch.manual_seed(7);model=make_model();layers={'0':model[0]};pairs=factor_pairs(layers)
            def build(layers):
                ps=factor_pairs(layers)
                opt=torch.optim.AdamW([p for _,a,b in ps for p in (a,b)],lr=5e-5,weight_decay=.01) if arm=='vanilla' else OfficialAltLoRA(ps,lr=5e-5,weight_decay=.01)
                return opt,torch.optim.lr_scheduler.LambdaLR(opt,lambda k:1-k/2000),OfficialSchedule(ps,'official-trainer' if arm=='altlora_true' else 'official-main')
            opt,sched,phase=build(layers);order=list(range(2000));identity=dict(arm=arm,pair_id='pair',rank_map_sha256='r',initial_adapter_sha256='i',train_order_sha256='o',base_hashes={},code_hashes={},protocol={})
            def step(m,o,s,p,k):
                p.prepare();set_step_rng(42,k);loss=m(torch.ones(2,4)).square().mean();loss.backward();o.step();s.step();o.zero_grad(set_to_none=True)
            for k in range(1,13):step(model,opt,sched,phase,k)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'checkpoint.pt';save_checkpoint(path,layers,opt,sched,phase,12,order,identity)
                saved_rng=rng_fingerprint(rng_state());step(model,opt,sched,phase,13)
                expected=adapter_state(layers);expected_opt=copy.deepcopy(opt.state_dict())
                # Copy unchanged base, then restore checkpoint factors and states.
                restored=copy.deepcopy(model);rs={'0':restored[0]};o,s,p=build(rs)
                self.assertEqual(load_checkpoint(path,rs,o,s,p,order,identity),12)
                self.assertEqual(rng_fingerprint(rng_state()),saved_rng)
                self.assertEqual(order[12],12)
                step(restored,o,s,p,13);equal_tree(self,adapter_state(rs),expected);equal_tree(self,o.state_dict(),expected_opt)
                with self.assertRaises(ValueError):load_checkpoint(path,rs,o,s,p,order,dict(identity,arm='invalid'))

    def test_saved_order_arbitrary_record_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.json';path.write_text(json.dumps([{'sample':i} for i in range(17)]))
            with patch('altlora_from_zero.protocol.TRAIN_SHA256',file_hash(path)):
                one=training_order(path);two=training_order(path)
            self.assertEqual(one,two);self.assertEqual(len(one['samples']),2000)
            self.assertEqual(len({x['index'] for x in one['samples'][:17]}),17)

    def test_metrics_distinct_fixed_dynamic_and_delta(self):
        layer=DenseAdapter(nn.Linear(4,4,bias=False),1)
        layer.weight_a.data.copy_(torch.tensor([[1.,0,0,0]]))
        gs=dict(question={'x':torch.eye(4)},token={'x':2*torch.eye(4)})
        rows=coverage_rows({'x':layer},gs,gs,0,'construction')
        self.assertEqual(len(rows),4);self.assertEqual(rows[0]['C_B'],0)
        self.assertEqual(rows[0]['active_oracle'],.25);self.assertEqual(rows[0]['nominal_full_oracle_top_2r'],.5)
        before={'x':(layer.weight_a.detach().clone(),layer.weight_b.detach().clone())}
        layer.weight_b.data[0,0]=-.1
        updates=update_rows({'x':layer},before,gs,gs,199,0)
        self.assertTrue(updates[0]['gradient_is_stale']);self.assertAlmostEqual(updates[0]['cos_dynamic'],.5)

    def test_official_formats_and_paired_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            for kind,count in [('gsm8k',1319),('math500',500)]:
                path=Path(directory)/kind
                rows=[dict(question=f'q{i}',answer='work\n#### 3') if kind=='gsm8k' else dict(problem=f'q{i}',answer='3') for i in range(count)]
                path.write_text('\n'.join(json.dumps(r) for r in rows))
                self.assertEqual(len(load_official_samples(path,kind)),count)
            path.write_text('{}\n')
            with self.assertRaises((ValueError,KeyError)):load_official_samples(path,'gsm8k')
        left=[{'sample_sha256':str(i),'correct':False} for i in range(3)]
        right=[dict(r,correct=True) for r in left]
        result=paired(left,right,repetitions=100)
        self.assertEqual(result['right_minus_left'],1.)
        self.assertEqual(result['paired_bootstrap_95'],[1.,1.]);self.assertEqual(wilson(0,10)[0],0.)

    def test_real_lr_scheduler_2000(self):
        p=nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([p],lr=5e-5,weight_decay=.01)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda k:max(0,1-k/2000))
        for k in range(2000):
            self.assertAlmostEqual(opt.param_groups[0]['lr'],5e-5*(1-k/2000))
            p.grad=torch.zeros_like(p);opt.step();sched.step();opt.zero_grad(set_to_none=True)
        self.assertEqual(sched.last_epoch,2000);self.assertEqual(opt.param_groups[0]['lr'],0.)

    def test_84_real_shape_adapter_parameter_count_without_weights(self):
        from test_protocol import ProtocolTests
        inventory=ProtocolTests().inventory()['common']['inventory']['modules']
        pattern={r['module']:dict(rank=r['rank'],A_shape=r['A_shape'],B_shape=r['B_shape'],scaling=r['scaling'],parameters=r['old_parameters']) for r in inventory}
        model=nn.Module();model.model=nn.Module();model.model.layers=nn.ModuleList()
        for _ in range(42):
            layer=nn.Module();layer.self_attn=nn.Module()
            layer.self_attn.q_proj=nn.Linear(2048,2048,bias=False,device='meta')
            layer.self_attn.v_proj=nn.Linear(2048,256,bias=False,device='meta')
            model.model.layers.append(layer)
        layers=attach_pattern(model,pattern)
        self.assertEqual(len(layers),84)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),480000)
        self.assertTrue(all(not m.base_layer.weight.requires_grad for m in layers.values()))

    def test_dual_collector_cpu_once_per_sample(self):
        from types import SimpleNamespace
        from global84_rank_direction import collect_weight_gradients_both,collect_weight_gradients
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.target=nn.Linear(2,2,bias=False);self.calls=0;self.backwards=0
            def forward(self,coeff,labels,use_cache=False):
                self.calls+=1;loss=(self.target.weight*coeff).sum()
                loss.register_hook(self.count);return SimpleNamespace(loss=loss)
            def count(self,g):self.backwards+=1;return g
        model=Toy();before=model.target.weight.detach().clone()
        model.target.weight.requires_grad_(False)
        a=torch.tensor([[1.,2.],[3.,4.]]);b=torch.tensor([[4.,3.],[2.,1.]])
        batches=[dict(coeff=g,labels=torch.tensor([[-100]+[1]*n])) for g,n in [(a,100),(b,300)]]
        fingerprint=rng_fingerprint(rng_state())
        result,profile=collect_weight_gradients_both(model,{'target':model.target},batches)
        self.assertTrue(torch.equal(result['question']['target'],(a+b)/2))
        self.assertTrue(torch.equal(result['token']['target'],a*.25+b*.75))
        self.assertEqual((model.calls,model.backwards),(2,2))
        self.assertEqual(fingerprint,rng_fingerprint(rng_state()))
        self.assertTrue(torch.equal(before,model.target.weight));self.assertFalse(model.target.weight.requires_grad)
        legacy,_=collect_weight_gradients(model,{'target':model.target},batches)
        self.assertTrue(torch.equal(legacy['target'],result['token']['target']))

    def test_atomic_failure_no_published_checkpoint(self):
        from altlora_from_zero.state import atomic_save
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            with patch('torch.save',side_effect=RuntimeError('injected write failure')):
                with self.assertRaises(RuntimeError):atomic_save(path,{'x':1})
            self.assertFalse(path.exists());self.assertEqual(list(Path(directory).iterdir()),[])
