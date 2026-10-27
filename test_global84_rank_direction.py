"""CPU-only consistency/restoration tests; no real model forward or GPU probe."""
import copy
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from types import SimpleNamespace

from global84_rank_direction import collect_weight_gradients
from minicpm_rank_gain_decisive import weighted_probe


class HeterogeneousLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(13,8)
        self.first=nn.Linear(8,6)
        self.second=nn.Linear(6,4)
        self.head=nn.Linear(4,13)
        self.dropout=nn.Dropout(.4)
        self.calls=0
        self.unused=nn.Linear(3,2)

    def forward(self,input_ids,labels,attention_mask=None,use_cache=False):
        self.calls+=1
        # These otherwise irrelevant draws verify ALL RNG state restoration.
        random.random(); np.random.random(); torch.rand(1)
        x=self.dropout(self.embedding(input_ids))
        x=self.second(torch.tanh(self.first(x)))
        logits=self.head(x)
        loss=F.cross_entropy(logits[:,:-1].reshape(-1,13),labels[:,1:].reshape(-1),ignore_index=-100)
        return SimpleNamespace(loss=loss,logits=logits)


def batches():
    return [dict(input_ids=torch.tensor([[1,2,3,4]]),labels=torch.tensor([[-100,2,3,4]])),
            dict(input_ids=torch.tensor([[4,3,2]]),labels=torch.tensor([[-100,-100,2]]))]


class SimultaneousGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); random.seed(42); np.random.seed(42)
        self.model=HeterogeneousLM()
        self.targets=dict(first=self.model.first,second=self.model.second,head=self.model.head)

    def test_one_forward_backward_per_batch_not_layer(self):
        gradients,report=collect_weight_gradients(self.model,self.targets,batches())
        self.assertEqual(self.model.calls,2)
        self.assertEqual(report['forward_calls'],2); self.assertEqual(report['backward_calls'],2)
        self.assertEqual(report['gradient_calls'],dict(first=2,second=2,head=2))
        self.assertEqual(gradients['first'].shape,(6,8)); self.assertEqual(gradients['second'].shape,(4,6))
        self.assertEqual(gradients['head'].shape,(13,4))
        for g in gradients.values(): self.assertEqual(g.dtype,torch.float32); self.assertEqual(g.device.type,'cpu')

    def test_matches_existing_single_layer_probe(self):
        gradients,_=collect_weight_gradients(self.model,self.targets,batches())
        examples=[{k:v.squeeze(0) for k,v in b.items()} for b in batches()]
        for name,layer in self.targets.items():
            expected,_=weighted_probe(self.model,layer,examples,'cpu')
            torch.testing.assert_close(gradients[name],expected,rtol=1e-6,atol=1e-7)

    def test_exact_token_weighted_aggregation(self):
        combined,report=collect_weight_gradients(self.model,self.targets,batches())
        g1,_=collect_weight_gradients(self.model,self.targets,batches()[:1])
        g2,_=collect_weight_gradients(self.model,self.targets,batches()[1:])
        for name in combined: torch.testing.assert_close(combined[name],(3*g1[name]+g2[name])/4)
        self.assertEqual(report['effective_tokens'],4)
        self.assertEqual([r['effective_tokens'] for r in report['batch_losses']],[3,1])

    def test_state_rng_optimizer_and_values_restored(self):
        m=self.model; m.train(); m.head.eval()
        params=list(m.parameters()); optimizer=torch.optim.AdamW(params,lr=.01)
        m(**batches()[0]).loss.backward(); optimizer.step()
        # Preserve nonempty parameter grads and optimizer tensors by identity/version/value.
        states={p:(s,{k:(id(v),v._version,v.clone()) for k,v in s.items() if torch.is_tensor(v)}) for p,s in optimizer.state.items()}
        flags=[p.requires_grad for p in params]; grads=[p.grad for p in params]
        values=copy.deepcopy(m.state_dict()); modes=[x.training for x in m.modules()]
        torch_rng=torch.get_rng_state().clone(); py_rng=random.getstate(); np_rng=np.random.get_state()
        with patch.object(optimizer,'step',side_effect=AssertionError('No optimizer step')):
            collect_weight_gradients(m,self.targets,batches())
        self.assertEqual(flags,[p.requires_grad for p in params]); self.assertTrue(all(p.grad is g for p,g in zip(params,grads)))
        self.assertEqual(modes,[x.training for x in m.modules()]); self.assertTrue(torch.equal(torch_rng,torch.get_rng_state()))
        self.assertEqual(py_rng,random.getstate()); after=np.random.get_state()
        self.assertEqual(np_rng[0],after[0]); np.testing.assert_array_equal(np_rng[1],after[1]); self.assertEqual(np_rng[2:],after[2:])
        for name,v in values.items(): self.assertTrue(torch.equal(v,m.state_dict()[name]))
        for p,(obj,tensors) in states.items():
            self.assertIs(optimizer.state[p],obj)
            for k,(identity,version,value) in tensors.items():
                v=optimizer.state[p][k]; self.assertEqual(id(v),identity); self.assertEqual(v._version,version); self.assertTrue(torch.equal(value,v))

    def test_unused_target_is_error_not_silent_omission(self):
        self.targets['unused']=self.model.unused
        flags=[p.requires_grad for p in self.model.parameters()]
        with self.assertRaisesRegex(ValueError,'Missing target gradients'):
            collect_weight_gradients(self.model,self.targets,batches())
        self.assertEqual(flags,[p.requires_grad for p in self.model.parameters()])
        self.assertTrue(all(not p._post_accumulate_grad_hooks for p in self.model.parameters()))

    def test_exception_restores_state_hooks_and_rng(self):
        flags=[p.requires_grad for p in self.model.parameters()]; self.model.train()
        rng=torch.get_rng_state().clone(); py=random.getstate()
        def fail(**kwargs):
            torch.rand(3); random.random(); raise RuntimeError('forward error')
        with patch.object(self.model,'forward',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'forward error'):
                collect_weight_gradients(self.model,self.targets,batches())
        self.assertEqual(flags,[p.requires_grad for p in self.model.parameters()]); self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(rng,torch.get_rng_state())); self.assertEqual(py,random.getstate())
        self.assertTrue(all(not p._post_accumulate_grad_hooks for p in self.model.parameters()))

    def test_nonfinite_gradient_fails(self):
        hook=self.model.first.weight.register_hook(lambda g:torch.full_like(g,float('nan')))
        try:
            with self.assertRaisesRegex(ValueError,'Nonfinite gradient'):
                collect_weight_gradients(self.model,self.targets,batches())
        finally: hook.remove()

    def test_zero_gradient_included_and_counted(self):
        with torch.no_grad(): self.model.head.weight.zero_()
        gradients,report=collect_weight_gradients(self.model,self.targets,batches())
        self.assertTrue(torch.equal(gradients['first'],torch.zeros_like(gradients['first'])))
        self.assertEqual(report['gradient_calls']['first'],2)

    def test_real_lora_factors_frozen_during_probe(self):
        from test_minicpm_single_layer_rank_gain import TinyLM, examples
        model=TinyLM(); target=model.target; observed=[]
        def check_flags(module,inputs):
            self.assertTrue(target.weight.requires_grad)
            self.assertFalse(target.weight_a.requires_grad); self.assertFalse(target.weight_b.requires_grad)
            self.assertFalse(model.embedding.weight.requires_grad)
            observed.append(True)
        hook=model.register_forward_pre_hook(check_flags)
        items=[{k:v.unsqueeze(0) for k,v in e.items()} for e in examples()]
        try: collect_weight_gradients(model,{'target':target},items)
        finally: hook.remove()
        self.assertEqual(len(observed),2)
        self.assertTrue(target.weight_a.requires_grad); self.assertTrue(target.weight_b.requires_grad)

    def test_empty_labels_and_alias_rejected(self):
        data=batches(); data[0]['labels'].fill_(-100)
        with self.assertRaisesRegex(ValueError,'Empty shifted'): collect_weight_gradients(self.model,self.targets,data)
        with self.assertRaisesRegex(ValueError,'Aliased'):
            collect_weight_gradients(self.model,dict(a=self.model.first,b=self.model.first),batches())
        self.assertEqual(self.model.calls,0)


if __name__=='__main__': unittest.main()
