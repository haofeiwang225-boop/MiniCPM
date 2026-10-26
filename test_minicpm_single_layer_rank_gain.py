"""CPU-only tests; no real checkpoints, datasets or GPU execution."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

import minicpm_single_layer_rank_gain as exp
from rank_gain_probe_support import load_original_lora_class


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        cls, config = load_original_lora_class(exp.ROOT)
        self.embedding = nn.Embedding(17, 12)
        self.target = cls(config(in_features=12,out_features=12,lora_rank=4,lora_scaler=4,run_lora_in_fp32=True))
        self.target.init_lora_weights()
        with torch.no_grad(): self.target.weight_b.normal_(std=.1)
        self.head = nn.Linear(12,17)

    def forward(self,input_ids,labels,attention_mask=None,use_cache=False):
        logits = self.head(self.target(self.embedding(input_ids)))
        loss = F.cross_entropy(logits[:,:-1].reshape(-1,17), labels[:,1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(logits=logits,loss=loss)


def examples():
    return [dict(input_ids=torch.tensor([1,2,3,4,5]),labels=torch.tensor([-100,-100,3,4,5]),attention_mask=torch.ones(5,dtype=torch.long)),
            dict(input_ids=torch.tensor([2,3,6,7]),labels=torch.tensor([-100,-100,-100,7]),attention_mask=torch.ones(4,dtype=torch.long))]


def metadata(groups):
    return dict(samples=[dict(index=i,sample_sha256=f's-{g}',original_question_sha256=f'q-{g}') for i,g in enumerate(groups)])


class RankGainTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = TinyLM()
        self.data = examples()

    def test_split_reproducible_and_group_disjoint(self):
        m,n = metadata(list(range(20))+[0]), metadata(range(30,40))
        a = exp.isolated_split(m,n,8,4,4,42)
        self.assertEqual(a,exp.isolated_split(m,n,8,4,4,42))
        groups = lambda key: {m['samples'][i]['original_question_sha256'] for i in a[key]}
        self.assertFalse(groups('train') & groups('probe'))
        self.assertEqual(len(groups('train')),8)

    def test_group_leakage_fails_even_different_sample(self):
        m,n=metadata(range(20)),metadata(range(10,30))
        for row in n['samples']: row['sample_sha256']+='different'
        with self.assertRaisesRegex(AssertionError,'overlap'): exp.isolated_split(m,n,4,4,2,42)

    def test_insufficient_groups_fail(self):
        with self.assertRaisesRegex(AssertionError,'Insufficient'):
            exp.isolated_split(metadata([1,1,1]),metadata([2]),2,1,1,42)

    def test_probe_mean_and_state_restoration(self):
        model=self.model
        model.train(); model.head.eval()
        p=model.target.weight_a; p.grad=torch.ones_like(p)
        saved_grad=p.grad; flags=[p.requires_grad for p in model.parameters()]
        modes=[m.training for m in model.modules()]; values=copy.deepcopy(model.state_dict()); rng=torch.get_rng_state().clone()
        batches=[exp.batch(x,'cpu') for x in self.data]
        g, losses=exp.averaged_gradient(model,model.target,batches)
        singles=[exp.averaged_gradient(model,model.target,[x])[0] for x in batches]
        torch.testing.assert_close(g,(singles[0]+singles[1])/2)
        self.assertEqual(len(losses),2)
        self.assertIs(p.grad,saved_grad)
        self.assertEqual(flags,[p.requires_grad for p in model.parameters()])
        self.assertEqual(modes,[m.training for m in model.modules()])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in values.items(): self.assertTrue(torch.equal(value,model.state_dict()[key]))

    def test_probe_exception_restores_state(self):
        model=self.model; model.train()
        modes=[m.training for m in model.modules()]; flags=[p.requires_grad for p in model.parameters()]
        rng=torch.get_rng_state().clone()
        with patch.object(model,'forward',side_effect=RuntimeError('test')):
            with self.assertRaisesRegex(RuntimeError,'test'):
                exp.averaged_gradient(model,model.target,[exp.batch(self.data[0],'cpu')])
        self.assertEqual(modes,[m.training for m in model.modules()])
        self.assertEqual(flags,[p.requires_grad for p in model.parameters()])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_directions_unit_orthogonal_residual_optimal_reproducible(self):
        a,b=self.model.target.weight_a,self.model.target.weight_b
        g=torch.randn(12,12)
        random, residual, stats=exp.new_directions(a,b,g,42)
        again=exp.new_directions(a,b,g,42)
        self.assertTrue(torch.equal(random,again[0]))
        qa,qb=exp.orth(a.T),exp.orth(b)
        r=exp.residual_gradient(g,qa,qb)
        self.assertAlmostEqual(float(random.norm()),1,places=6)
        self.assertAlmostEqual(float(residual.norm()),1,places=6)
        self.assertLess(float((qa.T@random.T).norm()),2e-5)
        self.assertLess(float((qa.T@residual.T).norm()),2e-5)
        self.assertAlmostEqual(float((r@residual.T).norm()),stats['residual_top_singular_value'],places=5)

    def test_zero_residual_rejected(self):
        with self.assertRaisesRegex(AssertionError,'residual'):
            exp.new_directions(self.model.target.weight_a,self.model.target.weight_b,torch.zeros(12,12),42)

    def test_rank_deficient_factors_rejected(self):
        with self.assertRaisesRegex(AssertionError,'Rank-deficient'):
            exp.new_directions(self.model.target.weight_a,torch.zeros(12,4),torch.randn(12,12),42)

    def test_token_weighted_nll_not_sample_mean(self):
        result=exp.evaluate(self.model,self.data,'cpu')
        self.assertEqual(result['effective_tokens'],4)
        self.assertEqual([r['effective_tokens'] for r in result['samples']],[3,1])
        expected=(result['samples'][0]['nll']*3+result['samples'][1]['nll'])/4
        self.assertEqual(result['token_weighted_nll'],expected)
        self.assertNotAlmostEqual(expected,sum(r['nll'] for r in result['samples'])/2,places=5)

    def test_insertion_preserves_full_logits_and_factor_identity(self):
        target=self.model.target; aid,bid=id(target.weight_a),id(target.weight_b)
        snapshots=copy.deepcopy(self.model.state_dict())
        before=exp.evaluate(self.model,self.data,'cpu')
        a,b,_=exp.new_directions(target.weight_a,target.weight_b,torch.randn(12,12),42)
        for direction in [a,b]:
            self.model.target=exp.ExpandedLoRA(target,direction,torch.zeros(12,1))
            exp.evaluate(self.model,self.data,'cpu',before['samples'])
            self.assertEqual(id(target.weight_a),aid); self.assertEqual(id(target.weight_b),bid)
            self.assertEqual(target.lora_scaler,1.)
            self.assertEqual(target.lora_rank,4)
            self.model.target=target
            for k,v in snapshots.items(): self.assertTrue(torch.equal(v,self.model.state_dict()[k]))

    def test_tiny_three_arm_training_frozen_weights_and_new_rank_updates(self):
        original=copy.deepcopy(self.model.state_dict())
        g,_=exp.averaged_gradient(self.model,self.model.target,[exp.batch(x,'cpu') for x in self.data])
        ra,ca,_=exp.new_directions(self.model.target.weight_a,self.model.target.weight_b,g,42)
        target=self.model.target; baseline=exp.evaluate(self.model,self.data,'cpu')
        for direction in [None,ra,ca]:
            self.model.target=target; self.model.load_state_dict(original)
            if direction is not None: self.model.target=exp.ExpandedLoRA(target,direction,torch.zeros(12,1))
            exp.evaluate(self.model,self.data,'cpu',baseline['samples'])
            with tempfile.TemporaryDirectory() as folder:
                result=exp.optimize(self.model,self.model.target,self.data,[0,1,0],'cpu',.01,0.,Path(folder)/'train.jsonl')
            for name in ['weight']:
                self.assertTrue(torch.equal(getattr(target,name),original['target.'+name]))
            self.assertTrue(torch.equal(self.model.head.weight,original['head.weight']))
            self.assertTrue(torch.equal(self.model.embedding.weight,original['embedding.weight']))
            self.assertEqual(result['trainable_parameters'],96 if direction is None else 120)
            if direction is not None:
                self.assertTrue(result['parameter_changes']['new_A']); self.assertTrue(result['parameter_changes']['new_B'])
            self.assertTrue(torch.isfinite(torch.tensor(exp.evaluate(self.model,self.data,'cpu')['token_weighted_nll'])))

    def test_validation_exception_restores_mode_and_rng(self):
        self.model.train(); rng=torch.get_rng_state().clone()
        with patch.object(self.model,'forward',side_effect=RuntimeError('validation failure')):
            with self.assertRaises(RuntimeError): exp.evaluate(self.model,self.data,'cpu')
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_fail_report_and_refuse_output_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'new'
            with patch('sys.argv',['test','--output-dir',str(output)]), patch.object(exp,'run',side_effect=RuntimeError('真实失败')):
                with self.assertRaises(RuntimeError): exp.main()
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['status'],'FAIL'); self.assertIn('真实失败',report['error'])
            original=(output/'report.json').read_bytes()
            with patch('sys.argv',['test','--output-dir',str(output)]):
                with self.assertRaises(FileExistsError): exp.main()
            self.assertEqual(original,(output/'report.json').read_bytes())

    def test_real_dataset_masking_dynamic_and_static(self):
        class Tokenizer:
            pad_token_id = 0
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                text = ''.join(m['role']+':'+m['content']+';' for m in messages)
                return text + ('assistant:' if add_generation_prompt else '')
            def __call__(self, text, **kwargs):
                return dict(input_ids=[ord(c) for c in text], offset_mapping=[(i,i+1) for i in range(len(text))])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.json'
            path.write_text(json.dumps([dict(messages=[dict(role='user',content='2+3?'),dict(role='assistant',content='5')])]))
            dynamic=exp.ChatTrainingDataset(path,Tokenizer(),64,dynamic_length=True)[0]
            static=exp.ChatTrainingDataset(path,Tokenizer(),64,dynamic_length=False)[0]
            n=len(dynamic['labels'])
            self.assertTrue(torch.equal(dynamic['labels'],static['labels'][:n]))
            self.assertTrue((static['labels'][n:]==-100).all())
            self.assertEqual(dynamic['labels'][dynamic['labels']!=-100].tolist(),[ord('5'),ord(';')])
            with self.assertRaisesRegex(ValueError,'truncated'):
                exp.ChatTrainingDataset(path,Tokenizer(),10,dynamic_length=True)

    def test_loader_rejects_missing_weights(self):
        from rank_gain_probe_support import load_probe_model
        with patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=(self.model,dict(missing_keys=['real.weight']))):
            with self.assertRaisesRegex(AssertionError,'Incomplete real checkpoint'):
                load_probe_model('/not-used',torch.float32)

    def test_loader_snapshot_verbatim_when_local_v3_available(self):
        source=exp.ROOT/'minicpm_rank1_safe_probe_v3.py'
        if not source.exists(): self.skipTest('Optional original local v3 not tracked')
        def functions(path):
            tree=ast.parse(path.read_text())
            return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
        old,new=functions(source),functions(exp.ROOT/'rank_gain_probe_support.py')
        for name,tree in new.items(): self.assertEqual(tree,old[name])


if __name__ == '__main__': unittest.main()
