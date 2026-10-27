"""CPU tests for global84 directions, exact budgets, training and statistics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from types import SimpleNamespace

import global84_rank_math as mathops
import global84_experiment as workflow
import global84_statistics as stats
from test_minicpm_single_layer_rank_gain import TinyLM, examples


def inventory():
    return [dict(module=f'model.layers.{i}.self_attn.{kind}',cost_per_added_rank=cost)
            for i in range(42) for kind,cost in [('q_proj',4096),('v_proj',2304)]]


def options(rows):
    return {r['module']:[dict(k=k,cost=k*r['cost_per_added_rank'],value=float(k),marginal_value=0. if k==0 else 1.) for k in range(5)] for r in rows}


class DirectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.a=torch.randn(2,12); self.b=torch.randn(7,2)
        self.gc=torch.randn(7,12,dtype=torch.float64); self.gs=torch.randn(7,12,dtype=torch.float64)

    def test_matrix_free_matches_dense_toy_eigenvalues(self):
        q=mathops.basis(self.a.T)
        H=(self.gc.T@self.gs+self.gs.T@self.gc)/2
        # Dense reference is deliberately tiny and test-only.
        complement=torch.linalg.svd(q.T,full_matrices=True).Vh[q.shape[1]:].T
        actual=complement.T@H@complement
        expected=torch.linalg.eigvalsh(actual).flip(0)[:4]
        V,info=mathops.symmetric_ritz(lambda x:mathops.project(H@mathops.project(x,q),q),q,4,11,max_basis=12,tolerance=1e-8)
        np.testing.assert_allclose(info['eigenvalues'],expected.numpy(),atol=1e-10)
        self.assertFalse(info['dense_gram_constructed'])
        torch.testing.assert_close(V.T@V,torch.eye(4,dtype=torch.float64))

    def test_largest_algebraic_not_largest_magnitude(self):
        H=torch.diag(torch.tensor([-100.,-3.,1.,2.,4.,5.],dtype=torch.float64))
        V,info=mathops.symmetric_ritz(lambda x:H@x,torch.empty(6,0,dtype=torch.float64),2,1,max_basis=6,tolerance=1e-8)
        np.testing.assert_allclose(info['eigenvalues'],[5.,4.],atol=1e-10)

    def test_repeated_zero_eigenvalues_and_no_false_gain(self):
        V,info=mathops.symmetric_ritz(lambda x:torch.zeros_like(x),torch.empty(8,0,dtype=torch.float64),4,1,max_basis=8)
        self.assertEqual(info['eigenvalues'],[0.]*4)
        torch.testing.assert_close(V.T@V,torch.eye(4,dtype=torch.float64))

    def test_nonconvergence_fails_loudly(self):
        H=torch.diag(torch.arange(1,21,dtype=torch.float64))
        with self.assertRaisesRegex(ValueError,'not converged'):
            mathops.symmetric_ritz(lambda x:H@x,torch.empty(20,0,dtype=torch.float64),2,3,max_basis=3,tolerance=1e-12)

    def test_all_four_candidates_orthogonal_and_reproducible(self):
        rng=torch.get_rng_state().clone()
        frames,report=mathops.candidate_frames(self.a,self.b,self.gc,self.gs,4,42,max_basis=12,tolerance=1e-8)
        other,_=mathops.candidate_frames(self.a,self.b,self.gc,self.gs,4,42,max_basis=12,tolerance=1e-8)
        self.assertEqual(set(frames),set(mathops.FAMILIES)); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for kind,V in frames.items():
            torch.testing.assert_close(V,other[kind]); torch.testing.assert_close(V.T@V,torch.eye(4,dtype=torch.float64))
            self.assertLess(float((self.a.double()@V).abs().max()),2e-5)
        self.assertEqual(set(report),set(mathops.FAMILIES)-{'random'})

    def test_cross_split_rayleigh_equals_U(self):
        frames,r=mathops.candidate_frames(self.a,self.b,self.gc,self.gs,4,42,max_basis=12,tolerance=1e-8)
        V=frames['cross_alignment']; U=((self.gc@V)*(self.gs@V)).sum(dim=0)
        torch.testing.assert_close(U,torch.tensor(r['cross_alignment']['eigenvalues'],dtype=torch.float64))

    def test_prefix_uncertainty_keeps_direction_covariance(self):
        V=torch.eye(2,dtype=torch.float64); ref=V
        blocks=[torch.diag(torch.tensor([x,-x],dtype=torch.float64)) for x in (1.,2.,3.,4.)]
        opts,_=mathops.proxy_prefixes(ref,blocks,[1,1,1,1],V,10)
        self.assertGreater(opts[1]['standard_error'],0.)
        self.assertEqual(opts[2]['standard_error'],0.)
        self.assertEqual(opts[2]['mean'],0.)

    def test_audit_is_reporting_only(self):
        frames,_=mathops.candidate_frames(self.a,self.b,self.gc,self.gs,4,42,max_basis=12)
        before,_=mathops.proxy_prefixes(self.gc,[self.gs]*4,[1]*4,frames['cross_alignment'],4096)
        counts={k:[1]*4 for k in workflow.ROLES}
        one=workflow.audit_scores(self.gc,[self.gc]*4,[self.gs]*4,[self.gc]*4,counts,frames['cross_alignment'])
        two=workflow.audit_scores(self.gc,[self.gc]*4,[self.gs]*4,[-100*self.gc]*4,counts,frames['cross_alignment'])
        after,_=mathops.proxy_prefixes(self.gc,[self.gs]*4,[1]*4,frames['cross_alignment'],4096)
        self.assertEqual(before,after); self.assertNotEqual(one['audit_U'],two['audit_U']); self.assertFalse(one['audit_used_for_selection'])


class BudgetTests(unittest.TestCase):
    def test_exact_budget_multi_rank_single_layer(self):
        rows=inventory(); opts=options(rows)
        preferred=[rows[0]['module'],rows[1]['module']]
        for n in preferred:
            for choice in opts[n]: choice.update(value=float(choice['k']*10),marginal_value=10. if choice['k'] else 0.)
        alloc,report=mathops.exact_stratified_allocation(opts,rows)
        self.assertEqual(alloc[preferred[0]],4); self.assertEqual(alloc[preferred[1]],4)
        self.assertEqual(sum(alloc[r['module']]*r['cost_per_added_rank'] for r in rows),25600)
        self.assertEqual(report['added_parameters'],25600)

    def test_integer_dp_considers_marginal_prefixes(self):
        rows=inventory(); opts=options(rows)
        for row in rows:
            for choice in opts[row['module']]: choice['value']=-100.*choice['k']
        for row in rows[:4]:
            for choice in opts[row['module']]: choice['value']=[0.,10.,20.,-5.,-6.][choice['k']]
        alloc,_=mathops.exact_stratified_allocation(opts,rows)
        self.assertEqual([alloc[r['module']] for r in rows[:4]],[2,2,2,2])

    def test_negative_proxies_do_not_change_exact_budget(self):
        rows=inventory(); opts=options(rows)
        for choices in opts.values():
            for c in choices: c.update(value=-float(c['k']),marginal_value=-1. if c['k'] else 0.)
        alloc,r=mathops.exact_stratified_allocation(opts,rows)
        self.assertEqual(sum(alloc.values()),8)
        self.assertEqual(r['strata']['q_proj']['negative_marginals_forced'],4)

    def test_incomplete_modules_or_wrong_real_cost_rejected(self):
        rows=inventory(); opts=options(rows)
        with self.assertRaisesRegex(ValueError,'All modules'): mathops.exact_stratified_allocation({**opts,'extra':[]},rows)
        rows[0]['cost_per_added_rank']=4095
        with self.assertRaisesRegex(ValueError,'costs'): mathops.exact_stratified_allocation(opts,rows)

    def test_random_allocation_reproducible_with_replacement(self):
        rows=inventory(); a=mathops.random_allocation(rows,42)
        self.assertEqual(a,mathops.random_allocation(rows,42)); self.assertEqual(sum(a.values()),8)
        with patch('random.Random.choice',side_effect=lambda seq:seq[0]):
            b=mathops.random_allocation(rows,42)
        self.assertEqual(b[rows[0]['module']],4); self.assertEqual(b[rows[1]['module']],4)


class WorkflowTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(42)

    def test_zero_output_multi_rank_and_all_old_factors_trainable(self):
        model=TinyLM(); layers={'target':model.target}; saved=copy.deepcopy(model.state_dict())
        q=mathops.basis(model.target.weight_a.T)
        V=mathops.random_frame(q,3,42).float()
        frames={'target':{'random':V}}
        batch=workflow.shared.batch(examples()[0],'cpu')
        model.eval(); before=model(**batch).logits.detach().clone()
        expanded,added=workflow.attach_branches(model,layers,{'target':3},frames,'random')
        self.assertEqual(added,3*(12+12)); self.assertTrue(torch.equal(before,model(**batch).logits))
        result=workflow.train_adapters(model,layers,expanded,examples(),[0,1],'cpu')
        self.assertEqual(result['trainable_parameters'],4*(12+12)+3*(12+12))
        self.assertTrue(torch.equal(model.target.base_layer.weight,saved['target.weight']))
        self.assertTrue(torch.equal(model.embedding.weight,saved['embedding.weight']))
        self.assertTrue(result['fresh_AdamW']); self.assertFalse(result['historical_moments_restored'])
        self.assertEqual(len(result['steps']),2)
        snap={'target':{k:saved['target.'+k] for k in ('weight_a','weight_b')}}
        workflow.restore_layers(model,layers,snap)
        self.assertIs(model.target,layers['target'])
        for k,v in saved.items(): self.assertTrue(torch.equal(v,model.state_dict()[k]))

    def test_all_original_adapter_modules_train_not_just_expanded(self):
        # Two ordinary dense LoRA modules chained in a tiny LM.
        model=TinyLM()
        from rank_gain_probe_support import load_original_lora_class
        cls,cfg=load_original_lora_class(workflow.ROOT)
        second=cls(cfg(in_features=12,out_features=12,lora_rank=2,lora_scaler=4,run_lora_in_fp32=True)); second.init_lora_weights()
        with torch.no_grad(): second.weight_b.normal_(std=.1)
        original=model.target
        model.target=nn.Sequential(original,second)
        layers={'target.0':original,'target.1':second}; state=copy.deepcopy(model.state_dict())
        result=workflow.train_adapters(model,layers,{},examples(),[0,1],'cpu')
        self.assertEqual(result['trainable_parameters'],(4+2)*24)
        for i,l in enumerate((original,second)):
            self.assertTrue(l.weight_a.requires_grad); self.assertTrue(l.weight_b.requires_grad); self.assertFalse(l.weight.requires_grad)
            self.assertTrue(torch.equal(l.weight,state[f'target.{i}.weight']))

    def test_dry_run_never_enters_gpu_path(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'new'; plan=dict(fingerprint='x',common=dict(input_sha256={},inventory={}))
            def prepare(args): workflow.write(args.output_dir/'plan.json',plan); return plan,{}
            with patch('sys.argv',['test','dry-run','--output-dir',str(out)]),patch.object(workflow,'prepare',side_effect=prepare),patch.object(workflow,'gpu_admission',side_effect=AssertionError('No GPU')):
                workflow.main()
            self.assertEqual(json.loads((out/'report.json').read_text())['status'],'PASS')
            with patch('sys.argv',['test','dry-run','--output-dir',str(out)]):
                with self.assertRaises(FileExistsError): workflow.main()

    def test_failure_report_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'new'
            with patch('sys.argv',['test','dry-run','--output-dir',str(out)]),patch.object(workflow,'prepare',side_effect=ValueError('data mismatch')):
                with self.assertRaises(ValueError): workflow.main()
            self.assertEqual(json.loads((out/'report.json').read_text())['status'],'FAIL')


class StatisticalTests(unittest.TestCase):
    def data(self):
        n=np.array([1.,3.,2.,6.]); x=np.tile(np.array([1.,2.,3.,4.])*n,(1,3,5,1)); x[:,:,4,:]-=.001*n
        meta=[dict(index=i,original_question_sha256=f'q{i}',family='GSM' if i<2 else 'MATH') for i in range(4)]
        return x,n,meta

    def test_paired_token_weighting_and_no_fake_independent_repeats(self):
        x,n,meta=self.data(); r=stats.analyze(x,n,meta,repeats=80)
        np.testing.assert_allclose(r['overall']['comparisons']['E-D']['ci95'],[-.001,-.001],atol=1e-12)
        repeated=stats.analyze(np.repeat(x,2,axis=-1),np.repeat(n,2),[m for m in meta for _ in range(2)],repeats=80)
        self.assertEqual(r['overall']['comparisons']['E-D']['ci95'],repeated['overall']['comparisons']['E-D']['ci95'])
        self.assertTrue(r['statistical_protocol']['engineering_PASS_is_not_efficacy'])

    def test_no_scientific_success_for_identical_arms(self):
        x,n,meta=self.data(); x[:,:,4,:]=x[:,:,3,:]
        r=stats.analyze(x,n,meta,repeats=80)
        self.assertEqual(r['scientific_interpretation'],'主要比较的收益证据不足')

    def test_complete_grid_loading_and_tampered_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); paths=[]
            meta=[dict(index=i,sample_sha256=f's{i}',original_question_sha256=f'g{i}',family='GSM' if i<200 else 'MATH') for i in range(400)]
            common=dict(protocol=workflow.PROTOCOL,indices=dict(train=list(range(400))),metadata=dict(dev=meta),tokens=dict(dev=[2]*400))
            fingerprint=workflow.digest(common)
            for seed in stats.SEEDS:
                out=root/str(seed); out.mkdir(); paths.append(out)
                allocation=dict(allocations={a:dict(layer=0 if a=='A' else 4) for a in stats.ARMS})
                workflow.write(out/'plan.json',dict(common=common,fingerprint=fingerprint))
                workflow.write(out/'allocation.json',allocation)
                workflow.write(out/'report.json',dict(mode='train',status='PASS',input_files_unchanged=True,plan_fingerprint=fingerprint,
                    seed=seed,source_order=list(range(400)),completed_arms=stats.ARMS))
                for arm in stats.ARMS:
                    d=out/arm; d.mkdir()
                    rows=[dict(index=i,source=meta[i],effective_tokens=2,nll=2.,nll_sum=4.) for i in range(400)]
                    result=dict(status='PASS',arm=arm,seed=seed,plan_fingerprint=fingerprint,allocation=allocation['allocations'][arm],
                        added_parameters=0 if arm=='A' else 25600,invariants=dict(base_weights_verified=381),
                        before=dict(samples=rows),after=dict(samples=rows,token_weighted_nll=2.),
                        training=dict(completed_steps=400,trainable_parameters=480000 if arm=='A' else 505600,
                                      steps=[dict(step=i+1,pool_index=i) for i in range(400)]))
                    workflow.write(d/'result.json',result); (d/'train.jsonl').write_text('test'); (d/'adapter.pt').write_bytes(b'test')
                    workflow.write(d/'artifacts.json',{n:stats.sha(d/n) for n in ('result.json','train.jsonl','adapter.pt')})
            sums,tokens,actual,_=stats.load(paths)
            self.assertEqual(sums.shape,(1,3,5,400)); self.assertEqual(int(tokens.sum()),800)
            bad=paths[-1]/'E'/'result.json'; r=json.loads(bad.read_text()); r['after']['samples'][0]['source']['sample_sha256']='wrong'
            workflow.write(bad,r)
            hashes=json.loads((bad.parent/'artifacts.json').read_text()); hashes['result.json']=stats.sha(bad); workflow.write(bad.parent/'artifacts.json',hashes)
            with self.assertRaisesRegex(ValueError,'source/hash'): stats.load(paths)

    def test_missing_grid_rejected(self):
        with self.assertRaisesRegex(ValueError,'complete 3 seeds'): stats.load([])


if __name__=='__main__': unittest.main()
