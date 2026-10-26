"""CPU-only unit tests. No real model forward or GPU is executed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

import minicpm_rank_gain_decisive as run
import minicpm_rank_gain_statistics as stats
from test_minicpm_single_layer_rank_gain import TinyLM, examples, metadata


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17); self.model=TinyLM(); self.examples=examples()

    def test_fixed_split_and_dev_order_all_seeds(self):
        train=metadata(range(1000)); dev=metadata(range(1000,1400))
        for i,r in enumerate(dev['samples']): r['family']='GSM' if i<200 else 'MATH'
        split=run.split_data(train,dev)
        self.assertEqual([len(split[k]) for k in ('train','construction','audit','dev')],[200,32,32,400])
        self.assertEqual(split['dev'],list(range(400)))
        all_train=split['train']+split['construction']+split['audit']
        self.assertEqual(len(all_train),len(set(all_train)))
        self.assertEqual(split,run.split_data(train,dev))

    def test_group_overlap_fails(self):
        train=metadata(range(1000)); dev=metadata(range(600,1000))
        with self.assertRaisesRegex(AssertionError,'overlap'): run.split_data(train,dev)

    def test_seed_domains_independent_and_reproducible(self):
        seeds=[run.seed_stream(42,k) for k in ('training_order','random_direction','probe_order')]
        self.assertEqual(len(set(seeds)),3)
        self.assertEqual(seeds[0],run.seed_stream(42,'training_order'))
        self.assertNotEqual(seeds[0],run.seed_stream(43,'training_order'))

    def test_probe_token_weighted_not_example_mean(self):
        m=self.model; layer=m.target
        g,rows=run.weighted_probe(m,layer,self.examples,'cpu')
        g1,_=run.weighted_probe(m,layer,self.examples[:1],'cpu')
        g2,_=run.weighted_probe(m,layer,self.examples[1:],'cpu')
        torch.testing.assert_close(g,(3*g1+g2)/4)
        self.assertGreater(float((g-(g1+g2)/2).norm()),1e-5)
        self.assertEqual([x['effective_tokens'] for x in rows],[3,1])

    def test_probe_restore_modes_flags_grads_rng_and_weights(self):
        m=self.model; m.train(); m.head.eval(); m.target.weight_a.grad=torch.ones_like(m.target.weight_a)
        params=list(m.parameters()); grads=[p.grad for p in params]; flags=[p.requires_grad for p in params]
        modes=[x.training for x in m.modules()]; state=copy.deepcopy(m.state_dict()); rng=torch.get_rng_state().clone()
        run.weighted_probe(m,m.target,self.examples,'cpu')
        self.assertEqual(flags,[p.requires_grad for p in params]); self.assertEqual(modes,[x.training for x in m.modules()])
        self.assertTrue(all(p.grad is g for p,g in zip(params,grads)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in state.items(): self.assertTrue(torch.equal(v,m.state_dict()[k]))

    def test_probe_exception_restore(self):
        m=self.model; flags=[p.requires_grad for p in m.parameters()]; m.train()
        with patch.object(m,'forward',side_effect=RuntimeError('probe broke')):
            with self.assertRaisesRegex(RuntimeError,'probe broke'): run.weighted_probe(m,m.target,self.examples,'cpu')
        self.assertEqual(flags,[p.requires_grad for p in m.parameters()]); self.assertTrue(m.training)

    def test_audit_score_uses_independent_gradient_and_no_selection(self):
        a,b=self.model.target.weight_a,self.model.target.weight_b
        g1,g2,ga=torch.randn(12,12),torch.randn(12,12),torch.randn(12,12)
        ra,ca,report=run.direction_audit(a,b,[(g1,3),(g2,1)],ga,123)
        ra2,ca2,report2=run.direction_audit(a,b,[(g1,3),(g2,1)],2*ga,123)
        self.assertTrue(torch.equal(ra,ra2)); self.assertTrue(torch.equal(ca,ca2))
        self.assertAlmostEqual(report2['residual_score'],4*report['residual_score'],places=10)
        expected=float((ga.double()@ca.double().T).square().sum())
        self.assertEqual(report['residual_score'],expected)
        self.assertFalse(report['audit_used_for_candidate_selection'])
        self.assertTrue(0<=report['construction_half_abs_cosine']<=1)
        _,ref,_=run.old.new_directions(a,b,(3*g1+g2)/4,123)
        torch.testing.assert_close(ref,ca)

    def test_identical_probe_halves_cosine_one(self):
        g=torch.randn(12,12)
        _,_,r=run.direction_audit(self.model.target.weight_a,self.model.target.weight_b,[(g,1),(g,4)],g,42)
        self.assertAlmostEqual(r['construction_half_abs_cosine'],1.,places=6)

    def test_zero_audit_score_ratio_explicit(self):
        g=torch.randn(12,12)
        _,_,r=run.direction_audit(self.model.target.weight_a,self.model.target.weight_b,[(g,1),(g,4)],torch.zeros_like(g),42)
        self.assertIsNone(r['C_over_B_score']); self.assertEqual(r['ratio_undefined_reason'],'zero_random_score')

    def test_training_weighted_normalization_and_frozen_parameters(self):
        target=self.model.target; state=copy.deepcopy(self.model.state_dict())
        r=run.train_arm(self.model,target,self.examples,[0,1],'cpu',None,[10,11])
        self.assertEqual(r['mean_train_tokens'],2)
        for row in r['steps']: self.assertAlmostEqual(row['backward_objective'],row['nll']*row['effective_tokens']/2,places=5)
        self.assertFalse(r['historical_AdamW_restored'])
        self.assertTrue(torch.equal(target.weight,state['target.weight']))
        self.assertTrue(torch.equal(self.model.head.weight,state['head.weight']))
        self.assertEqual([x['source_index'] for x in r['steps']],[10,11])

    def test_expansion_training_and_real_added_update_norm(self):
        target=self.model.target; g=torch.randn(12,12)
        direction,_,_=run.old.new_directions(target.weight_a,target.weight_b,g,42)
        before=run.old.evaluate(self.model,self.examples,'cpu')
        self.model.target=run.old.ExpandedLoRA(target,direction,torch.zeros(12,1))
        run.old.evaluate(self.model,self.examples,'cpu',before['samples'])
        r=run.train_arm(self.model,self.model.target,self.examples,[0,1],'cpu',None,[10,11])
        expected=float((self.model.target.add_b.double()@self.model.target.add_a.double()).norm())
        self.assertAlmostEqual(r['added_BA_fro_norm'],expected,places=12)
        self.assertTrue(r['parameter_changes']['new_A']['changed'])
        self.assertTrue(r['parameter_changes']['new_B']['changed'])

    def test_training_rejects_wrong_order(self):
        with self.assertRaisesRegex(ValueError,'visit each example'):
            run.train_arm(self.model,self.model.target,self.examples,[0,0],'cpu',None,[10,11])

    def test_resume_rejects_wrong_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); (p/'plan.json').write_text(json.dumps(dict(fingerprint='old',common={})))
            with self.assertRaisesRegex(ValueError,'mismatch'): run.reusable_arms(p,dict(fingerprint='new',common={}))

    def test_resume_skips_only_completed_verified_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); plan=dict(fingerprint='same',common={}); (p/'plan.json').write_text(json.dumps(plan))
            a=p/'A'; a.mkdir()
            result=dict(status='PASS',arm='A',fingerprint='same',training=dict(completed_steps=200),after=dict(num_samples=400),invariants=dict(base_tensors_verified=381,foo=True))
            (a/'result.json').write_text(json.dumps(result)); (a/'train.jsonl').write_text('steps'); (a/'target_factors.pt').write_bytes(b'weights')
            run.old.write_json(a/'artifacts.json',{k:run.old.sha256(a/k) for k in ('result.json','train.jsonl','target_factors.pt')})
            self.assertEqual(set(run.reusable_arms(p,plan)),{'A'})
            (a/'train.jsonl').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'Corrupted'): run.reusable_arms(p,plan)

    def test_resume_ignores_uncommitted_partial_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); plan=dict(fingerprint='same',common={})
            run.old.write_json(p/'plan.json',plan); (p/'A').mkdir()
            run.old.write_json(p/'A'/'result.json',dict(status='PASS'))
            # Interrupted before artifact manifest: not a completed combination.
            self.assertEqual(run.reusable_arms(p,plan),{})

    def test_dry_run_does_not_call_gpu_path(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'new'
            with patch('sys.argv',['test','dry-run','--module',run.old.MODULES[0],'--seed','42','--output-dir',str(out)]), \
                 patch.object(run,'prepare',return_value=(dict(fingerprint='x',common=dict(input_sha256={})),{},{})), \
                 patch.object(run,'gpu_run',side_effect=AssertionError('GPU forbidden')):
                run.main()
            report=json.loads((out/'report.json').read_text()); self.assertEqual(report['status'],'PASS'); self.assertTrue(report['cpu_only'])

    def test_failure_report_and_existing_output_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'new'; argv=['test','dry-run','--module',run.old.MODULES[0],'--seed','42','--output-dir',str(out)]
            with patch('sys.argv',argv),patch.object(run,'prepare',side_effect=RuntimeError('bad data')):
                with self.assertRaises(RuntimeError): run.main()
            report=json.loads((out/'report.json').read_text()); self.assertEqual(report['status'],'FAIL')
            with patch('sys.argv',argv):
                with self.assertRaises(FileExistsError): run.main()


class StatisticsTests(unittest.TestCase):
    def data(self):
        tokens=np.array([1.,3.,2.,6.]); mean=np.array([2.,3.,4.,5.])
        sums=np.tile(mean*tokens,(2,5,3,1))
        sums[:,:,2,:]-=.001*tokens
        meta=[(i,i,str(i),f'g{i}','GSM' if i<2 else 'MATH',int(tokens[i])) for i in range(4)]
        return sums,tokens,meta

    def test_token_weighted_point_estimate(self):
        sums,tokens,meta=self.data(); sums[:,:,2,0]-=.04
        r=stats.analyze(sums,tokens,meta,repeats=80)
        expected=-.001-.04/12
        self.assertAlmostEqual(r['overall']['comparisons']['C-B']['mean_token_nll_difference'],expected,places=12)
        self.assertNotAlmostEqual(expected,r['overall']['comparisons']['C-B']['per_question']['mean'],places=5)

    def test_repeated_question_clusters_not_independent(self):
        sums,tokens,meta=self.data(); groups=[x[3] for x in meta]
        original=stats.crossed_bootstrap(sums,tokens,groups,repeats=80)
        repeated=stats.crossed_bootstrap(np.repeat(sums,2,axis=-1),np.repeat(tokens,2),np.repeat(groups,2),repeats=80)
        np.testing.assert_allclose(original,repeated,atol=1e-12)

    def test_identical_seed_repetitions_do_not_shrink_question_ci(self):
        sums,tokens,meta=self.data(); groups=[x[3] for x in meta]
        one=stats.crossed_bootstrap(sums[:,:1],tokens,groups,repeats=80)
        five=stats.crossed_bootstrap(sums,tokens,groups,repeats=80)
        np.testing.assert_allclose(one,five,atol=1e-12)

    def test_perfect_pairing_constant_arm_difference(self):
        sums,tokens,meta=self.data()
        r=stats.analyze(sums,tokens,meta,repeats=80)
        cb=r['overall']['comparisons']['C-B']
        np.testing.assert_allclose(cb['ci95'],[-.001,-.001],atol=1e-12)
        self.assertEqual(cb['better_coordinates'],10)
        self.assertEqual(r['decision']['label'],'观察到收益')
        self.assertEqual(cb['per_question']['improved'],4)

    def test_decision_insufficient_not_invalid_math(self):
        r=stats.decision(-.0002,[-.0008,.0001],[-.001,.0002],7,[-.0004,0.])
        self.assertEqual(r['label'],'证据不足'); self.assertTrue(r['not_a_mathematical_proof'])

    def test_decision_degradation(self):
        self.assertEqual(stats.decision(.001,[.0002,.002],[0,.001],0,[.001,.001])['label'],'观察到退化')

    def test_noninferiority_required_for_benefit(self):
        r=stats.decision(-.001,[-.002,-.0005],[-.001,.0006],10,[-.001,-.001])
        self.assertEqual(r['label'],'证据不足'); self.assertFalse(r['gates']['C_A_noninferiority'])

    def test_full_grid_artifact_alignment_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); paths=[]
            meta=[dict(index=i,sample_sha256=f's{i}',original_question_sha256=f'q{i}',family='GSM' if i<200 else 'MATH') for i in range(400)]
            common=dict(protocol=run.PROTOCOL,indices=dict(train=list(range(200))),metadata=dict(dev=meta),effective_tokens=dict(dev=[2]*400))
            for module in stats.MODULES:
                for seed in stats.SEEDS:
                    path=root/f'{module}_{seed}'; path.mkdir(); paths.append(path)
                    coordinate=dict(module=module,seed=seed,training_order=list(range(200)),common_fingerprint=run.digest(common))
                    fingerprint=run.digest(coordinate)
                    run.old.write_json(path/'plan.json',dict(common=common,coordinate=coordinate,fingerprint=fingerprint))
                    run.old.write_json(path/'report.json',dict(status='PASS',mode='train',input_files_unchanged=True,fingerprint=fingerprint))
                    for arm in stats.ARMS:
                        f=path/arm; f.mkdir()
                        before=[dict(index=i,source=meta[i],effective_tokens=2,nll=2.,nll_sum=4.,logits_sha256=f'logit{i}',logits_shape=[1,4,10],logits_dtype='float32') for i in range(400)]
                        after=copy.deepcopy(before)
                        delta=-.001 if arm=='C' else 0.
                        for row in after: row.update(nll=2.+delta,nll_sum=4.+2*delta)
                        result=dict(status='PASS',fingerprint=fingerprint,arm=arm,invariants=dict(base_tensors_verified=381),
                            before=dict(samples=before),after=dict(samples=after,token_weighted_nll=2.+delta),
                            training=dict(completed_steps=200,steps=[dict(step=i+1,source_index=i) for i in range(200)]))
                        run.old.write_json(f/'result.json',result); (f/'train.jsonl').write_text('test'); (f/'target_factors.pt').write_bytes(b'test')
                        run.old.write_json(f/'artifacts.json',{name:run.old.sha256(f/name) for name in ('result.json','train.jsonl','target_factors.pt')})
            sums,tokens,actual_meta,_=stats.load_runs(paths)
            self.assertEqual(sums.shape,(2,5,3,400)); self.assertEqual(int(tokens.sum()),800)
            self.assertEqual(len(actual_meta),400)
            broken=paths[-1]/'C'/'result.json'; obj=json.loads(broken.read_text()); obj['after']['samples'][0]['source']['sample_sha256']='wrong'
            run.old.write_json(broken,obj)
            art=broken.parent/'artifacts.json'; hashes=json.loads(art.read_text()); hashes['result.json']=run.old.sha256(broken); run.old.write_json(art,hashes)
            with self.assertRaisesRegex(ValueError,'manifest'): stats.load_runs(paths)

    def test_incomplete_grid_rejected(self):
        with self.assertRaisesRegex(ValueError,'all 2 modules'): stats.load_runs([])


if __name__=='__main__': unittest.main()
