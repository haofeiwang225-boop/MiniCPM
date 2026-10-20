"""Tiny/CPU-only subset audit tests; no real model loading."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from adaptive_lora.audit_joint_contributions import (checked_result,make_policies,validate_metadata,
    validate_isolation,policy_norms,evaluate_policies,interactions,fingerprint,execute)
from adaptive_lora.data_provenance import file_sha256
from test_cross_batch_acceptance import Tiny,sample,candidate


def result(names, selected, negative=None):
    negative=set(names) if negative is None else set(negative)
    return dict(pre_factors_restored=True,invariant_violations=[],groups={'proposal':[0],'selector':[1]},
        candidates=[dict(module=n,side='B',proposal=dict(mean=-1. if n in negative else 1.,per_sample=[-1. if n in negative else 1.]),
                         selector=dict(mean=-1. if n in selected else 1.,per_sample=[-1. if n in selected else 1.]),
                         selected=n in selected,check=dict(feasible=True)) for n in names])


def meta(groups):
    return dict(samples=[dict(original_question_sha256=g) for g in groups])


class JointTests(unittest.TestCase):
    def test_mask_algebra_and_seed(self):
        names=[str(i) for i in range(83)]
        fa=set(names[:24]); fb=set(names[:8]+names[24:46]); negative=names[:54]
        records=[dict(module=n,side='B') for n in names]
        a,b=result(names,fa,negative),result(names,fb,negative)
        state=random.getstate()
        policies,info=make_policies(a,b,records,42)
        self.assertEqual(info['counts'],dict(FA=24,FB=30,COMMON=8,FA_ONLY=16,FB_ONLY=22,proposal_negative=54))
        self.assertEqual(len(policies['UNION']),46)
        self.assertEqual(len(policies['RANDOM30']),30)
        self.assertTrue(set(policies['RANDOM30'])<=set(negative))
        self.assertEqual(state,random.getstate())
        self.assertEqual((policies,info),make_policies(a,b,records,42))

    def test_random_fallback_recorded(self):
        names=['a','b','c']; records=[dict(module=n,side='B') for n in names]
        r=result(names,{'a'},['a'])
        policies,info=make_policies(r,r,records,42,2)
        self.assertTrue(info['random_fallback_to_remaining'])
        self.assertIn('a',policies['RANDOM30'])

    def test_invalid_saved_mask_rejected(self):
        r=result(['a'],{'a'}); r['candidates'][0]['selected']=False
        with self.assertRaisesRegex(ValueError,'selection/slope'): checked_result(r,[dict(module='a',side='B')])

    def test_different_proposal_rejected(self):
        a=result(['a'],{'a'}); b=copy.deepcopy(a); b['groups']['proposal']=[2]
        with self.assertRaisesRegex(ValueError,'identical Proposal'): make_policies(a,b,[dict(module='a',side='B')],42,1)

    def test_duplicate_and_missing_candidates(self):
        r=result(['a'],{'a'}); r['candidates']*=2
        with self.assertRaises(ValueError): checked_result(r,[dict(module='a',side='B')])

    def test_original_group_isolation_not_just_index(self):
        train=meta(['q1','q2','q1','q3'])
        with self.assertRaisesRegex(ValueError,'original-question'):
            validate_isolation(train,train,[1],[2],{0},False)
        valid=validate_isolation(train,train,[1],[3],{0},False)
        self.assertEqual(valid['original_group_overlap'],0)
        with self.assertRaises(ValueError): validate_isolation(train,meta(['q1']),[1],[0],{1},True)
        self.assertEqual(validate_isolation(train,meta(['q4']),[1],[0],{1},True)['evaluation_scope'],'held_out_dataset')

    def test_metadata_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.json'; raw=[dict(messages=[])]
            path.write_text(json.dumps(raw))
            m=dict(data_sha256=file_sha256(path),num_samples=1,samples=[dict(index=0,sample_sha256=fingerprint(raw[0]),original_question_sha256='a'*64)])
            self.assertEqual(validate_metadata(path,m),raw)
            m['samples'][0]['sample_sha256']='b'*64
            with self.assertRaises(ValueError): validate_metadata(path,m)

    def payload(self):
        from test_cross_batch_acceptance import CrossBatchTests
        helper=CrossBatchTests(); helper.setUp()
        return helper.model,helper.payloads()[0]

    def test_norm_matches_dense_and_joint_energy(self):
        model,pre=self.payload(); c=candidate(model)
        pre['layers']['other']=copy.deepcopy(pre['layers']['layer'])
        records=[dict(module='layer',side='B',target=c['target']),dict(module='other',side='B',target=c['target'])]
        norms=policy_norms(pre,records,dict(PRE=[],ONE=['layer'],TWO=['layer','other']))
        a,b=model.layer.weight_a.detach(),model.layer.weight_b.detach()
        dense=float(torch.linalg.norm(model.layer.lora_scaler*((c['target']-b).double()@a.double())))
        self.assertAlmostEqual(norms['ONE']['lora_update_fro_norm'],dense)
        self.assertAlmostEqual(norms['TWO']['lora_update_fro_norm'],2**.5*dense)
        self.assertEqual(norms['PRE']['lora_update_fro_norm'],0)

    def test_combination_evaluation_restores_params_rng(self):
        model=Tiny(); c=candidate(model)
        data=[sample(0),sample(3,3)]
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        identities={n:id(p) for n,p in model.named_parameters()}; rng=torch.get_rng_state().clone()
        out=evaluate_policies(model,[c],dict(PRE=[],ALL=['layer']),
                              dict(calibration=(data,[0]),evaluation=(data,[1])),.05)
        self.assertEqual(out['ALL']['evaluation']['samples'][0]['index'],1)
        self.assertEqual(out['ALL']['evaluation']['effective_tokens'],3)
        self.assertIn('nll',out['ALL']['evaluation']['samples'][0])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for n,p in model.named_parameters():
            self.assertEqual(id(p),identities[n]); self.assertTrue(torch.equal(p,before[n]))

    def test_exception_restores_updated_factors(self):
        model=Tiny(); c=candidate(model); before=c['parameter'].clone()
        def loss(*args): raise RuntimeError('injected')
        with patch('adaptive_lora.audit_joint_contributions.evaluate_dataset',side_effect=loss):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                evaluate_policies(model,[c],dict(ALL=['layer'],PRE=[]),dict(calibration=([sample()],[0])),.05)
        self.assertTrue(torch.equal(c['parameter'],before))

    def test_interaction_contrast(self):
        losses=dict(PRE=10.,COMMON=9.,FA_ONLY=9.5,FB_ONLY=9.8,FA=8.2,FB=8.7,UNION=7.5,RANDOM30=9.)
        results={k:dict(calibration=dict(token_weighted_nll=v),evaluation=dict(token_weighted_nll=v)) for k,v in losses.items()}
        out=interactions(results)['evaluation']
        self.assertAlmostEqual(out['FA_block_interaction'],-.3)
        self.assertAlmostEqual(out['FB_block_interaction'],-.1)
        self.assertAlmostEqual(out['union_block_interaction'],-.8)

    def test_singleton_interaction_and_no_score_based_selection(self):
        names=['a','b']; records=[dict(module=n,side='B') for n in names]
        r=result(names,set(names)); r['policies']={'filtered': {'token_weighted_nll':float('nan')}}
        policies,_=make_policies(r,r,records,42,1,True)
        self.assertEqual(policies['SINGLE::a'],['a'])
        self.assertEqual(policies['SINGLE::b'],['b'])

    def test_execute_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'existing output'):
                execute(Path(tmp)/'absent',Path(tmp),'cpu')


if __name__=='__main__': unittest.main()
