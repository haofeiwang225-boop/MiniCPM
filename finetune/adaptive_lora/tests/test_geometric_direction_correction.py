import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from adaptive_lora.direction_correction import (
    grassmann_geodesic, factor_product_fro_norm, relative_A_projection_change,
    relative_B_projection_change, plan_geometric_direction_correction, apply_geometric_correction)
from adaptive_lora.subspace_matching import orth
from adaptive_lora.dense_init import make_dense_layer
from adaptive_lora.official import LoRAConfig
from adaptive_lora.adaptation import geometric_correction_step


def example(side='B'):
    g=torch.diag(torch.tensor([4.,3.,.1,0.]))
    a=torch.tensor([[1.,0.,0.,0.]])
    b=torch.tensor([[0.],[.2],[.98],[0.]])
    if side=='A': a,b=b.T,a.T
    return g,a,b


def plan(side='B',**kwargs):
    g,a,b=example(side)
    oracle=25/25.01
    return a,b,*plan_geometric_direction_correction(g,a,b,oracle,2.,**kwargs)


def dense(a,b):
    layer=make_dense_layer(nn.Linear(a.shape[1],b.shape[0],bias=False),
        LoRAConfig(a.shape[1],b.shape[0],lora_rank=a.shape[0],lora_scaler=2.,run_lora_in_fp32=True))
    layer.weight.requires_grad_(False)
    with torch.no_grad(): layer.weight_a.copy_(a);layer.weight_b.copy_(b)
    return layer


class GeometricTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(33)

    def test_endpoints_and_orthonormality(self):
        q0,q1=orth(torch.randn(10,3)),orth(torch.randn(10,3))
        for eta in (0.,.03125,.5,1.):
            q=grassmann_geodesic(q0,q1,eta)
            torch.testing.assert_close(q.T@q,torch.eye(3),atol=1e-6,rtol=1e-6)
            if eta in (0.,1.):
                target=q0 if eta==0 else q1
                torch.testing.assert_close(q@q.T,target@target.T,atol=1e-6,rtol=1e-6)

    def test_basis_invariance_both_bases(self):
        q0,q1=orth(torch.randn(12,3)),orth(torch.randn(12,3))
        r0,r1=orth(torch.randn(3,3)),orth(torch.randn(3,3))
        x=grassmann_geodesic(q0,q1,.25)
        y=grassmann_geodesic(q0@r0,q1@r1,.25)
        torch.testing.assert_close(x@x.T,y@y.T,atol=2e-6,rtol=2e-6)
        b=torch.randn(12,3)
        torch.testing.assert_close(x@(x.T@b),y@(y.T@b),atol=3e-6,rtol=3e-6)

    def test_cut_locus_basis_invariance(self):
        q0,q1=torch.eye(6)[:,:2],torch.eye(6)[:,2:4]
        r=orth(torch.randn(2,2))
        x=grassmann_geodesic(q0,q1,.5)
        y=grassmann_geodesic(q0@r,q1@r.T,.5)
        torch.testing.assert_close(x@x.T,y@y.T,atol=2e-6,rtol=2e-6)

    def test_same_subspace_path(self):
        q=orth(torch.randn(8,3));r=orth(torch.randn(3,3))
        x=grassmann_geodesic(q,q@r,.5)
        torch.testing.assert_close(x@x.T,q@q.T,atol=2e-6,rtol=2e-6)

    def test_gram_norm_matches_dense(self):
        a,b=torch.randn(3,9),torch.randn(7,3)
        self.assertAlmostEqual(factor_product_fro_norm(b,a),float((b.double()@a.double()).norm()),places=10)
        an,bn=a+.01,b+.01
        for rel,change,den in (relative_A_projection_change(a,b,an),relative_B_projection_change(a,b,bn)):
            self.assertAlmostEqual(rel,change/den,places=12)

    def test_B_side_selected_trust_gain_and_single_mutation(self):
        self.check_side('B')

    def test_A_side_selected_trust_gain_and_single_mutation(self):
        self.check_side('A')

    def check_side(self,side):
        a,b,row,factors=plan(side)
        self.assertEqual(row['applied_side'],side)
        self.assertGreaterEqual(row['applied_gain'],.001)
        self.assertLessEqual(row['applied_relative_update_change'],.05)
        layer=dense(a,b);base=layer.weight.detach().clone();scale=layer.lora_scaler
        ids=[id(layer.weight_a),id(layer.weight_b)]
        count=sum(p.numel() for p in layer.parameters())
        opt=torch.optim.AdamW([layer.weight_a,layer.weight_b])
        apply_geometric_correction(layer,factors,row,opt)
        self.assertTrue(torch.equal(layer.weight_a,a) if side=='B' else torch.equal(layer.weight_b,b))
        self.assertFalse(torch.equal(layer.weight_b,b) if side=='B' else torch.equal(layer.weight_a,a))
        self.assertTrue(torch.equal(base,layer.weight))
        self.assertEqual(ids,[id(layer.weight_a),id(layer.weight_b)])
        self.assertEqual(count,sum(p.numel() for p in layer.parameters()))
        self.assertEqual(scale,layer.lora_scaler)
        self.assertEqual(orth(layer.weight_a.T).shape[1],1)
        self.assertEqual(orth(layer.weight_b).shape[1],1)
        delta=(layer.weight_b.double()@layer.weight_a.double())-(b.double()@a.double())
        self.assertLessEqual(float(delta.norm()/((b.double()@a.double()).norm()+1e-12)),.05)

    def test_no_feasible_candidate_is_noop(self):
        for options in (dict(function_tolerance=0.),dict(min_gain=2.)):
            a,b,row,factors=plan(**options)
            self.assertEqual(row['applied_side'],'none')
            self.assertIsNone(factors)
            self.assertEqual(row['correction_skip_reason'],'no_feasible_candidate')

    def test_rank_deficient_and_near_zero(self):
        g,a,b=example()
        row,factors=plan_geometric_direction_correction(g,a,b*0,1.,2.)
        self.assertEqual(row['correction_skip_reason'],'rank_deficient_factor')
        self.assertIsNone(factors)
        row,_=plan_geometric_direction_correction(g,a*1e-9,b*1e-9,1.,2.)
        self.assertEqual(row['correction_skip_reason'],'near_zero_lora_update')

    def test_capacity_deferred(self):
        g=torch.eye(10);a=torch.eye(10)[:1];b=torch.eye(10)[:,1:2]
        row,factors=plan_geometric_direction_correction(g,a,b,.2,2.)
        self.assertEqual(row['correction_skip_reason'],'deferred_capacity_no_expansion_in_innovation3')
        self.assertIsNone(factors)

    def test_target_rank_deficient(self):
        from adaptive_lora.direction_correction import evaluate_A_candidate
        g=torch.zeros(4,4);a=torch.eye(4)[:2];b=torch.eye(4)[:,:2]
        result,factors=evaluate_A_candidate(g,a,b,1.,[.5],.05,.001)
        self.assertEqual(result['target_rank'],0)
        self.assertFalse(result['feasible'])
        self.assertIsNone(factors)

    def test_optimizer_selective_reset_scheduler_preserved(self):
        for side in ('A','B'):
            a,b,row,factors=plan(side)
            layer=dense(a,b);unrelated=nn.Parameter(torch.ones(3))
            opt=torch.optim.AdamW([layer.weight_a,layer.weight_b,unrelated],lr=.001)
            sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda t:1-t/10)
            for p in (layer.weight_a,layer.weight_b,unrelated):p.grad=torch.ones_like(p)
            opt.step();sched.step();opt.zero_grad(set_to_none=True)
            # Restore planned point without disturbing moments.
            with torch.no_grad():layer.weight_a.copy_(a);layer.weight_b.copy_(b)
            old=copy.deepcopy(opt.state_dict());before=copy.deepcopy(sched.state_dict())
            applied=layer.weight_a if side=='A' else layer.weight_b
            keep=layer.weight_b if side=='A' else layer.weight_a
            keep_state=copy.deepcopy(opt.state[keep]);other_state=copy.deepcopy(opt.state[unrelated])
            result=apply_geometric_correction(layer,factors,row,opt)
            self.assertEqual(result['optimizer_state_reset_count'],1)
            self.assertNotIn(applied,opt.state)
            for p,state in ((keep,keep_state),(unrelated,other_state)):
                for key,value in state.items():self.assertTrue(torch.equal(opt.state[p][key],value))
            self.assertEqual(sched.state_dict(),before)
            self.assertEqual(opt.param_groups[0]['lr'],old['param_groups'][0]['lr'])

    def test_planner_preserves_rng_and_factors(self):
        g,a,b=example();before=(a.clone(),b.clone(),torch.get_rng_state().clone())
        plan_geometric_direction_correction(g,a,b,1.,2.)
        for old,new in zip(before,(a,b,torch.get_rng_state())):self.assertTrue(torch.equal(old,new))

    def test_event_does_not_replace_optimizer_or_parameters(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();_,a,b=example();self.proj=dense(a,b)
            def forward(self,x,labels):return SimpleNamespace(loss=(self.proj(x)-labels).square().mean())
        model=Toy();batch=dict(x=torch.randn(3,4),labels=torch.randn(3,4))
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1.)
        model(**batch).loss.backward();opt.step();sched.step();opt.zero_grad(set_to_none=True)
        before=model.proj.weight.detach().clone();rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(analysis_device='cpu',adaptive_residual_energy_target=.9,
                adaptive_rank_max=4,adaptive_direction_efficiency_threshold=.8,adaptive_oracle_coverage_threshold=.9,
                adaptive_residual_energy_threshold=.1,adaptive_rank_step_cap=2,direction_function_tolerance=.05,
                direction_min_gain=.001,direction_eta_start=1.,direction_eta_decay=.5,direction_eta_min=.03125,
                calibration_indices_path=Path(tmp)/'manifest',calibration_indices_sha256='test',output_dir=Path(tmp))
            summary=geometric_correction_step(model,['proj'],[batch],args,opt,sched,2)
            self.assertTrue(summary['optimizer_object_preserved'] and summary['scheduler_preserved'])
            self.assertEqual(summary['old_total_lora_params'],summary['new_total_lora_params'])
        self.assertTrue(torch.equal(before,model.proj.weight))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_all_layers_measured_before_mutation(self):
        from unittest.mock import patch
        model=nn.Module()
        for name,side in (('first','A'),('second','B')):
            _,a,b=example(side);setattr(model,name,dense(a,b))
        params=[p for p in model.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(params)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1.)
        for p in params:p.grad=torch.ones_like(p)
        opt.step();sched.step();opt.zero_grad(set_to_none=True)
        for name,side in (('first','A'),('second','B')):
            _,a,b=example(side)
            with torch.no_grad():model.get_submodule(name).weight_a.copy_(a);model.get_submodule(name).weight_b.copy_(b)
        snapshot=copy.deepcopy(model.state_dict());calls=[]
        def fake_probe(model,layer,batches,return_losses=False):
            for k,v in snapshot.items():self.assertTrue(torch.equal(model.state_dict()[k],v))
            calls.append(layer)
            return example()[0], [1.]
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(analysis_device='cpu',adaptive_residual_energy_target=.9,
                adaptive_rank_max=4,adaptive_direction_efficiency_threshold=.8,adaptive_oracle_coverage_threshold=.9,
                adaptive_residual_energy_threshold=.1,adaptive_rank_step_cap=2,direction_function_tolerance=.05,
                direction_min_gain=.001,direction_eta_start=1.,direction_eta_decay=.5,direction_eta_min=.03125,
                calibration_indices_path=Path(tmp)/'manifest',calibration_indices_sha256='test',output_dir=Path(tmp))
            with patch('adaptive_lora.adaptation.probe_weight',side_effect=fake_probe):
                summary=geometric_correction_step(model,['first','second'],[{}],args,opt,sched,2)
            self.assertEqual(len(calls),2)
            self.assertEqual(summary['applied_sides'],{'A':1,'B':1})
            self.assertEqual(summary['optimizer_state_reset_count'],2)
