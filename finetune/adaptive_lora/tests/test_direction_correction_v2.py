import copy
from contextlib import ExitStack
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from transformers import default_data_collator
from adaptive_lora.direction_correction_v2 import (factor_directional_derivatives, passes_direction_check,
    joint_loss_search, mean_sample_loss, classify_loss_change)
from adaptive_lora.adaptation_v2 import geometric_v2_step, compute_rsc_candidates
from adaptive_lora.rsc_calibration import write_rsc_manifest, load_rsc_samples, training_source
from adaptive_lora.calibration import write_calibration_indices
from test_geometric_direction_correction import dense


class Toy(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.proj = dense(torch.tensor([[1., .2]]), torch.tensor([[1.], [.3]]))
        self.extra = nn.Parameter(torch.tensor(0.))
        self.to(dtype=dtype)
        self.dropout = nn.Dropout(.5)

    def forward(self, x, labels):
        # Tests have true batch size 1 with variable token lengths.
        y = self.proj(self.dropout(x)) + self.extra
        return SimpleNamespace(loss=(y.float()-labels.float()).square().mean())


def make_args(folder, dataset):
    folder = Path(folder)
    args = SimpleNamespace(direction_correction='geometric_v2', adaptive_lora=False, lora_structure='dense',
        model_name_or_path='toy', analysis_device='cpu', output_dir=folder, train_data_path=folder/'train.json',
        tiny_model=True, rsc_selection_samples=2, rsc_selection_indices=None,
        rsc_proposal_descent_margin=0.,rsc_selection_descent_margin=0.,rsc_loss_min_improvement=0.,
        rsc_trial_strengths=[1., .5, .25], direction_function_tolerance=.5,
        adaptive_residual_energy_threshold=.1,direction_min_gain=.001,
        direction_eta_start=1.,direction_eta_decay=.5,direction_eta_min=.03125)
    args.calibration_indices_path=folder/'calibration.json'
    args.calibration_indices_sha256=write_calibration_indices(args.calibration_indices_path,dataset,1,2,'synthetic_training')
    args.rsc_manifest_path=folder/'rsc.json'
    args.rsc_manifest_sha256=write_rsc_manifest(args.rsc_manifest_path,dataset,args)
    return args


def data(dtype=torch.float32):
    return [dict(x=torch.full((i+1,2), 1.+i*.1,dtype=dtype), labels=torch.zeros((i+1,2),dtype=dtype)) for i in range(4)]


def candidate(model, side='B', multiplier=.9):
    p=model.proj.weight_b if side=='B' else model.proj.weight_a
    return dict(module='proj',layer=model.proj,parameter=p,side=side,target=p.detach().cpu().clone()*multiplier)


def initialized(dtype=torch.float32):
    model=Toy(dtype)
    with torch.no_grad():model.proj.weight.zero_()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    sched=torch.optim.lr_scheduler.StepLR(opt,step_size=5)
    for p in model.parameters():
        if p.requires_grad:p.grad=torch.ones_like(p)
    opt.step();sched.step();opt.zero_grad(set_to_none=True)
    return model,opt,sched


class RSCTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(123)

    def test_direction_gate_signs_and_thresholds(self):
        self.assertTrue(passes_direction_check(-.1,-.2))
        self.assertFalse(passes_direction_check(-.1,.2))
        self.assertFalse(passes_direction_check(.1,-.2))
        self.assertFalse(passes_direction_check(0.,-.2))
        self.assertFalse(passes_direction_check(-.1,-.2,.15,0.))

    def test_strict_actual_loss_classification_and_positive_margin(self):
        self.assertEqual(classify_loss_change(1.,1.)['status'],'numerically_equal')
        self.assertFalse(classify_loss_change(1.,1.)['accepted'])
        self.assertEqual(classify_loss_change(1.,1.1)['status'],'loss_increase')
        self.assertTrue(classify_loss_change(1.,.9)['accepted'])
        self.assertFalse(classify_loss_change(1.,.9,.2)['accepted'])
        self.assertTrue(classify_loss_change(1.,.5,.5)['accepted'])

    def test_factor_gradient_exact_sample_mean_and_state_restore(self):
        model, opt, sched=initialized(); model.train()
        p=model.proj.weight_b; p.grad=torch.ones_like(p)
        grad=p.grad; flags=[x.requires_grad for x in model.parameters()]
        rng=torch.get_rng_state().clone(); c=candidate(model)
        batches=[default_data_collator([r]) for r in data()[:2]]
        result=factor_directional_derivatives(model,[c],batches)['proj']
        # Unequal token lengths must still have equal sample weights.
        expected=[]
        model.eval()
        for batch in batches:
            g=torch.autograd.grad(model(**batch).loss,p)[0]
            expected.append(float((g.double()*(c['target']-p.detach()).double()).sum()))
        model.train()
        self.assertAlmostEqual(result['mean'],sum(expected)/2,places=6)
        self.assertIs(p.grad,grad);self.assertTrue(torch.equal(grad,torch.ones_like(grad)))
        self.assertEqual(flags,[x.requires_grad for x in model.parameters()])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(model.training)

    def test_joint_loss_worse_rolls_back(self):
        model,_,_=initialized(); c=candidate(model,multiplier=1.1)
        batches=[default_data_collator([r]) for r in data()[:2]]
        before=copy.deepcopy(model.state_dict())
        values,report=joint_loss_search(model,[c],batches,batches,[1.,.5],.5)
        self.assertIsNone(values);self.assertEqual(report['reason'],'no_joint_trial_accepted')
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))

    def test_temporary_exception_rolls_back(self):
        model,_,_=initialized(); c=candidate(model)
        batches=[default_data_collator([data()[0]])]
        before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
        original=model.forward;calls=[]
        def forward(**batch):
            calls.append(1)
            if len(calls)==3:raise RuntimeError('injected trial failure')
            return original(**batch)
        with patch.object(model,'forward',side_effect=forward), self.assertRaisesRegex(RuntimeError,'injected'):
            joint_loss_search(model,[c],batches,batches,[1.],.5)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(model.training)

    def run_event(self, side='B', reject=False, dtype=torch.float32, no_candidate=False, trial_error=False, overshoot=False, conflict=False):
        model,opt,sched=initialized(dtype);ds=data(dtype)
        if conflict:
            for sample in ds[2:]:sample['labels'].fill_(100)
        before=copy.deepcopy(model.state_dict());ids=[id(p) for p in model.parameters()]
        states={p:(s,copy.deepcopy(s),{k:v._version for k,v in s.items()}) for p,s in opt.state.items()}
        schedule=copy.deepcopy(sched.state_dict());rng=torch.get_rng_state().clone()
        py_rng=random.getstate();np_rng=np.random.get_state()
        c=candidate(model,side,multiplier=-3. if overshoot else .9)
        plan=dict(applied_side=side, applied_gain=.01, applied_eta=.25,correction_skip_reason=None)
        factors=(c['target'],model.proj.weight_b.detach().clone()) if side=='A' else (model.proj.weight_a.detach().clone(),c['target'])
        with tempfile.TemporaryDirectory() as tmp:
            args=make_args(tmp,ds)
            if reject:args.rsc_proposal_descent_margin=1e10
            if overshoot:
                args.direction_function_tolerance=10.
                args.rsc_trial_strengths=[1.]
            with patch('adaptive_lora.adaptation_v2.plan_layer',return_value=(plan,None if no_candidate else factors)):
                if trial_error:
                    with patch('adaptive_lora.direction_correction_v2.trial_factors',side_effect=RuntimeError('trial injected')):
                        with self.assertRaisesRegex(RuntimeError,'trial injected'):
                            geometric_v2_step(model,['proj'],ds,default_data_collator,args,opt,sched,2)
                    result=dict(accepted=False)
                else:
                    result=geometric_v2_step(model,['proj'],ds,default_data_collator,args,opt,sched,2)
                    saved=json.loads((Path(tmp)/'corrections_v2/step_2/summary.json').read_text())
                    self.assertEqual(saved,result)
        self.assertEqual(ids,[id(p) for p in model.parameters()]);self.assertEqual(schedule,sched.state_dict())
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(py_rng,random.getstate())
        self.assertTrue(np.array_equal(np_rng[1],np.random.get_state()[1]))
        self.assertTrue(torch.equal(before['proj.weight'],model.proj.weight))
        selected=model.proj.weight_a if side=='A' else model.proj.weight_b
        for name,p in model.named_parameters():
            if result['accepted'] and p is selected:
                self.assertNotIn(p,opt.state);self.assertFalse(torch.equal(before[name],p))
            else:
                self.assertTrue(torch.equal(before[name],p))
                if p in states:
                    state,values,versions=states[p];self.assertIs(opt.state[p],state)
                    for k,v in state.items():
                        self.assertTrue(torch.equal(v,values[k]));self.assertEqual(v._version,versions[k])
        return result

    def test_accepted_only_B_state_reset(self):
        result=self.run_event('B');self.assertTrue(result['accepted']);self.assertEqual(result['optimizer_state_reset_count'],1)

    def test_accepted_only_A_state_reset(self):
        self.assertTrue(self.run_event('A')['accepted'])

    def test_no_qualified_layers_strict_noop(self):
        self.assertFalse(self.run_event(reject=True)['accepted'])

    def test_selection_positive_slope_rejects_event(self):
        result=self.run_event(conflict=True)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['num_direction_qualified'],0)

    def test_joint_overshoot_does_not_reset_adam(self):
        result=self.run_event(overshoot=True)
        self.assertEqual(result['num_direction_qualified'],1)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['joint_loss_check']['reason'],'no_joint_trial_accepted')

    def test_no_geometric_candidates_noop(self):
        self.assertFalse(self.run_event(no_candidate=True)['accepted'])

    def test_event_exception_does_not_reset_adam(self):
        self.assertFalse(self.run_event(trial_error=True)['accepted'])

    def test_bfloat16_and_float32_paths(self):
        for dtype in (torch.float32,torch.bfloat16):
            result=self.run_event(dtype=dtype)
            self.assertTrue(result['accepted'])
            self.assertEqual(result['old_total_lora_params'],result['new_total_lora_params'])

    def test_training_only_disjoint_manifest(self):
        ds=data()
        with tempfile.TemporaryDirectory() as tmp:
            args=make_args(tmp,ds)
            proposal,selection,manifest=load_rsc_samples(ds,default_data_collator,args)
            self.assertFalse(set(manifest['proposal_indices'])&set(manifest['selection_indices']))
            ds[2]['x'].add_(1)
            with self.assertRaisesRegex(ValueError,'sample changed'):load_rsc_samples(ds,default_data_collator,args)
        for source in ('dev.json','test.json','benchmarks/math500/train.json'):
            with self.assertRaises(ValueError):training_source(source,False)

    def test_overlap_and_duplicate_content_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds=data();args=make_args(tmp,ds)
            args.rsc_selection_indices=[0,2]
            with self.assertRaisesRegex(ValueError,'overlap'):write_rsc_manifest(Path(tmp)/'other.json',ds,args)
            args.rsc_selection_indices=[2,3];ds[2]=copy.deepcopy(ds[0])
            with self.assertRaisesRegex(ValueError,'duplicate encoded'):write_rsc_manifest(Path(tmp)/'other.json',ds,args)

    def test_cli_constraints_and_v1_default(self):
        from train_adaptive_lora import get_args
        base=['train','--output_dir','/tmp/rsc_cli','--tiny_model','--direction_correction','geometric_v2',
              '--direction_correction_steps','2','--max_steps','5']
        with patch('sys.argv',base):self.assertEqual(get_args().direction_correction,'geometric_v2')
        for extra in (['--adaptive_lora','true'],['--batch_size','2'],['--rsc_trial_strengths','.5','1'],
                      ['--rsc_loss_min_improvement','-1'],['--diagnostic_steps','2']):
            with patch('sys.argv',base+extra),patch('sys.stderr'),self.assertRaises(SystemExit):get_args()
        with patch('sys.argv',['train','--output_dir','/tmp/rsc_cli']):
            self.assertEqual(get_args().direction_correction,'none')


class TwoFactorToy(Toy):
    def __init__(self, mixed=False, equal_loss=False):
        super().__init__()
        self.other = dense(torch.tensor([[.8,.4]]),torch.tensor([[.7],[.5]]))
        self.equal_loss = equal_loss
        for layer in (self.proj,self.other):
            with torch.no_grad():
                layer.weight.zero_()
            if mixed:
                # Match the real supported mixed storage: BF16 base, FP32 factors.
                layer.weight.data = layer.weight.data.to(torch.bfloat16)
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.to(torch.bfloat16)

    def forward(self,x,labels):
        loss = super().forward(x,labels).loss
        other = self.other(x).float()
        loss = loss + (other-labels.float()).square().mean()
        if self.equal_loss:
            # Cast has a nonzero autograd derivative, but finite changes round
            # to the identical BF16 loss. A negative slope must NOT authorize it.
            loss = (16. + loss*1e-4).to(torch.bfloat16).float()
        return SimpleNamespace(loss=loss)


class RSCTransactionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(321)

    def snapshot(self,model,opt,sched):
        return dict(
            params={n:(p,p.detach().clone(),p.requires_grad,p.grad,
                       None if p.grad is None else p.grad.clone()) for n,p in model.named_parameters()},
            modes=[(m,m.training) for m in model.modules()],
            states={p:(s,{k:(v,v.clone(),v._version) if torch.is_tensor(v) else copy.deepcopy(v)
                          for k,v in s.items()}) for p,s in opt.state.items()},
            state_object=opt.state,
            groups=[(g,list(g['params']),copy.deepcopy({k:v for k,v in g.items() if k!='params'}))
                    for g in opt.param_groups],
            schedule=copy.deepcopy(sched.state_dict()),
            ranks=[(m,m.lora_rank,m.lora_scaler) for m in (model.proj,model.other)],
            rng=torch.get_rng_state().clone(),python=random.getstate(),numpy=np.random.get_state())

    def assert_preserved(self,model,opt,sched,snapshot,changed=()):
        changed=set(changed)
        self.assertEqual(set(dict(model.named_parameters())),set(snapshot['params']))
        for n,p in model.named_parameters():
            old,value,flag,grad,grad_value=snapshot['params'][n]
            self.assertIs(p,old)
            self.assertEqual(p.shape,value.shape)
            self.assertEqual(p.requires_grad,flag)
            self.assertIs(p.grad,grad)
            if grad is not None:self.assertTrue(torch.equal(grad,grad_value))
            self.assertTrue(torch.isfinite(p).all())
            self.assertEqual(torch.equal(p,value),p not in changed)
        self.assertEqual(set(opt.state),set(snapshot['states'])-changed)
        self.assertIs(opt.state,snapshot['state_object'])
        for p,(state,fields) in snapshot['states'].items():
            if p in changed:continue
            self.assertIs(opt.state[p],state)
            self.assertEqual(set(state),set(fields))
            for k,old in fields.items():
                if torch.is_tensor(state[k]):
                    ref,value,version=old
                    self.assertIs(state[k],ref)
                    self.assertTrue(torch.equal(state[k],value))
                    self.assertEqual(state[k]._version,version)
                else:self.assertEqual(state[k],old)
        self.assertEqual(len(opt.param_groups),len(snapshot['groups']))
        for actual,(old,ps,options) in zip(opt.param_groups,snapshot['groups']):
            self.assertIs(actual,old)
            self.assertEqual([id(p) for p in actual['params']],[id(p) for p in ps])
            self.assertEqual({k:v for k,v in actual.items() if k!='params'},options)
        self.assertEqual(sched.state_dict(),snapshot['schedule'])
        for m,rank,scale in snapshot['ranks']:
            self.assertEqual((m.lora_rank,m.lora_scaler),(rank,scale))
        for m,mode in snapshot['modes']:self.assertEqual(m.training,mode)
        self.assertTrue(torch.equal(torch.get_rng_state(),snapshot['rng']))
        self.assertEqual(random.getstate(),snapshot['python'])
        np_state=np.random.get_state()
        self.assertEqual(np_state[0],snapshot['numpy'][0])
        self.assertTrue(np.array_equal(np_state[1],snapshot['numpy'][1]))
        self.assertEqual(np_state[2:],snapshot['numpy'][2:])

    def event(self,failure=None,mixed=False,equal_loss=False):
        model=TwoFactorToy(mixed,equal_loss)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        sched=torch.optim.lr_scheduler.StepLR(opt,5)
        for p in model.parameters():
            if p.requires_grad:p.grad=torch.ones_like(p)
        opt.step();sched.step();opt.zero_grad(set_to_none=True)
        ds=data(torch.bfloat16 if mixed else torch.float32)
        selected=[model.proj.weight_a,model.other.weight_b]
        originals=[p.detach().clone() for p in selected]
        injected=[]

        def confirm_committed():
            self.assertTrue(all(not torch.equal(p,old) for p,old in zip(selected,originals)))

        def plan(_model,layer,_proposal,_args):
            side='A' if layer is model.proj else 'B'
            a,b=layer.weight_a.detach().clone(),layer.weight_b.detach().clone()
            if side=='A':a.mul_(.9)
            else:b.mul_(.9)
            return dict(applied_side=side,applied_gain=.01,applied_eta=.25,
                        correction_skip_reason=None),(a,b)

        if failure=='partial_adam':
            class FailingPop(dict):
                def pop(mapping,key,default=None):
                    value=super().pop(key,default)
                    if any(key is p for p in selected) and not injected:
                        confirm_committed()
                        self.assertNotIn(key,mapping)
                        self.assertTrue(any(p in mapping for p in selected))
                        injected.append('partial_adam')
                        raise RuntimeError('injected partial Adam reset failure')
                    return value
            opt.state=FailingPop(opt.state)
        # Pre-existing gradients and mixed module modes must survive even failures.
        for p in selected:p.grad=torch.full_like(p,.123)
        model.train();model.other.eval()
        with tempfile.TemporaryDirectory() as tmp:
            args=make_args(tmp,ds)
            snap=self.snapshot(model,opt,sched)
            original_schedule=sched.state_dict
            original_write=Path.write_text
            def schedule_state():
                if not injected and all(not torch.equal(p,v) for p,v in zip(selected,originals)):
                    confirm_committed()
                    injected.append('after_commit')
                    raise RuntimeError('injected after multiple committed factors')
                return original_schedule()
            def write(path,*a,**kw):
                if path.name=='summary.json':
                    confirm_committed()
                    self.assertTrue(all(p not in opt.state for p in selected))
                    injected.append('final_log')
                    raise RuntimeError('injected final summary write failure')
                return original_write(path,*a,**kw)
            with ExitStack() as stack:
                stack.enter_context(patch('adaptive_lora.adaptation_v2.plan_layer',side_effect=plan))
                if failure=='after_commit':stack.enter_context(patch.object(sched,'state_dict',side_effect=schedule_state))
                if failure=='final_log':stack.enter_context(patch.object(Path,'write_text',new=write))
                if failure:
                    with self.assertRaisesRegex(RuntimeError,'injected'):
                        geometric_v2_step(model,['proj','other'],ds,default_data_collator,args,opt,sched,2)
                    result=json.loads((Path(tmp)/'corrections_v2/step_2/error.json').read_text())
                    self.assertEqual(injected,[failure])
                    self.assertFalse(result['accepted'])
                    self.assertTrue(result['factors_rolled_back'])
                    rows=result['module_abort_diagnostics']
                else:
                    result=geometric_v2_step(model,['proj','other'],ds,default_data_collator,args,opt,sched,2)
                    rows=[json.loads(line) for line in (Path(tmp)/'corrections_v2/step_2/modules.jsonl').read_text().splitlines()]
            changed=selected if result['accepted'] else []
            self.assert_preserved(model,opt,sched,snap,changed)
            if equal_loss:
                self.assertFalse(result['accepted'])
                self.assertEqual(result['optimizer_state_reset_count'],0)
                self.assertEqual(result['num_direction_qualified'],2)
                for row in rows:
                    self.assertLess(row['proposal_derivative']['mean'],0)
                    self.assertLess(row['selection_derivative']['mean'],0)
                for trial in result['joint_loss_check']['attempts']:
                    self.assertTrue(all(check['changed'] for check in trial['checks']))
                    self.assertEqual(trial['reason'],'numerically_equal')
                    for change in trial['loss_changes'].values():
                        self.assertEqual(change['status'],'numerically_equal')
                        self.assertEqual(change['improvement'],0.)
            if mixed:
                for layer in (model.proj,model.other):
                    self.assertEqual(layer.weight.dtype,torch.bfloat16)
                    self.assertEqual(layer.weight_a.dtype,torch.float32)
                    self.assertEqual(layer.weight_b.dtype,torch.float32)
            return result

    def test_multiple_committed_factors_exception_rolls_back(self):
        self.event(failure='after_commit')

    def test_partial_adam_reset_exception_restores_state_objects(self):
        self.event(failure='partial_adam')

    def test_final_log_failure_restores_parameters_and_optimizer(self):
        self.event(failure='final_log')

    def test_bf16_equal_loss_rejects_nonzero_negative_derivative_candidates(self):
        self.event(mixed=True,equal_loss=True)

    def test_bf16_base_fp32_factors_transaction(self):
        result=self.event(mixed=True)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['optimizer_state_reset_count'],2)

    def test_shared_offline_computation_returns_candidates_without_committing(self):
        model=TwoFactorToy()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        sched=torch.optim.lr_scheduler.StepLR(opt,5)
        for p in model.parameters():
            if p.requires_grad:p.grad=torch.ones_like(p)
        opt.step();sched.step();opt.zero_grad(set_to_none=True)
        ds=data()
        def plan(_model,layer,_proposal,_args):
            return dict(applied_side='B',applied_gain=.01,correction_skip_reason=None), (
                layer.weight_a.detach().clone(),layer.weight_b.detach().clone()*.9)
        with tempfile.TemporaryDirectory() as tmp:
            args=make_args(tmp,ds)
            proposal,selection,_=load_rsc_samples(ds,default_data_collator,args)
            snapshot=self.snapshot(model,opt,sched)
            rows,candidates=[],[]
            with patch('adaptive_lora.adaptation_v2.plan_layer',side_effect=plan):
                qualified,values,joint,timings=compute_rsc_candidates(
                    model,{'proj':model.proj,'other':model.other},proposal,selection,args,rows,candidates)
            self.assertEqual(len(qualified),2)
            self.assertIsNotNone(values)
            self.assertEqual(joint['reason'],'accepted')
            self.assertEqual(set(timings),{'candidate_generation','gradient_gating','joint_loss_search'})
            # Existing optimizer/scheduler are ONLY witnesses; computation never receives them.
            self.assert_preserved(model,opt,sched,snapshot)
