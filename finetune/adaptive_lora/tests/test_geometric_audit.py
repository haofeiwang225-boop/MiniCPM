import ast
import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from adaptive_lora.gradient_probe import evaluate_calibration_loss
from adaptive_lora.diagnostics import write_geometric_diagnostics
from adaptive_lora.adaptation import geometric_correction_step


class LossToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(3,2)
        self.dropout=nn.Dropout(.7)
    def forward(self,x,labels):
        # Consume RNG even in eval to test fork_rng explicitly.
        ignored=torch.rand(())*0
        return SimpleNamespace(loss=(self.dropout(self.linear(x))-labels).square().mean()+ignored)


class AuditTests(unittest.TestCase):
    def test_loss_helper_preserves_all_training_state(self):
        torch.manual_seed(19)
        model=LossToy();model.train();model.dropout.eval() # mixed module modes
        opt=torch.optim.AdamW(model.parameters());sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1.)
        batches=[dict(x=torch.randn(2,3),labels=torch.randn(2,2)) for _ in range(2)]
        model(**batches[0]).loss.backward();opt.step();sched.step()
        weights=copy.deepcopy(model.state_dict());states=copy.deepcopy(opt.state_dict())
        modes=[m.training for m in model.modules()];flags=[p.requires_grad for p in model.parameters()]
        grads=[p.grad.clone() for p in model.parameters()];scheduler=copy.deepcopy(sched.state_dict())
        rng=torch.get_rng_state().clone()
        losses,mean=evaluate_calibration_loss(model,batches)
        self.assertEqual(len(losses),2);self.assertAlmostEqual(mean,sum(losses)/2)
        self.assertTrue(torch.isfinite(torch.tensor(losses)).all())
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(modes,[m.training for m in model.modules()])
        self.assertEqual(flags,[p.requires_grad for p in model.parameters()])
        for p,g in zip(model.parameters(),grads):self.assertTrue(torch.equal(p.grad,g))
        for key,v in weights.items():self.assertTrue(torch.equal(model.state_dict()[key],v))
        for key,state in states['state'].items():
            for k,v in state.items():self.assertTrue(torch.equal(opt.state_dict()['state'][key][k],v))
        self.assertEqual(opt.state_dict()['param_groups'],states['param_groups'])
        self.assertEqual(sched.state_dict(),scheduler)

    def test_loss_helper_exception_restores_modes_and_rng(self):
        model=LossToy();model.dropout.eval()
        modes=[m.training for m in model.modules()];rng=torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
            evaluate_calibration_loss(model,[dict(x=torch.ones(2,3),labels=torch.full((2,2),float('nan')))])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(modes,[m.training for m in model.modules()])

    def summary(self,norms):
        rows=[]
        for norm,before,after in zip(norms,(.2,.8),(.3,.9)):
            rows.append(dict(gradient_fro_norm=norm,tangent_coverage_before=before,
                tangent_coverage_after=after,applied_side='none',correction_skip_reason='test',
                optimizer_state_reset_count=0,applied_relative_update_change=0.,base_weight_unchanged=True))
        with tempfile.TemporaryDirectory() as tmp:
            return write_geometric_diagnostics(Path(tmp)/'stats',rows,{})

    def test_weighted_coverage_not_simple_mean(self):
        s=self.summary([10.,1.])
        self.assertEqual(s['global_gradient_energy'],101)
        self.assertAlmostEqual(s['global_tangent_energy_before'],20.8)
        self.assertAlmostEqual(s['global_tangent_energy_after'],30.9)
        self.assertAlmostEqual(s['global_tangent_coverage_before'],20.8/101)
        self.assertAlmostEqual(s['global_tangent_coverage_after'],30.9/101)
        self.assertAlmostEqual(s['global_tangent_coverage_gain'],.1)
        self.assertAlmostEqual(s['mean_tangent_coverage_before'],.5)
        self.assertGreater(abs(s['mean_tangent_coverage_before']-s['global_tangent_coverage_before']),.2)

    def test_zero_gradient_weighted_convention(self):
        s=self.summary([0.,0.])
        self.assertEqual(s['global_gradient_energy'],0)
        self.assertEqual(s['global_tangent_energy_before'],0)
        self.assertEqual(s['global_tangent_energy_after'],0)
        self.assertEqual(s['global_tangent_coverage_before'],1)
        self.assertEqual(s['global_tangent_coverage_after'],1)
        self.assertEqual(s['global_tangent_coverage_gain'],0)

    def test_geometric_invariants_not_python_asserts(self):
        tree=ast.parse(inspect.getsource(geometric_correction_step))
        self.assertFalse(any(isinstance(n,ast.Assert) for n in ast.walk(tree)))
