"""CPU-only tests; no real checkpoints or training runs."""
import copy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from transformers import default_data_collator
from adaptive_lora.audit_cross_batch_acceptance import (
    audit_candidates, validate_groups, directional_slopes, temporary_candidates,
    candidate_check, generate_candidates, preserved_state)
from adaptive_lora.dense_init import make_dense_layer
from adaptive_lora.official import LoRAConfig
from evaluate_adapter_loss import evaluate_dataset


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = make_dense_layer(nn.Linear(3, 3, bias=False),
            LoRAConfig(3, 3, lora_rank=1, lora_scaler=1., run_lora_in_fp32=True))
        with torch.no_grad():
            self.layer.weight.zero_()
            self.layer.weight_a.copy_(torch.tensor([[1., 0., 0.]]))
            self.layer.weight_b.copy_(torch.tensor([[1.], [0.], [0.]]))
        self.layer.weight.requires_grad_(False)
        self.fail = False

    def forward(self, input_ids, labels, **kw):
        # A causal token-mean test loss with exactly the evaluator's shifted mask.
        x = torch.nn.functional.one_hot(input_ids[:, :-1], 3).float()
        output = self.layer(x)[..., 0]
        mask = labels[:, 1:] != -100
        loss = (output[mask] - labels[:, 1:][mask].float()).square().mean()
        if self.fail:
            raise RuntimeError('injected forward failure')
        return SimpleNamespace(loss=loss)


def sample(target=0, tokens=1):
    return dict(input_ids=torch.zeros(tokens+1, dtype=torch.long),
                labels=torch.tensor([-100]+[target]*tokens))


def candidate(model, side='B'):
    p = model.layer.weight_b if side == 'B' else model.layer.weight_a
    return dict(module='layer', layer=model.layer, parameter=p, side=side, target=p.detach().clone()*.99)


class CrossBatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = Tiny()
        self.groups = dict(proposal=[0], selector=[1], audit=[2, 3])
        self.data = [sample(), sample(), sample(), sample(2, 3)]

    def test_both_negative_selected(self):
        r = audit_candidates(self.model, [candidate(self.model)], self.data, self.groups, .05)
        self.assertTrue(r['candidates'][0]['selected'])
        self.assertEqual(r['policies']['filtered']['num_updated_modules'], 1)
        self.assertFalse(r['audit_used_for_selection'])

    def test_selector_positive_rejected(self):
        self.data[1] = sample(2)
        r = audit_candidates(self.model, [candidate(self.model)], self.data, self.groups, .05)
        self.assertLess(r['candidates'][0]['proposal']['mean'], 0)
        self.assertGreater(r['candidates'][0]['selector']['mean'], 0)
        self.assertFalse(r['candidates'][0]['selected'])
        self.assertEqual(r['policies']['original']['token_weighted_nll'],
                         r['policies']['filtered']['token_weighted_nll'])

    def test_zero_slope_not_selected(self):
        c = candidate(self.model); c['target'].copy_(c['parameter'])
        r = audit_candidates(self.model, [c], self.data, self.groups, .05)
        self.assertFalse(r['candidates'][0]['selected'])

    def test_trust_violation_rejected(self):
        c = candidate(self.model); c['target'].mul_(.1)
        self.assertFalse(candidate_check(c, .05)['feasible'])
        r = audit_candidates(self.model, [c], self.data, self.groups, .05)
        self.assertEqual(r['policies']['all_candidates']['num_updated_modules'], 0)

    def test_indices_and_content_isolation(self):
        validate_groups(self.groups, 4, ['a', 'b', 'c', 'd'])
        for group in (dict(proposal=[0], selector=[0], audit=[2]),
                      dict(proposal=[0], selector=[1], audit=[2, 2]),
                      dict(proposal=[0], selector=[1], audit=[4])):
            with self.assertRaises(ValueError): validate_groups(group, 4)
        with self.assertRaisesRegex(ValueError, 'Identical'):
            validate_groups(self.groups, 4, ['a', 'b', 'a', 'd'])

    def test_token_weighted_not_sample_mean(self):
        dataset = [sample(0, 1), sample(3, 3)]
        result, rows = evaluate_dataset(self.model, dataset, 'cpu')
        self.assertEqual(result['effective_tokens'], 4)
        self.assertAlmostEqual(result['token_weighted_nll'], (1+4*3)/4)
        self.assertNotEqual(result['token_weighted_nll'], (1+4)/2)
        self.assertEqual([r['effective_tokens'] for r in rows], [1, 3])

    def test_a_factor_derivative_and_only_a_trial(self):
        c = candidate(self.model, 'A')
        b = self.model.layer.weight_b.clone()
        slopes = directional_slopes(self.model, [c], [default_data_collator([sample()])])
        self.assertLess(slopes['layer']['mean'], 0)
        with temporary_candidates([c]):
            self.assertTrue(torch.equal(self.model.layer.weight_b, b))
            self.assertTrue(torch.equal(c['parameter'], c['target']))

    def snapshot(self):
        return {n: p.detach().clone() for n, p in self.model.named_parameters()}

    def test_exact_restore_and_optimizer_scheduler_rng_untouched(self):
        model = self.model
        opt = torch.optim.AdamW(model.parameters())
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
        # Seed real Adam-shaped state without any training/update.
        for p in model.parameters():
            p.grad = torch.ones_like(p)
            opt.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.ones_like(p), exp_avg_sq=torch.ones_like(p))
        state = {p: (s, {k: (v, v.clone(), v._version) for k,v in s.items()}) for p,s in opt.state.items()}
        values = self.snapshot(); params = dict(model.named_parameters())
        flags = [p.requires_grad for p in model.parameters()]
        grads = [p.grad for p in model.parameters()]
        modes = [m.training for m in model.modules()]
        rng = torch.get_rng_state().clone(); py = random.getstate(); npstate = np.random.get_state()
        scheduler = copy.deepcopy(sched.state_dict())
        rank, scaling = model.layer.lora_rank, model.layer.lora_scaler
        count = sum(p.numel() for p in model.parameters())
        groups = copy.deepcopy(opt.state_dict()['param_groups'])
        audit_candidates(model, [candidate(model)], self.data, self.groups, .05)
        self.assertEqual((rank, scaling), (model.layer.lora_rank, model.layer.lora_scaler))
        self.assertEqual(count, sum(p.numel() for p in model.parameters()))
        self.assertEqual(groups, opt.state_dict()['param_groups'])
        for n,p in model.named_parameters():
            self.assertIs(p, params[n]); self.assertTrue(torch.equal(p, values[n]))
        self.assertEqual(flags, [p.requires_grad for p in model.parameters()])
        for p,g in zip(model.parameters(), grads): self.assertIs(p.grad, g)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state())); self.assertEqual(py, random.getstate())
        np.testing.assert_equal(npstate, np.random.get_state())
        self.assertEqual(scheduler, sched.state_dict())
        for p,(s, vs) in state.items():
            self.assertIs(opt.state[p], s)
            for k,(v, old, version) in vs.items():
                self.assertIs(s[k], v); self.assertTrue(torch.equal(v,old)); self.assertEqual(v._version, version)

    def test_forward_exception_restores_factors_and_state(self):
        before = self.snapshot(); rng = torch.get_rng_state().clone()
        self.model.fail = True
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with preserved_state(self.model), temporary_candidates([candidate(self.model)]):
                evaluate_dataset(self.model, self.data, 'cpu')
        for n,p in self.model.named_parameters(): self.assertTrue(torch.equal(p, before[n]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(self.model.training)

    def test_scoring_exception_rolls_back_all_policy(self):
        before = self.snapshot()
        actual = evaluate_dataset
        calls = []
        def score(*args):
            calls.append(1)
            if len(calls) == 2: raise RuntimeError('policy failure')
            return actual(*args)
        with patch('adaptive_lora.audit_cross_batch_acceptance.evaluate_dataset', side_effect=score):
            with self.assertRaisesRegex(RuntimeError, 'policy failure'):
                audit_candidates(self.model, [candidate(self.model)], self.data, self.groups, .05)
        for n,p in self.model.named_parameters(): self.assertTrue(torch.equal(p, before[n]))

    def test_empty_candidates_noop(self):
        r = audit_candidates(self.model, [], self.data, self.groups, .05)
        self.assertEqual(r['policies']['original'], r['policies']['filtered'])

    def test_audit_values_never_change_selection(self):
        original = audit_candidates(self.model, [candidate(self.model)], self.data, self.groups, .05)
        self.data[2:] = [sample(17), sample(23, 3)]
        changed = audit_candidates(self.model, [candidate(self.model)], self.data, self.groups, .05)
        self.assertEqual(original['candidates'], changed['candidates'])
        self.assertNotEqual(original['policies']['original']['token_weighted_nll'],
                            changed['policies']['original']['token_weighted_nll'])

    def test_rank_deficient_candidate_rejected(self):
        c = candidate(self.model); c['target'].zero_()
        self.assertFalse(candidate_check(c, 1.)['feasible'])

    def test_slope_exception_restores_grad_flags(self):
        flags = [p.requires_grad for p in self.model.parameters()]
        self.model.fail = True
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            directional_slopes(self.model, [candidate(self.model)],
                               [default_data_collator([sample()])])
        self.assertEqual(flags, [p.requires_grad for p in self.model.parameters()])

    def test_v1_probe_and_planner_integration(self):
        args = SimpleNamespace(analysis_device='cpu', adaptive_residual_energy_threshold=.1,
            direction_function_tolerance=.05, direction_min_gain=.001,
            direction_eta_start=1., direction_eta_decay=.5, direction_eta_min=.03125)
        before = self.snapshot()
        with preserved_state(self.model):
            candidates, rows = generate_candidates(self.model, ['layer'],
                [default_data_collator([sample()])], args)
        self.assertEqual(len(rows), 1)
        self.assertTrue(math_isfinite(rows[0]['gid_full']))
        for n,p in self.model.named_parameters(): self.assertTrue(torch.equal(p,before[n]))


def math_isfinite(value):
    return bool(torch.isfinite(torch.tensor(value)))


if __name__ == '__main__': unittest.main()
