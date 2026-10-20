"""CPU-only tests; no real checkpoints or training runs."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
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
    candidate_check, extract_historical_candidates, bind_historical_candidates, preserved_state, main, verified)
from adaptive_lora.data_provenance import file_sha256
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
        with self.assertRaisesRegex(ValueError, 'violates rank/trust'):
            audit_candidates(self.model, [c], self.data, self.groups, .05)

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

    def payloads(self):
        spec = dict(structure='dense', config=asdict(self.model.layer.lora_config),
                    scaling=self.model.layer.lora_scaler,
                    state={k: getattr(self.model.layer, k).detach().clone() for k in ('weight_a','weight_b')})
        pre = dict(format_version=1, base_model='/toy/base', layers={'layer': spec})
        post = copy.deepcopy(pre)
        post['layers']['layer']['state']['weight_b'].mul_(.99)
        history = dict(step=1600, invariants_passed=True, num_modules=1,
                       actual_changed_sides={'A':0, 'B':1, 'none':0})
        return pre, post, history

    def test_historical_candidates_integration(self):
        pre, post, history = self.payloads()
        before = self.snapshot()
        records, rows, counts = extract_historical_candidates(pre, post, .05, history)
        candidates = bind_historical_candidates(self.model, pre, records)
        self.assertEqual(counts, history['actual_changed_sides'])
        self.assertEqual(rows[0]['side'], 'B')
        self.assertTrue(torch.equal(candidates[0]['target'], post['layers']['layer']['state']['weight_b']))
        result = audit_candidates(self.model, candidates, self.data, self.groups, .05)
        self.assertEqual(result['policies']['all_candidates']['num_updated_modules'], 1)
        for n,p in self.model.named_parameters(): self.assertTrue(torch.equal(p,before[n]))

    def test_extract_mixed_a_b_none(self):
        pre, post, history = self.payloads()
        pre['layers']['other'] = copy.deepcopy(pre['layers']['layer'])
        pre['layers']['unchanged'] = copy.deepcopy(pre['layers']['layer'])
        post['layers']['other'] = copy.deepcopy(pre['layers']['other'])
        post['layers']['other']['state']['weight_a'].mul_(.99)
        post['layers']['unchanged'] = copy.deepcopy(pre['layers']['unchanged'])
        history.update(num_modules=3, actual_changed_sides={'A':1,'B':1,'none':1})
        records, rows, counts = extract_historical_candidates(pre, post, .05, history)
        self.assertEqual(counts, {'A':1,'B':1,'none':1})
        self.assertEqual(len(records), 2)
        self.assertEqual([r['side'] for r in rows], ['B','A','none'])

    def test_reject_both_factors_changed(self):
        pre, post, history = self.payloads()
        post['layers']['layer']['state']['weight_a'].mul_(.99)
        with self.assertRaisesRegex(ValueError, 'Both factors changed'):
            extract_historical_candidates(pre, post, .05, history)

    def test_checkpoint_metadata_mismatches(self):
        for kind in ('base','targets','rank','scaling','shape','dtype','missing','nonfinite','format'):
            with self.subTest(kind=kind):
                pre, post, history = self.payloads()
                spec = post['layers']['layer']
                if kind == 'base': post['base_model'] = '/other'
                elif kind == 'targets': post['layers']['other'] = post['layers'].pop('layer')
                elif kind == 'rank': spec['config']['lora_rank'] += 1
                elif kind == 'scaling': spec['scaling'] += 1
                elif kind == 'shape': spec['state']['weight_a'] = torch.ones(1,4)
                elif kind == 'dtype': spec['state']['weight_a'] = spec['state']['weight_a'].double()
                elif kind == 'missing': del spec['state']['weight_a']
                elif kind == 'nonfinite': spec['state']['weight_b'][0] = float('nan')
                elif kind == 'format': post['format_version'] = 2
                with self.assertRaises(ValueError): extract_historical_candidates(pre, post, .05, history)

    def test_historical_count_mismatch_aborts(self):
        pre, post, history = self.payloads()
        history['actual_changed_sides'] = {'A':1,'B':0,'none':0}
        with self.assertRaisesRegex(ValueError, 'actual=.*expected='):
            extract_historical_candidates(pre, post, .05, history)

    def test_historical_trust_failure_aborts(self):
        pre, post, history = self.payloads()
        post['layers']['layer']['state']['weight_b'].mul_(.5)
        with self.assertRaisesRegex(ValueError, 'violates rank/trust'):
            extract_historical_candidates(pre, post, .05, history)

    def test_historical_rank_failure_aborts(self):
        pre, post, history = self.payloads()
        post['layers']['layer']['state']['weight_b'].zero_()
        with self.assertRaisesRegex(ValueError, 'violates rank/trust'):
            extract_historical_candidates(pre, post, .05, history)

    def test_loaded_pre_mismatch_aborts(self):
        pre, post, history = self.payloads()
        records, _, _ = extract_historical_candidates(pre, post, .05, history)
        with torch.no_grad(): self.model.layer.weight_a.add_(1)
        with self.assertRaisesRegex(ValueError, 'Loaded PRE factor mismatch'):
            bind_historical_candidates(self.model, pre, records)

    def test_hash_mismatch_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'post.pt'; path.write_bytes(b'toy')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                verified(path, '0'*64)

    def test_no_candidate_regeneration_imports(self):
        import inspect
        from adaptive_lora import audit_cross_batch_acceptance as module
        source = inspect.getsource(module)
        for forbidden in ('probe_weight', 'svd_with_gid', 'plan_geometric_direction_correction', 'generate_candidates'):
            self.assertNotIn(forbidden, source)

    def test_model_load_failure_does_not_reserve_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pre, post, history = self.payloads()
            pre_path, post_path = root/'pre.pt', root/'post.pt'
            torch.save(pre, pre_path); torch.save(post, post_path)
            data = root/'train.json'
            data.write_text(json.dumps([{'text': str(i)} for i in range(48)]))
            calibration = root/'calibration.json'
            calibration.write_text(json.dumps(dict(split='train', dataset_source=str(data),
                batch_size=1, batch_indices=[[i] for i in range(8)], dataset_length=48)))
            history.update(checkpoint_pre_sha256=file_sha256(pre_path),
                           checkpoint_post_sha256=file_sha256(post_path))
            audit = root/'audit.json'; audit.write_text(json.dumps(history))
            cfg = dict(model_family='minicpm5', lora_structure='dense', batch_size=1,
                model_name_or_path='/toy/base', train_data_path=str(data),
                calibration_indices_path=str(calibration), train_data_sha256=file_sha256(data),
                calibration_indices_sha256=file_sha256(calibration), train_samples=48,
                targets=['layer'], direction_function_tolerance=.05, seed=42,
                model_max_length=2048, dynamic_length=True, dtype='float32')
            config = root/'config.json'; config.write_text(json.dumps(cfg))
            output = root/'output'
            argv = ['audit', '--run_config',str(config),'--pre_adapter',str(pre_path),
                    '--post_adapter',str(post_path),'--historical_audit',str(audit),
                    '--expected_adapter_sha256',file_sha256(pre_path),'--output_dir',str(output),
                    '--device','cpu']
            with patch('sys.argv', argv), patch('transformers.AutoTokenizer.from_pretrained') as tok, \
                    patch('adaptive_lora.training_data.ChatTrainingDataset'), \
                    patch('adaptive_lora.model_loading.load_training_model', side_effect=RuntimeError('load failure')):
                tok.return_value.chat_template = 'toy'
                with self.assertRaisesRegex(RuntimeError, 'load failure'): main()
            self.assertFalse(output.exists())


if __name__ == '__main__': unittest.main()
