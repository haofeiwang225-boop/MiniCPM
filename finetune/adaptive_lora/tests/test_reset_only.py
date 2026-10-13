import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from adaptive_lora.reset_only import reset_only_step
from test_geometric_direction_correction import dense, example


class ResetOnlyTests(unittest.TestCase):
    def test_only_B_state_removed_and_repeat_events(self):
        _, a, b = example()
        model = nn.ModuleDict({'first': dense(a, b), 'second': dense(a, b)})
        extra = nn.Parameter(torch.ones(2))
        model.register_parameter('extra', extra)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        with tempfile.TemporaryDirectory() as directory:
            for step in (2, 10):
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad = torch.ones_like(p)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                states = {p: (s, copy.deepcopy(s)) for p, s in optimizer.state.items()}
                schedule = copy.deepcopy(scheduler.state_dict())
                rng = torch.get_rng_state().clone()
                with patch.object(model, 'forward', side_effect=AssertionError('No forward allowed')), \
                     patch('torch.linalg.svd', side_effect=AssertionError('No SVD allowed')):
                    result = reset_only_step(model, ['first', 'second'], optimizer, scheduler, step, directory)
                self.assertEqual(result['optimizer_state_reset_count'], 2)
                self.assertEqual(result['old_total_lora_params'], result['new_total_lora_params'])
                self.assertEqual(schedule, scheduler.state_dict())
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for name, p in model.named_parameters():
                    self.assertTrue(torch.equal(before[name], p))
                    if name.endswith('weight_b'):
                        self.assertNotIn(p, optimizer.state)
                    elif p in states:
                        original, values = states[p]
                        self.assertIs(original, optimizer.state[p])
                        for key, value in values.items():
                            torch.testing.assert_close(optimizer.state[p][key], value, rtol=0, atol=0)
                saved = json.loads((Path(directory)/'reset_only'/f'step_{step}'/'summary.json').read_text())
                self.assertEqual(saved, result)
            # No existing B states is safe and counted as zero actual removals.
            result = reset_only_step(model, ['first', 'second'], optimizer, scheduler, 11, directory)
            self.assertEqual(result['optimizer_state_reset_count'], 0)

    def test_cli_constraints(self):
        from train_adaptive_lora import get_args
        base = ['train', '--output_dir', '/tmp/reset_cli', '--direction_correction', 'reset_only',
                '--lora_structure', 'dense', '--adaptive_lora', 'false', '--max_steps', '12']
        with patch('sys.argv', base + ['--direction_correction_steps', '2', '10']):
            self.assertEqual(get_args().direction_correction, 'reset_only')
        invalid = [[], ['--direction_correction_steps', '2', '10', '--adaptive_lora', 'true'],
                   ['--direction_correction_steps', '2', '10', '--lora_structure', 'block'],
                   ['--direction_correction_steps', '2', '10', '--diagnostic_steps', '2']]
        for args in invalid:
            with patch('sys.argv', base + args), patch('sys.stderr'), self.assertRaises(SystemExit):
                get_args()

    def test_mixed_manifest_and_none_preserve_unselected_states(self):
        import hashlib
        _, a, b = example()
        model = nn.ModuleDict({name: dense(a, b) for name in ('first', 'second', 'third')})
        model.register_parameter('extra', nn.Parameter(torch.ones(2)))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        values = {p: p.detach().clone() for p in model.parameters()}
        states = {p: (state, copy.deepcopy(state), {k: (id(v), v._version)
                  for k, v in state.items()}) for p, state in optimizer.state.items()}
        rng = torch.get_rng_state().clone()
        schedule = copy.deepcopy(scheduler.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            path.write_text(json.dumps(dict(step=2, num_modules=3,
                                           sides={'first': 'A', 'second': 'B', 'third': 'none'})))
            result = reset_only_step(model, ['first', 'second', 'third'], optimizer, scheduler,
                                     2, directory, side_manifest=path)
            self.assertEqual(result['reset_mode'], 'matched_manifest')
            self.assertEqual(result['reset_sides'], {'A': 1, 'B': 1, 'none': 1})
            self.assertEqual(result['optimizer_state_reset_count'], 2)
            self.assertEqual(result['manifest_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        selected = {model['first'].weight_a, model['second'].weight_b}
        self.assertEqual(set(optimizer.state), set(states) - selected)
        for p, value in values.items():
            self.assertTrue(torch.equal(p, value))
        for p, (state, snapshot, versions) in states.items():
            if p in selected:
                continue
            self.assertIs(optimizer.state[p], state)
            for key, value in state.items():
                self.assertEqual((id(value), value._version), versions[key])
                self.assertTrue(torch.equal(value, snapshot[key]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(schedule, scheduler.state_dict())

    def test_invalid_manifests_fail_before_reset(self):
        _, a, b = example()
        model = nn.ModuleDict({'first': dense(a, b)})
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        good = dict(step=2, num_modules=1, sides={'first': 'A'})
        invalid = [dict(good, step=3), dict(good, num_modules=2),
                   dict(good, sides={}), dict(good, sides={'first': 'A', 'extra': 'B'}),
                   dict(good, sides={'first': 'both'}), dict(good, sides={'first': []})]
        raw = [json.dumps(item) for item in invalid]
        raw.append('{"step":2,"num_modules":1,"sides":{"first":"A","first":"B"}}')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            for text in raw:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    reset_only_step(model, ['first'], optimizer, scheduler, 2, directory, path)
                self.assertFalse((Path(directory)/'reset_only').exists())

    def test_manifest_cli_constraints(self):
        from train_adaptive_lora import get_args
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            path.write_text('{}')
            base = ['train', '--output_dir', directory, '--reset_side_manifest', str(path),
                    '--lora_structure', 'dense', '--adaptive_lora', 'false', '--max_steps', '12']
            with patch('sys.argv', base + ['--direction_correction', 'reset_only',
                                          '--direction_correction_steps', '2']):
                self.assertEqual(get_args().reset_side_manifest, path)
            for mode in ('geometric', 'none', 'blend'):
                args = base + ['--direction_correction', mode]
                if mode == 'geometric':
                    args += ['--direction_correction_steps', '2']
                with patch('sys.argv', args), patch('sys.stderr'), self.assertRaises(SystemExit):
                    get_args()
            with patch('sys.argv', base + ['--direction_correction', 'reset_only',
                       '--direction_correction_steps', '2', '--reset_side_manifest', str(path)+'missing']), \
                 patch('sys.stderr'), self.assertRaises(SystemExit):
                get_args()
