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
