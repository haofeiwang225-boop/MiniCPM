"""Exercise experiment bookkeeping with simulated generation, never model weights."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import GenerationConfig
import yaml

from benchmarks.math500 import run_experiment as runner
from benchmarks.math500.compare_experiments import compare


class ExperimentTests(unittest.TestCase):
    def test_variable_isolation(self):
        root = Path(runner.__file__).with_name('experiments')
        configs = {p.stem: yaml.safe_load(p.read_text()) for p in root.glob('*.yaml')}
        base = configs['sampling_16384']
        changes = {'baseline_8192': {'max_new_tokens'}, 'sampling_16384': set(),
                   'penalty_16384': {'repetition_penalty'}, 'prompt_16384': set(),
                   'greedy_16384': {'do_sample'}}
        for name, config in configs.items():
            self.assertEqual(config['question_ids'], base['question_ids'])
            self.assertEqual({k for k in config['generation'] if config['generation'][k] != base['generation'][k]}, changes[name])
            self.assertEqual(bool(config['prompt_suffix']), name == 'prompt_16384')

    def test_runner_resume_and_comparison(self):
        class Inputs(dict):
            def to(self, device):
                return self

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                assert kwargs == dict(tokenize=True, add_generation_prompt=True, enable_thinking=True,
                                      return_dict=True, return_tensors='pt')
                return Inputs(input_ids=torch.tensor([[10, 11]]))

            def decode(self, tokens, **kwargs):
                if tokens.tolist() == [10, 11]:
                    return 'rendered prompt'
                return '</think>\\boxed{4}<|im_end|>'

        class Model:
            device = torch.device('cpu')
            dtype = torch.float32
            config = SimpleNamespace(max_position_embeddings=131072)
            generation_config = GenerationConfig(eos_token_id=[1, 130073])

            def eval(self):
                return self

            def generate(self, **kwargs):
                assert kwargs['generation_config'].max_new_tokens == 16384
                return torch.tensor([[10, 11, 12, 1]])

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = yaml.safe_load(Path(runner.__file__).with_name('experiments').joinpath('sampling_16384.yaml').read_text())
            config.update(question_ids=[9, 59], previously_truncated_ids=[9], previously_correct_ids=[59], output_dir=str(root/'run'))
            config_path = root/'config.yaml'
            config_path.write_text(yaml.safe_dump(config))
            samples = [{'question_id': qid, 'problem': 'Compute two plus two.', 'answer': '4'} for qid in [9, 59]]
            command = ['run_experiment.py', '--config', str(config_path)]
            with patch.object(sys, 'argv', command), patch.object(runner, 'load_samples', return_value=samples), \
                    patch.object(runner.AutoTokenizer, 'from_pretrained', return_value=Tokenizer()), \
                    patch.object(runner.AutoModelForCausalLM, 'from_pretrained', return_value=Model()), \
                    patch.object(runner, 'set_seed') as seed, contextlib.redirect_stdout(io.StringIO()):
                runner.main()
                self.assertEqual([call.args[0] for call in seed.call_args_list], [51, 101])
            result = compare([root/'run'])
            summary = result['experiments'][0]['summary']
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['strata']['previously_truncated']['ended_and_correct'], 1)
            self.assertEqual(summary['strata']['previously_correct']['correct'], 1)
            original = (root/'run/results.jsonl').read_bytes()
            with patch.object(sys, 'argv', command + ['--resume']), patch.object(runner, 'load_samples', return_value=samples), \
                    patch.object(runner.AutoModelForCausalLM, 'from_pretrained') as load, contextlib.redirect_stdout(io.StringIO()):
                runner.main()
                load.assert_not_called()
            self.assertEqual((root/'run/results.jsonl').read_bytes(), original)
            summary_path = root/'run/summary.json'
            summary['complete'] = False
            summary_path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                compare([root/'run'])


if __name__ == '__main__':
    unittest.main()
