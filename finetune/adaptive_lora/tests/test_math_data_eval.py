import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM, default_data_collator
from adaptive_lora.training_data import ChatTrainingDataset
from adaptive_lora.gradient_probe import probe_weight
from adaptive_lora.dense_init import attach_dense, dense_first_probe
from adaptive_lora.checkpoint import save_adapter
from adaptive_lora.calibration import write_calibration_indices, load_calibration_batches
from adaptive_lora.data_provenance import file_sha256, git_version, save_source_metadata
from evaluate_adapter_loss import evaluate_dataset, load_evaluation_model
from test_training_data import CharacterTokenizer
from test_dense_init import arguments


def tiny_model():
    return LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, use_cache=False))


def write_data(path):
    rows = [dict(messages=[dict(role='user', content=f'{i}+1=?'),
                            dict(role='assistant', content='reason. '*i + f'Answer: {i+1}')])
            for i in range(8)]
    path.write_text(json.dumps(rows))
    return rows


def run_dynamic_smoke(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(19)
    path = out/'synthetic_train.json'
    write_data(path)
    dataset = ChatTrainingDataset(path, CharacterTokenizer(), 2048, dynamic_length=True)
    args = arguments(out)
    args.dense_lora_param_budget = 128
    args.calibration_indices_path = out/'calibration_indices.json'
    args.calibration_indices_sha256 = write_calibration_indices(
        args.calibration_indices_path, dataset, 1, 8, path.resolve())
    batches = load_calibration_batches(args.calibration_indices_path,
                                      args.calibration_indices_sha256, dataset, default_data_collator)
    model = tiny_model()
    names = attach_dense(model, args)
    dense_first_probe(model, names, batches, args)
    base = {name: model.get_submodule(name).weight.detach().clone() for name in names}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0, 1-step/5))
    losses = []
    for step in range(5):
        model.train()
        loss = model(**batches[step]).loss
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite smoke loss')
        loss.backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    for name in names:
        gradient = probe_weight(model, model.get_submodule(name), batches)
        if not torch.isfinite(gradient).all() or not torch.equal(base[name], model.get_submodule(name).weight):
            raise RuntimeError('Smoke probe invariant failed')
    save_adapter(model, names, out/'adapter.pt', 'synthetic_tiny_model')
    result = dict(steps=5, losses=losses, dynamic_length=True, max_length=2048,
                  sequence_lengths=[int(x['input_ids'].shape[-1]) for x in batches],
                  supervised_tokens=dataset.supervised_tokens, calibration_batches=8,
                  base_weights_unchanged=True, scheduler_last_epoch=scheduler.last_epoch,
                  train_data_sha256=file_sha256(path), **git_version())
    (out/'summary.json').write_text(json.dumps(result, indent=2))
    return result


class MathDataEvalTests(unittest.TestCase):
    def test_dynamic_static_labels_and_weighted_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.json'; write_data(path)
            dynamic = ChatTrainingDataset(path, CharacterTokenizer(), 256, dynamic_length=True)
            static = ChatTrainingDataset(path, CharacterTokenizer(), 256)
            self.assertEqual(dynamic.supervised_tokens, static.supervised_tokens)
            for a, b in zip(dynamic, static):
                valid = b['attention_mask'].bool()
                for key in a:
                    self.assertTrue(torch.equal(a[key], b[key][valid]))
                self.assertEqual(default_data_collator([a])['input_ids'].shape[0], 1)
            model = tiny_model(); model.train()
            before = copy.deepcopy(model.state_dict())
            with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('No optimizer step')):
                actual, rows = evaluate_dataset(model, dynamic, 'cpu')
                padded, _ = evaluate_dataset(model, static, 'cpu')
            self.assertAlmostEqual(actual['token_weighted_nll'], padded['token_weighted_nll'], places=5)
            self.assertEqual(actual['effective_tokens'], sum(dynamic.supervised_tokens))
            expected = sum(row['nll_sum'] for row in rows)/sum(dynamic.supervised_tokens)
            self.assertAlmostEqual(actual['token_weighted_nll'], expected)
            self.assertAlmostEqual(actual['ppl'], math.exp(expected))
            self.assertTrue(model.training)
            self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()))

    def test_empty_answer_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.json'
            rows = write_data(path)
            rows[0]['messages'][-1]['content'] = '  '
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'Empty assistant answer'):
                ChatTrainingDataset(path, CharacterTokenizer(), dynamic_length=True)

    def test_dynamic_cli_defaults_and_batch_constraint(self):
        from train_adaptive_lora import get_args
        base = ['train', '--output_dir', '/tmp/math_cli', '--model_family', 'minicpm5']
        with patch('sys.argv', base):
            args = get_args()
            self.assertFalse(args.dynamic_length)
            self.assertEqual(args.model_max_length, 256)
        with patch('sys.argv', base + ['--dynamic_length', 'true']):
            self.assertEqual(get_args().model_max_length, 2048)
        with patch('sys.argv', base + ['--dynamic_length', 'true', '--batch_size', '2']), \
             patch('sys.stderr'), self.assertRaises(SystemExit):
            get_args()

    def test_partial_answer_truncation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.json'; write_data(path)
            for dynamic in (False, True):
                with self.assertRaisesRegex(ValueError, 'Complete assistant answer would be truncated'):
                    ChatTrainingDataset(path, CharacterTokenizer(), 30, dynamic_length=dynamic)

    def test_load_adapter_uses_existing_loader_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = tiny_model(); original = copy.deepcopy(base)
            args = arguments(directory); names = attach_dense(base, args)
            with torch.no_grad():
                for name in names:
                    base.get_submodule(name).weight_b.fill_(.2)
            path = Path(directory)/'adapter.pt'; save_adapter(base, names, path, 'toy')
            with patch('evaluate_adapter_loss.load_training_model', return_value=original) as loader:
                restored = load_evaluation_model('toy', torch.float32, 'minicpm5', 'cpu', path)
            loader.assert_called_once_with('toy', torch.float32, 'minicpm5')
            inputs = torch.tensor([[1, 2, 3]])
            base.eval(); restored.eval()
            torch.testing.assert_close(base(inputs).logits, restored(inputs).logits, rtol=0, atol=0)

    def test_five_step_dynamic_training_initial_and_periodic_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_dynamic_smoke(Path(directory)/'smoke')
            self.assertEqual(result['steps'], 5)
            self.assertEqual(result['scheduler_last_epoch'], 5)
            self.assertGreater(len(set(result['sequence_lengths'])), 1)

    def test_source_sidecar_keeps_split_bytes_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'dev.json'; rows = write_data(path)
            source = Path(directory)/'source.jsonl'
            source.write_text('\n'.join(json.dumps(dict(query=x['messages'][0]['content'],
                response=x['messages'][1]['content'], type='GSM_AnsAug', original_question=str(i)))
                for i, x in enumerate(rows)))
            before = path.read_bytes()
            meta = save_source_metadata(path, source, Path(directory)/'dev.metadata.json')
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([x['index'] for x in meta['samples']], list(range(8)))
            self.assertTrue(all(x['family']=='GSM' for x in meta['samples']))
