import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from adaptive_lora.dense_init import allocate_dense_ranks, attach_dense, dense_first_probe
from adaptive_lora.official import LinearWithLoRA, dense_factors
from adaptive_lora.checkpoint import save_adapter, load_adapter, merge_for_export
from adaptive_lora.adaptation import adaptive_step


class DenseToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj=nn.Linear(6,6,bias=False)
        self.v_proj=nn.Linear(6,4,bias=False)
    def forward(self,x,labels):
        return SimpleNamespace(loss=(self.v_proj(self.q_proj(x))-labels).square().mean())


def arguments(path):
    return SimpleNamespace(target_modules=['q_proj','v_proj'],lora_rank=2,lora_alpha=4.,
        lora_dropout=0.,weight_a_init_method='kaiming',analysis_device='cpu',
        dense_lora_param_budget=66,dense_rank_min=1,dense_rank_max=4,output_dir=Path(path),
        adaptive_lora=True,adaptive_residual_energy_target=.9,
        adaptive_direction_efficiency_threshold=.8,adaptive_oracle_coverage_threshold=.9,
        adaptive_residual_energy_threshold=.1,adaptive_rank_max=6,adaptive_rank_step_cap=2,
        adaptive_stats_dir=Path(path)/'stats',direction_correction='none',
        adaptive_rank_init='zero_b_residual_v',residual_init_scale=.001)


class DenseTests(unittest.TestCase):
    def test_budget_gid_bounds_and_zero(self):
        rows=[dict(module='a',shape=[10,10],effective_rank=1),
              dict(module='b',shape=[10,10],effective_rank=4)]
        ranks,total=allocate_dense_ranks(rows,120,1,5)
        self.assertGreater(ranks['b'],ranks['a'])
        self.assertLessEqual(total,120)
        self.assertEqual(total,sum(ranks[r['module']]*sum(r['shape']) for r in rows))
        with self.assertRaises(ValueError): allocate_dense_ranks(rows,39)
        for row in rows: row['effective_rank']=0
        self.assertEqual(allocate_dense_ranks(rows,120)[1],40)

    def test_dense_probe_diagnostics_checkpoint_and_backward(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(2)
            model=DenseToy()
            original=copy.deepcopy(model.state_dict())
            args=arguments(tmp)
            batch=dict(x=torch.randn(3,6),labels=torch.randn(3,4))
            expected=model(**batch).loss.detach()
            # Fail if the new path accidentally invokes block_diag anywhere.
            with patch('torch.block_diag',side_effect=AssertionError('block path used')):
                names=attach_dense(model,args)
                dense_first_probe(model,names,[batch],args)
                torch.testing.assert_close(model(**batch).loss,expected,rtol=0,atol=0)
                for name in names:
                    layer=model.get_submodule(name)
                    self.assertIs(type(layer),LinearWithLoRA)
                    self.assertFalse(hasattr(layer,'n_split'))
                    a,b=dense_factors(layer)
                    self.assertEqual((a.ndim,b.ndim),(2,2))
                    self.assertTrue(torch.equal(layer.weight,original[name+'.weight']))
                    self.assertEqual(layer.lora_scaler,4/a.shape[0])
                budget=json.loads((Path(tmp)/'initial_probe/budget.json').read_text())
                self.assertLessEqual(budget['actual_lora_params'],66)
                opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
                model(**batch).loss.backward();opt.step();opt.zero_grad(set_to_none=True)
                summary=adaptive_step(model,names,[batch],args,diagnostic_only=True)
                self.assertNotIn('num_block_relaxation',summary)
                rows=[json.loads(x) for x in (args.adaptive_stats_dir/'modules.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['tangent_scope']=='exact_dense_BA_tangent' for r in rows))
                save_adapter(model,names,Path(tmp)/'adapter.pt','toy')
                restored=DenseToy();restored.load_state_dict(original)
                load_adapter(restored,Path(tmp)/'adapter.pt')
                torch.testing.assert_close(restored(**batch).loss,model(**batch).loss)
                merge_for_export(restored,names)
                torch.testing.assert_close(restored(**batch).loss,model(**batch).loss)
