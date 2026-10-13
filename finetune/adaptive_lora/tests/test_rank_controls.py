import copy
import json
import tempfile
import unittest
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import torch
from adaptive_lora.control_initialization import budget_ranks, select_ranks, initialize_control
from adaptive_lora.dense_init import attach_dense, dense_first_probe
from test_dense_init import DenseToy, arguments


class RankControlTests(unittest.TestCase):
    def test_real_shapes_exact_budget_without_gid(self):
        shapes={f'layer{i}.{kind}':shape for i in range(42) for kind,shape in [('q',(2048,2048)),('v',(256,2048))]}
        with patch('pathlib.Path.read_text',side_effect=AssertionError('No GID data reads')):
            ranks,total=budget_ranks(shapes,480000,maximum=4)
            self.assertEqual(total,480000)
            uniform,total=select_ranks(shapes,'vanilla',2,480000,4,42)
            self.assertEqual(total,537600)
            self.assertEqual(set(uniform.values()),{2})
        self.assertEqual(sum(ranks[n]*sum(shapes[n]) for n in ranks),480000)

    def test_shuffled_shape_groups_budget_and_rng(self):
        shapes={f'layer{i}.{k}':s for i in range(8) for k,s in [('q',(8,8)),('v',(4,8))]}
        ranks={n:1+i%3 for i,n in enumerate(shapes)}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ranks.json';p.write_text(json.dumps(ranks))
            state=torch.get_rng_state().clone()
            out,total=select_ranks(shapes,'shuffled_gid',2,9999,4,42,p)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            self.assertNotEqual(out,ranks)
            self.assertEqual(Counter(out.values()),Counter(ranks.values()))
            self.assertEqual(total,sum(ranks[n]*sum(shapes[n]) for n in ranks))
            for shape in set(shapes.values()):
                self.assertEqual(Counter(out[n] for n in shapes if shapes[n]==shape),Counter(ranks[n] for n in shapes if shapes[n]==shape))

    def test_replay_matches_gid_factors_and_rng_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            torch.manual_seed(42); model=DenseToy();args=arguments(Path(d)/'gid')
            args.output_dir.mkdir();names=attach_dense(model,args)
            replay=copy.deepcopy(model)
            batch=dict(x=torch.ones(2,6),labels=torch.zeros(2,4))
            state=torch.get_rng_state().clone()
            dense_first_probe(model,names,[batch],args)
            final_state=torch.get_rng_state().clone()
            args.rank_manifest=args.output_dir/'initial_probe/rank.json'
            args.output_dir=Path(d)/'replay';args.output_dir.mkdir()
            args.dense_init_mode='replay_gid';args.seed=42
            torch.set_rng_state(state)
            with patch('torch.linalg.svd',side_effect=AssertionError('No SVD in replay')):
                initialize_control(replay,names,args)
            self.assertTrue(torch.equal(final_state,torch.get_rng_state()))
            for name in names:
                for factor in ('weight_a','weight_b'):
                    self.assertTrue(torch.equal(getattr(model.get_submodule(name),factor),getattr(replay.get_submodule(name),factor)))
