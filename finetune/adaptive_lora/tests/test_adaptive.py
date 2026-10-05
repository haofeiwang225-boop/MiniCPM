import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

from adaptive_lora.subspace_matching import orth, match_subspaces
from adaptive_lora.residual_gradient import residual_gradient, block_tangent_residual
from adaptive_lora.residual_gid import svd_with_gid, gid_from_singular_values
from adaptive_lora.decision import make_adaptive_decision, allocate_rank
from adaptive_lora.rank_expansion import expand_layer
from adaptive_lora.direction_correction import apply_direction_correction
from adaptive_lora.official import LinearWithRaLoRA, LoRAConfig, freeze_base, dense_factors
from adaptive_lora.gradient_probe import probe_weight
from adaptive_lora.adaptation import adaptive_step
from adaptive_lora.checkpoint import save_adapter, load_adapter, merge_for_export


def layer(n_split=1):
    l = LinearWithRaLoRA(LoRAConfig(4, 4, lora_rank=1, lora_scaler=2,
                                    weight_a_init_method='kaiming', run_lora_in_fp32=True))
    l.dynamic_init(1, 1, n_split)
    l.weight.requires_grad_(False)
    return l


class Toy(nn.Module):
    def __init__(self, split=1):
        super().__init__()
        self.proj = layer(split)

    def forward(self, x, labels):
        return SimpleNamespace(loss=((self.proj(x)-labels)**2).mean())


class MathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_matching_both_sides(self):
        g = torch.diag(torch.tensor([3., 2., 0., 0.]))
        u, s, vh, _ = svd_with_gid(g)
        qa, qb, ca, cb = match_subspaces(vh[:2], u[:, :2], u, s, vh)
        self.assertAlmostEqual(ca, 1, places=6)
        self.assertAlmostEqual(cb, 1, places=6)
        self.assertLess(residual_gradient(g, qa, qb).norm(), 1e-6)

    def test_orthogonal(self):
        g = torch.diag(torch.tensor([2., 0., 0., 0.]))
        u, s, vh, _ = svd_with_gid(g)
        a, b = torch.eye(4)[3:], torch.eye(4)[:, 3:]
        _, _, ca, cb = match_subspaces(a, b, u, s, vh)
        self.assertEqual((ca, cb), (0., 0.))

    def test_zero_and_dependent_bases(self):
        self.assertEqual(orth(torch.zeros(4, 2)).shape, (4, 0))
        x = torch.tensor([[0., 1., 1.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        self.assertEqual(orth(x).shape[1], 1)

    def test_tangent_sum_not_intersection(self):
        qa, qb = torch.eye(4)[:, :1], torch.eye(4)[:, 1:2]
        g = qb @ torch.randn(1, 4) + torch.randn(4, 1) @ qa.T
        self.assertLess(residual_gradient(g, qa, qb).norm(), 1e-6)

    def test_residual_is_two_sided_orthogonal(self):
        g = torch.randn(7, 5)
        qa, qb = orth(torch.randn(5, 2)), orth(torch.randn(7, 2))
        r = residual_gradient(g, qa, qb)
        self.assertLess((qb.T @ r).norm(), 1e-5)
        self.assertLess((r @ qa).norm(), 1e-5)

    def test_residual_gid_rank_growth_and_zero(self):
        for rank in (1, 2, 4):
            r = torch.diag(torch.tensor([1.]*rank+[0.]*(6-rank)))
            self.assertAlmostEqual(svd_with_gid(r)[3], rank, places=5)
        self.assertEqual(gid_from_singular_values(torch.zeros(4)), 0)

    def test_gid_matches_official_nonzero_definition(self):
        from common.lora_modules.ralora import compute_effective_rank
        g = torch.randn(5, 3)
        self.assertAlmostEqual(svd_with_gid(g)[3], compute_effective_rank(g), places=4)

    def test_decisions_and_budget(self):
        d = make_adaptive_decision(.99, 1., .99, .4, .6, .01, 1., 2, 8, 1, .5, .8)
        self.assertEqual(d['decision'], 'keep')
        d = make_adaptive_decision(.2, 1., .2, .01, .19, .8, 3., 2, 8, 3, .02, .38)
        self.assertEqual((d['decision'], d['mismatch_side'], d['delta_rank']), ('direction_mismatch','A',0))
        d = make_adaptive_decision(.4, .4, 1., .2, .2, .6, 3.5, 2, 8, 4, 1., 1.)
        self.assertEqual((d['decision'], d['new_rank']), ('expand',6))
        self.assertEqual(allocate_rank(10, 7, 8, 4), 1)
        self.assertEqual(allocate_rank(10, 9, 8, 4), 0)

    def test_expansion_preserves_factors_and_scaling(self):
        for split in (1, 2):
            l = layer(split)
            a, b = dense_factors(l)
            x = torch.randn(2, 4)
            old_y = l(x).detach()
            u, s, vh, _ = svd_with_gid(torch.randn(4, 4))
            new = expand_layer(l, u, s, vh, 1, .01, "symmetric_residual")
            aa, bb = dense_factors(new.base_layer)
            self.assertTrue(torch.equal(a, aa) and torch.equal(b, bb))
            self.assertEqual(new.base_layer.lora_scaler, 2)
            expected = old_y + x @ (.01 * (u[:, :1]*s[:1]) @ vh[:1]).T
            torch.testing.assert_close(new(x), expected)
            opt = torch.optim.AdamW([p for p in new.parameters() if p.requires_grad])
            new(x).square().mean().backward()
            self.assertIsNotNone(new.add_a.grad)
            self.assertIsNotNone(new.add_b.grad)
            opt.step()
            self.assertIn(new.add_a, opt.state)

    def test_direction_none_and_blend(self):
        a, b = torch.eye(4)[3:], torch.eye(4)[:, 3:]
        u, _, vh, _ = svd_with_gid(torch.diag(torch.tensor([3., 2., 1., 0.])))
        aa, bb = apply_direction_correction(a, b, u, vh, 'both', 'none')
        self.assertIs(aa, a)
        self.assertIs(bb, b)
        aa, bb = apply_direction_correction(a, b, u, vh, 'both', 'blend', .1)
        self.assertGreater(bb[0, 0].abs(), 0)
        self.assertGreater(bb[3, 0].abs(), .8)

    def test_block_tangent_differs_from_dense_relaxation(self):
        a = torch.tensor([[1., 0.]])
        b = a.T
        g = torch.zeros(4, 4); g[0, 2] = 1
        qa, qb = orth(torch.block_diag(a, a).T), orth(torch.block_diag(b, b))
        self.assertEqual(float(residual_gradient(g, qa, qb).norm()), 0)
        self.assertEqual(float(block_tangent_residual(g, [(a,b),(a,b)]).norm()), 1)

    def test_probe_restores_weights_grads_modes_rng_optimizer(self):
        model = Toy()
        freeze_base(model)
        batch = dict(x=torch.randn(2,4), labels=torch.randn(2,4))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        model(**batch).loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        old = copy.deepcopy(model.state_dict()); state = copy.deepcopy(optimizer.state_dict())
        rng = torch.get_rng_state().clone()
        actual = probe_weight(model, model.proj, [batch, batch])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(model.training)
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, old[name]))
        self.assertFalse(model.proj.weight.requires_grad)
        self.assertIsNone(model.proj.weight.grad)
        self.assertFalse(hasattr(model.proj.weight, 'grad_stored'))
        for key, item in state['state'].items():
            for k, v in item.items():
                self.assertTrue(torch.equal(v, optimizer.state_dict()['state'][key][k]))
        model.proj.weight.requires_grad_(True)
        expected = torch.autograd.grad(model(**batch).loss, model.proj.weight)[0]
        torch.testing.assert_close(actual, expected)

    def test_probe_restores_after_exception(self):
        m=Toy(); freeze_base(m)
        with self.assertRaises(TypeError):
            probe_weight(m,m.proj,[{'bad': torch.ones(1)}])
        self.assertFalse(m.proj.weight.requires_grad)
        self.assertFalse(hasattr(m.proj.weight, 'grad_stored'))
        self.assertTrue(m.training)

    def test_disabled_exact_baseline(self):
        m=Toy(); before=copy.deepcopy(m.state_dict()); rng=torch.get_rng_state().clone()
        self.assertIsNone(adaptive_step(m, ['proj'], [], SimpleNamespace(adaptive_lora=False)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,before[k]))

    def test_checkpoint_roundtrip_and_merge(self):
        for split in (1,2):
            m=Toy(split)
            base_state = copy.deepcopy(m.proj.weight.detach())
            u,s,vh,_=svd_with_gid(torch.randn(4,4))
            m.proj=expand_layer(m.proj,u,s,vh,1,.01,"symmetric_residual")
            x=torch.randn(3,4)
            expected=m.proj(x).detach()
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'adapter.pt'
                save_adapter(m,['proj'],path,'toy')
                restored=nn.Module();restored.proj=nn.Linear(4,4,bias=False)
                restored.proj.weight.data.copy_(base_state)
                load_adapter(restored,path)
                torch.testing.assert_close(restored.proj(x),expected)
                merge_for_export(restored,['proj'])
                torch.testing.assert_close(restored.proj(x),expected)


if __name__ == '__main__':
    unittest.main()
