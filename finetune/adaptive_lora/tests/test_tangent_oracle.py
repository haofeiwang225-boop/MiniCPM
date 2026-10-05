import copy
import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
import torch
from adaptive_lora.subspace_matching import match_subspaces
from adaptive_lora.residual_gradient import residual_gradient
from adaptive_lora.residual_gid import svd_with_gid
from adaptive_lora.tangent_metrics import tangent_metrics, residual_energy_rank
from adaptive_lora.decision import make_adaptive_decision
from adaptive_lora.direction_correction import compute_A_target_given_B, compute_B_target_given_A
from adaptive_lora.rank_expansion import expand_layer
from test_adaptive import layer, Toy
from adaptive_lora.adaptation import adaptive_step
from adaptive_lora.official import freeze_base


def analyze(g, a, b):
    u, s, vh, _ = svd_with_gid(g)
    qa, qb, ca, cb = match_subspaces(a, b, u, s, vh)
    r = residual_gradient(g, qa, qb)
    m = tangent_metrics(g, r, s, a.shape[0], ca, cb)
    _, sr, _, gidr = svd_with_gid(r)
    d = make_adaptive_decision(m['tangent_coverage'], m['oracle_tangent_coverage'],
        m['direction_efficiency'], ca, cb, m['residual_energy_ratio'], gidr,
        a.shape[0], min(g.shape), residual_energy_rank(sr), m['efficiency_A'], m['efficiency_B'])
    return m, d, r


class TangentOracleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_orthogonal_decomposition_and_random_oracle_bound(self):
        for shape in ((9, 7), (7, 9), (8, 8)):
            g = torch.randn(*shape)
            for rank in (1, 2, 4):
                a, b = torch.randn(rank, shape[1]), torch.randn(shape[0], rank)
                m, _, r = analyze(g, a, b)
                torch.testing.assert_close(g.square().sum(), (g-r).square().sum()+r.square().sum())
                self.assertAlmostEqual(m['tangent_coverage']+m['residual_energy_ratio'], 1, places=5)
                self.assertLessEqual(m['tangent_coverage'], m['oracle_tangent_coverage']+2e-5)

    def test_complementary_optimal_top_2r(self):
        g = torch.diag(torch.tensor([5., 4., 3., 2., 1., .5]))
        eye = torch.eye(6)
        m, d, _ = analyze(g, eye[2:4], eye[:, :2])
        self.assertAlmostEqual(m['direction_efficiency'], 1, places=6)
        self.assertEqual(d['decision'], 'keep')
        # A individual coverage is low, but joint top-2r utilization is ideal.
        self.assertLess(m['efficiency_A'], .5)
        self.assertEqual(d['mismatch_side'], 'none')

    def test_wrong_directions_same_rank(self):
        g = torch.diag(torch.tensor([5., 4., 0., 0., 0., 0.]))
        m, d, _ = analyze(g, torch.eye(6)[4:5], torch.eye(6)[:, 5:6])
        self.assertEqual(m['tangent_coverage'], 0)
        self.assertAlmostEqual(m['oracle_tangent_coverage'], 1)
        self.assertEqual(m['direction_efficiency'], 0)
        self.assertEqual(d['decision'], 'direction_mismatch')
        self.assertEqual(d['delta_rank'], 0)

    def test_true_capacity_mismatch(self):
        g = torch.eye(10)
        m, d, _ = analyze(g, torch.eye(10)[1:2], torch.eye(10)[:, :1])
        self.assertAlmostEqual(m['direction_efficiency'], 1)
        self.assertAlmostEqual(m['oracle_tangent_coverage'], .2)
        self.assertEqual(d['decision'], 'expand')
        self.assertEqual(d['delta_rank'], 4)

    def test_energy_threshold_minimum(self):
        self.assertEqual(residual_energy_rank(torch.tensor([5., 4., 1., 0.])), 2)
        self.assertEqual(residual_energy_rank(torch.tensor([5., 4., 1., 0.]), 1), 3)
        self.assertEqual(residual_energy_rank(torch.zeros(4)), 0)
        with self.assertRaises(ValueError):
            residual_energy_rank(torch.ones(3), 0)

    def test_zero_b_expansion_output_and_first_gradient(self):
        for split in (1, 2):
            base = layer(split).eval()
            old = copy.deepcopy(base.state_dict())
            x = torch.randn(7, 4)
            expected = base(x).detach()
            u, s, vh, _ = svd_with_gid(torch.randn(4, 4))
            expanded = expand_layer(base, u, s, vh, 2, .001)
            torch.testing.assert_close(expanded(x), expected, rtol=0, atol=0)
            torch.testing.assert_close(expanded.add_a, vh[:2])
            self.assertEqual(int(torch.count_nonzero(expanded.add_b)), 0)
            for key, value in base.state_dict().items():
                self.assertTrue(torch.equal(value, old[key]))
            opt = torch.optim.AdamW([p for p in expanded.parameters() if p.requires_grad])
            expanded(x).square().sum().backward()
            self.assertGreater(float(expanded.add_b.grad.norm()), 0)
            self.assertEqual(float(expanded.add_a.grad.norm()), 0)
            opt.step()
            self.assertTrue(torch.isfinite(expanded(x)).all())

    def test_complementary_targets(self):
        g = torch.diag(torch.tensor([5., 4., 3., 2.]))
        eye = torch.eye(4)
        btarget = compute_B_target_given_A(g, eye[:1], 1)
        atarget = compute_A_target_given_B(g, eye[:, :1], 1)
        torch.testing.assert_close(btarget @ btarget.T, eye[:, 1:2] @ eye[1:2])
        torch.testing.assert_close(atarget @ atarget.T, eye[:, 1:2] @ eye[1:2])

    def test_zero_and_tiny_gradient(self):
        for scale in (0., 1e-8, 1.):
            m, _, _ = analyze(scale*torch.eye(4), torch.eye(4)[:1], torch.eye(4)[:, 1:2])
            self.assertAlmostEqual(m['tangent_coverage']+m['residual_energy_ratio'], 1, places=5)
            self.assertAlmostEqual(m['direction_efficiency'], 1, places=5)

    def test_gid_does_not_allocate_rank(self):
        args = dict(tangent_coverage=.2, oracle_tangent_coverage=.2,
                    direction_efficiency=1., coverage_A=.1, coverage_B=.1,
                    residual_energy_ratio=.8, current_rank=1, max_rank=10,
                    residual_rank=2, efficiency_A=1., efficiency_B=1.)
        self.assertEqual(make_adaptive_decision(gid_residual=1.1, **args)['delta_rank'], 2)
        self.assertEqual(make_adaptive_decision(gid_residual=9., **args)['delta_rank'], 2)


    def test_diagnostic_event_preserves_training_state(self):
        model = Toy()
        freeze_base(model)
        batch = dict(x=torch.randn(3, 4), labels=torch.randn(3, 4))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        model(**batch).loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        original = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(opt.state_dict())
        flags = [p.requires_grad for p in model.parameters()]
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(adaptive_lora=False, analysis_device='cpu',
                adaptive_residual_energy_target=.9, adaptive_direction_efficiency_threshold=.8,
                adaptive_oracle_coverage_threshold=.9, adaptive_residual_energy_threshold=.1,
                adaptive_rank_max=4, adaptive_rank_step_cap=2,
                adaptive_stats_dir=Path(tmp), direction_correction='none')
            summary = adaptive_step(model, ['proj'], [batch], args, diagnostic_only=True)
            self.assertEqual(summary['old_total_lora_params'], summary['new_total_lora_params'])
            self.assertEqual(summary['old_rank_distribution'], summary['new_rank_distribution'])
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(flags, [p.requires_grad for p in model.parameters()])
        for key, val in original.items():
            self.assertTrue(torch.equal(val, model.state_dict()[key]))
        for key, values in optimizer_state['state'].items():
            for name, val in values.items():
                self.assertTrue(torch.equal(val, opt.state_dict()['state'][key][name]))
