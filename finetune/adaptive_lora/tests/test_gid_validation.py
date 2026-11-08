"""Review-only tests: deliberately not executed during code authoring."""
import unittest
import torch
from adaptive_lora.gid_validation import spectrum_metrics,coverage_diagnostics,strict_uniform_control,FixedReference,actual_update,budget

class GIDValidationTests(unittest.TestCase):
    def test_entropy_uses_singular_values_not_energy(self):
        r=spectrum_metrics(torch.diag(torch.tensor([3.,1.])))
        self.assertEqual(r['singular_distribution'],[.75,.25])
        self.assertAlmostEqual(r['gid'],1.7547653506033232)
    def test_zero_spectrum(self):self.assertEqual(spectrum_metrics(torch.zeros(2,2))['gid'],0.)
    def test_initial_zero_b_active_oracle(self):
        r=coverage_diagnostics(torch.eye(4),torch.eye(4)[:1],torch.zeros(4,1))
        self.assertEqual(r['C_B'],0);self.assertEqual(r['active_oracle'],.25)
        self.assertEqual(r['nominal_full_oracle_top_2r'],.5);self.assertEqual(r['E'],1.)
        self.assertAlmostEqual(r['C_T']+r['rho'],1.)
    def test_zero_gradient_undefined(self):
        self.assertEqual(coverage_diagnostics(torch.zeros(4,4),torch.ones(1,4),torch.zeros(4,1))['status'],'UNDEFINED_ZERO_GRADIENT')
    def test_uniform_budget_impossible(self):
        shapes={**{f'q{i}':(2048,2048) for i in range(42)},**{f'v{i}':(256,2048) for i in range(42)}}
        with self.assertRaises(ValueError):strict_uniform_control(shapes,480000)
    def test_fixed_reference_cannot_be_overwritten(self):
        g={'x':torch.ones(2,2)};r=FixedReference(g);g['x'].zero_();r.get()['x'].zero_()
        self.assertTrue(torch.equal(r.get()['x'],torch.ones(2,2)))
    def test_actual_update_and_nonnegative_fit(self):
        a=torch.tensor([[1.,0.]]);b=torch.zeros(2,1);bn=torch.tensor([[-1.],[0.]])
        g=torch.diag(torch.tensor([1.,0.]));r=actual_update(a,b,a,bn,1.,g,g)
        self.assertTrue(torch.equal(r['delta'],bn@a));self.assertEqual(r['cos_dynamic'],1.)
        self.assertEqual(r['optimal_nonnegative_scale'],1.);self.assertEqual(r['update_approximation_error'],0.)
    def test_independent_random_budget_controls(self):
        from adaptive_lora.gid_validation import random_budget_controls
        shapes={'a':(4,4),'b':(4,4),'c':(4,4)}
        maps=random_budget_controls(shapes,48,count=5)
        self.assertEqual(maps,random_budget_controls(shapes,48,count=5))
        self.assertEqual(len({tuple(sorted(m.items())) for m in maps}),5)
        self.assertTrue(all(budget(m,shapes)==48 for m in maps))
    def test_rank_deficient_factors(self):
        a=torch.tensor([[1.,0.,0.],[2.,0.,0.]])
        b=torch.tensor([[0.,0.],[1.,2.],[0.,0.]])
        row=coverage_diagnostics(torch.eye(3),a,b)
        self.assertEqual(row['numerical_rank_A'],1);self.assertEqual(row['numerical_rank_B'],1)
        self.assertAlmostEqual(row['C_T']+row['rho'],1.)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):spectrum_metrics(torch.full((2,2),float('nan')))
    def test_positive_update_cannot_fit_descent(self):
        a=torch.eye(2);b=torch.zeros(2,2);g=torch.eye(2)
        r=actual_update(a,b,a,g,1.,g,g)
        self.assertEqual(r['optimal_nonnegative_scale'],0.)
        self.assertEqual(r['update_approximation_error'],1.)
