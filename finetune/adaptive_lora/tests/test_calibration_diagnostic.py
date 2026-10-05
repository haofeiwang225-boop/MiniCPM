import json
import tempfile
import unittest
from pathlib import Path
import torch
from adaptive_lora.calibration import write_calibration_indices, load_calibration_batches
from adaptive_lora.gradient_probe import probe_weight
from adaptive_lora.tangent_metrics import frobenius_metrics
from test_adaptive import Toy


class CalibrationTests(unittest.TestCase):
    def test_manifest_reload_and_tampering(self):
        dataset=list(range(9))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'calibration_indices.json'
            sha=write_calibration_indices(p,dataset,1,8,'training')
            for step in (2,20,100,500):
                self.assertEqual(load_calibration_batches(p,sha,dataset,list),[[i] for i in range(8)])
            self.assertEqual(json.loads(p.read_text())['num_calibration_batches'],8)
            p.write_text(p.read_text()+' ')
            with self.assertRaises(ValueError):load_calibration_batches(p,sha,dataset,list)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_calibration_indices(Path(tmp)/'x',list(range(7)),1,8,'training')

    def test_eight_batch_mean_gradient_and_losses(self):
        torch.manual_seed(13)
        model=Toy()
        batches=[dict(x=torch.randn(2,4),labels=torch.randn(2,4)) for _ in range(8)]
        individual=[probe_weight(model,model.proj,[b],return_losses=True) for b in batches]
        g,losses=probe_weight(model,model.proj,batches,return_losses=True)
        torch.testing.assert_close(g,torch.stack([x[0] for x in individual]).mean(0))
        self.assertEqual(losses,[x[1][0] for x in individual])

    def test_absolute_norm_energy_and_zero(self):
        g=torch.diag(torch.tensor([3.,4.]))
        r=torch.diag(torch.tensor([0.,4.]))
        m=frobenius_metrics(g,r,16/25)
        self.assertEqual(m['gradient_fro_norm'],5)
        self.assertEqual(m['residual_fro_norm'],4)
        self.assertEqual(m['tangent_fro_norm'],3)
        self.assertLess(m['residual_norm_consistency_error'],1e-12)
        self.assertEqual(m['energy_decomposition_error'],0)
        self.assertEqual(frobenius_metrics(g*0,r*0,0)['gradient_fro_norm'],0)
