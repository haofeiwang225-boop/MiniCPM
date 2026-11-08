"""Unexecuted regression: actual original class, not a retyped forward oracle."""
import copy,json,unittest
from pathlib import Path
import torch
from torch import nn
from altlora.adapters import DenseAdapter
from rank_gain_probe_support import load_original_lora_class


def errors(actual,expected):
    delta=(actual.detach().double()-expected.detach().double()).abs()
    scale=expected.detach().double().abs().clamp_min(1e-30)
    return dict(max_abs=float(delta.max()),max_relative=float((delta/scale).max()))

class HistoricalForwardTests(unittest.TestCase):
    def test_eval_train_output_and_factor_gradients(self):
        cls,config=load_original_lora_class(Path(__file__).resolve().parents[3])
        for dtype in (torch.float32,torch.bfloat16):
            for rank in (1,2,3,4):
                for training in (False,True):
                    with self.subTest(dtype=dtype,rank=rank,training=training),torch.random.fork_rng(devices=[]):
                        torch.manual_seed(17)
                        base=nn.Linear(12,9,bias=True).to(dtype)
                        original=cls(config(in_features=12,out_features=9,bias=True,lora_rank=rank,lora_scaler=4.,lora_dropout=.1,run_lora_in_fp32=True))
                        original.weight=nn.Parameter(base.weight.detach().clone(),requires_grad=False)
                        original.bias=nn.Parameter(base.bias.detach().clone(),requires_grad=False)
                        original.weight_a=nn.Parameter(torch.randn(rank,12))
                        original.weight_b=nn.Parameter(torch.randn(9,rank))
                        current=DenseAdapter(copy.deepcopy(base),rank,4.,.1)
                        with torch.no_grad():
                            current.weight_a.copy_(original.weight_a);current.weight_b.copy_(original.weight_b)
                        original.train(training);current.train(training)
                        x=torch.randn(2,5,12,dtype=dtype)
                        rng=torch.get_rng_state();old=original(x);post_old=torch.get_rng_state()
                        torch.set_rng_state(rng);new=current(x);post_new=torch.get_rng_state()
                        old.float().square().mean().backward();new.float().square().mean().backward()
                        report=dict(dtype=str(dtype),rank=rank,training=training,output=errors(new,old),
                                    A_grad=errors(current.weight_a.grad,original.weight_a.grad),
                                    B_grad=errors(current.weight_b.grad,original.weight_b.grad))
                        print('historical_forward_error '+json.dumps(report))
                        self.assertTrue(torch.equal(post_old,post_new))
                        self.assertTrue(torch.equal(new,old),report)
                        self.assertTrue(torch.equal(current.weight_a.grad,original.weight_a.grad),report)
                        self.assertTrue(torch.equal(current.weight_b.grad,original.weight_b.grad),report)
                        self.assertEqual(current.weight_a.dtype,torch.float32)
                        self.assertEqual(current.scaling,original.lora_scaler)
