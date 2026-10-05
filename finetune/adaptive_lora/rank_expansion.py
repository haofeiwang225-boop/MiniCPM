"""Append residual directions at the phase boundary; never resize a live Parameter."""
import torch
from torch import nn
from torch.nn import functional as F


def residual_factors(u, s, vh, k, gamma, method="zero_b_residual_v"):
    if gamma < 0 or k < 1 or k > len(s):
        raise ValueError('Invalid residual initialization')
    if method == "zero_b_residual_v":
        return vh[:k].clone(), torch.zeros_like(u[:, :k])
    if method != "symmetric_residual":
        raise ValueError("Unknown rank initialization")
    root = (gamma * s[:k]).sqrt()
    return root[:, None] * vh[:k], u[:, :k] * root[None, :]


class ExpandedLoRA(nn.Module):
    """Retain official block layout/scaling; additive branch has unit scaling.

    If base delta=s*B*A, expanded delta=s*B*A + B_add*A_add.
    Thus old parameters AND old function are preserved, and default new delta=0 (symmetric ablation: gamma*R_k).
    """
    def __init__(self, base_layer, a_add, b_add):
        super().__init__()
        self.base_layer = base_layer
        self.add_a = nn.Parameter(a_add.to(device=base_layer.weight.device, dtype=torch.float32).clone())
        self.add_b = nn.Parameter(b_add.to(device=base_layer.weight.device, dtype=torch.float32).clone())
        self.in_features, self.out_features = base_layer.in_features, base_layer.out_features

    @property
    def weight(self):
        return self.base_layer.weight

    def forward(self, x):
        result = self.base_layer(x)
        branch = F.linear(F.linear(self.base_layer.lora_dropout(x.float()), self.add_a), self.add_b)
        return result + branch.to(result.dtype)


def expand_layer(layer, u, s, vh, delta_rank, gamma, method="zero_b_residual_v"):
    a, b = residual_factors(u, s, vh, delta_rank, gamma, method)
    return ExpandedLoRA(layer, a, b)
