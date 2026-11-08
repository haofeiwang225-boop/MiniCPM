"""Fresh initialization from rank metadata; accepts no historical adapter path."""
import torch
from altlora.adapters import DenseAdapter, adapter_state, restore_factors


def attach_pattern(model, pattern):
    targets={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)
             and n.rsplit('.',1)[-1] in ('q_proj','v_proj')}
    if set(targets)!=set(pattern):raise ValueError('Real model targets differ from rank inventory')
    for p in model.parameters():p.requires_grad_(False)
    layers={}
    for n in sorted(pattern):
        spec=pattern[n];base=targets[n];r=spec['rank']
        if [r,base.in_features]!=spec['A_shape'] or [base.out_features,r]!=spec['B_shape']:
            raise ValueError('Real model shape mismatch: '+n)
        layer=DenseAdapter(base,r,alpha=4.,dropout=.1)
        if layer.scaling!=spec['scaling']:raise ValueError('Scaling mismatch')
        parent,_,child=n.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model,child,layer);layers[n]=layer
    actual=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual!=sum(s['parameters'] for s in pattern.values()):raise ValueError('Parameter count mismatch')
    return layers


def restore_initial(layers, saved):
    restore_factors(layers,saved)
    for n,m in layers.items():
        if not torch.equal(m.weight_a.detach().cpu(),saved[n]['A']) or not torch.equal(m.weight_b.detach().cpu(),saved[n]['B']):
            raise ValueError('Initial factor copy was not exact: '+n)
        if torch.count_nonzero(m.weight_b):raise ValueError('Fresh initial B must be exactly zero')


def assert_zero_b(layers):
    if any(torch.count_nonzero(m.weight_b) for m in layers.values()):raise ValueError('Initial B not zero')
