"""GID-guided Dense LoRA initialization with a budget in actual parameters.

Uses the upstream ordinary LinearWithLoRA, never MELoRA or dynamic_init.
This is a GID Dense variant, not the unchanged official RaLoRA baseline.
"""
import json
import math
from collections import Counter
from pathlib import Path
import torch
from .official import LinearWithLoRA, LoRAConfig, freeze_base
from .gradient_probe import probe_weight
from .residual_gid import svd_with_gid


def allocate_dense_ranks(records, budget, min_rank=1, max_rank=64):
    """r_l ~= lambda * GID_l, bounded and rounded under sum r_l*(in+out).

    Zero-GID spectra receive the minimum rank. Remaining integer budget is
    assigned by largest deficit from the continuous solution, deterministically.
    Unspendable budget is retained, never exceeded.
    """
    if not records or budget < 1 or min_rank < 1 or max_rank < min_rank:
        raise ValueError('Invalid dense budget/rank bounds')
    costs = [sum(r['shape']) for r in records]
    caps = [min(max_rank, min(r['shape'])) for r in records]
    gids = [float(r['effective_rank']) for r in records]
    if any(not math.isfinite(g) or g < 0 for g in gids):
        raise ValueError('Invalid GID')
    if any(c < min_rank for c in caps) or sum(min_rank*c for c in costs) > budget:
        raise ValueError('Dense budget cannot fund minimum ranks')
    def ideal(scale):
        return [max(min_rank, min(cap, scale*g)) for cap,g in zip(caps,gids)]
    lo, hi = 0., max((cap/g for cap,g in zip(caps,gids) if g > 0), default=0.)
    for _ in range(80):
        mid = (lo+hi)/2
        if sum(r*c for r,c in zip(ideal(mid),costs)) <= budget:
            lo = mid
        else:
            hi = mid
    target = ideal(lo)
    ranks = [int(r) for r in target]
    remaining = budget - sum(r*c for r,c in zip(ranks,costs))
    while True:
        eligible = [i for i in range(len(ranks)) if costs[i] <= remaining
                    and ranks[i] < caps[i] and gids[i] > 0]
        if not eligible:
            break
        i = max(eligible, key=lambda i: (target[i]-ranks[i], gids[i], -i))
        ranks[i] += 1
        remaining -= costs[i]
    return {row['module']: r for row,r in zip(records,ranks)}, budget-remaining


def make_dense_layer(old, config):
    layer = LinearWithLoRA(config)
    layer.lora_config = config  # upstream base class does not retain this metadata
    layer.weight, layer.bias = old.weight, old.bias
    layer.init_lora_weights()
    layer.to(device=old.weight.device)
    return layer


def attach_dense(model, args):
    targets = [(n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)
               and n.rsplit('.',1)[-1] in args.target_modules]
    if {n.rsplit('.',1)[-1] for n,_ in targets} != set(args.target_modules):
        raise ValueError('Unmatched Dense LoRA target modules')
    for name, old in targets:
        config = LoRAConfig(old.in_features, old.out_features, bias=old.bias is not None,
            lora_rank=min(args.lora_rank, old.in_features, old.out_features),
            lora_scaler=args.lora_alpha, lora_dropout=args.lora_dropout,
            weight_a_init_method=args.weight_a_init_method, run_lora_in_fp32=True)
        parent,_,child=name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model,child,make_dense_layer(old,config))
    freeze_base(model)
    return [n for n,_ in targets]


def dense_first_probe(model, names, batches, args):
    records=[]
    manifest_path = getattr(args, 'calibration_indices_path', None)
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else None
    if manifest and manifest['num_calibration_batches'] != len(batches):
        raise ValueError('Initial probe batch count differs from fixed manifest')
    # All provisional B=0: measurements refer to the same unchanged base function.
    for name in names:
        layer=model.get_submodule(name)
        g=probe_weight(model,layer,batches).to(args.analysis_device)
        u,s,vh,gid=svd_with_gid(g)
        records.append(dict(module=name,shape=list(g.shape),effective_rank=gid, gid_full=gid,
                            num_calibration_batches=len(batches), base_weight_unchanged=True))
        print(f'Initial probe {len(records)}/{len(names)}: {name}, K={len(batches)}, GID={gid:.6f}', flush=True)
        del g,u,s,vh
    budget=args.dense_lora_param_budget
    if budget is None:
        budget=sum(min(args.lora_rank,min(r['shape']))*sum(r['shape']) for r in records)
    ranks,total=allocate_dense_ranks(records,budget,args.dense_rank_min,args.dense_rank_max)
    from dataclasses import replace
    for name in names:
        old=model.get_submodule(name)
        config=replace(old.lora_config,lora_rank=ranks[name])
        parent,_,child=name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model,child,make_dense_layer(old,config))
    freeze_base(model)
    actual=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert actual == total and actual <= budget
    folder=Path(args.output_dir)/'initial_probe'
    folder.mkdir(parents=True,exist_ok=True)
    for row in records:
        row['allocated_rank'] = ranks[row['module']]
        row['calibration_indices'] = manifest['batch_indices'] if manifest else None
    (folder/'modules.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
    (folder/'rank.json').write_text(json.dumps(ranks,indent=2)+'\n')
    (folder/'svd_profile.json').write_text(json.dumps(dict(modules=records),indent=2)+'\n')
    summary=dict(structure='dense',allocation='bounded_GID_proportional_actual_parameter_budget',
                 parameter_budget=budget,actual_lora_params=actual,unused_budget=budget-actual,
                 rank_min=args.dense_rank_min,rank_max=args.dense_rank_max,
                 scaling='lora_alpha / allocated_dense_rank',rank_pattern=ranks,
                 rank_distribution=dict(Counter(ranks.values())),total_lora_parameters=actual,
                 num_calibration_batches=len(batches),
                 calibration_indices=manifest['batch_indices'] if manifest else None,
                 calibration_indices_path=str(manifest_path) if manifest_path else None,
                 calibration_indices_sha256=getattr(args, 'calibration_indices_sha256', None))
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (folder/'budget.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary,flush=True)
