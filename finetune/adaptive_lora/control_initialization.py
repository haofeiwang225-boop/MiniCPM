"""Rank-only controls: no gradients, GID or SVD. Same Dense factor initializer."""
from collections import Counter, defaultdict
from dataclasses import replace
import json
import math
import random
from pathlib import Path
from .dense_init import make_dense_layer
from .official import freeze_base
from .data_provenance import file_sha256


def budget_ranks(shapes, budget, minimum=1, maximum=4):
    """Maximize spent budget, then minimize squared distance from uniform rank.

    Shape-only bounded integer DP; lexical module order breaks ties. No GID input.
    """
    names = sorted(shapes)
    costs = [sum(shapes[n]) for n in names]
    unit = math.gcd(*costs)
    target = budget / sum(costs)
    states = {0: (0., ())}
    for name, cost in zip(names, costs):
        nxt = {}
        for spent, (penalty, ranks) in states.items():
            for rank in range(minimum, min(maximum, min(shapes[name]))+1):
                new = spent + rank * (cost // unit)
                if new > budget // unit:
                    continue
                candidate = (penalty + (rank-target)**2, ranks + (rank,))
                if new not in nxt or candidate < nxt[new]:
                    nxt[new] = candidate
        states = nxt
    if not states:
        raise ValueError('Budget cannot fund minimum ranks')
    spent = max(states)
    return dict(zip(names, states[spent][1])), spent * unit


def select_ranks(shapes, mode, rank, budget, maximum, seed, source=None):
    if mode == 'vanilla':
        if any(rank > min(shape) or rank < 1 for shape in shapes.values()):
            raise ValueError('Uniform rank invalid for target shapes')
        ranks = dict.fromkeys(shapes, rank)
    elif mode == 'budget_matched':
        return budget_ranks(shapes, budget, maximum=maximum)
    elif mode in ('shuffled_gid', 'replay_gid'):
        ranks = json.loads(Path(source).read_text())
        if set(ranks) != set(shapes) or any(type(r) is not int or not 1 <= r <= min(shapes[n]) for n,r in ranks.items()):
            raise ValueError('Rank manifest must match all target names and valid dimensions')
        if mode == 'shuffled_gid':
            groups = defaultdict(list)
            for name in sorted(shapes):
                groups[tuple(shapes[name])].append(name)
            rng = random.Random(seed)  # never consume training RNG
            for names in groups.values():
                values = [ranks[n] for n in names]
                rng.shuffle(values)
                ranks.update(zip(names, values))
    else:
        raise ValueError('Unknown control initialization')
    return ranks, sum(ranks[n]*sum(shapes[n]) for n in shapes)


def initialize_control(model, names, args):
    shapes = {n: tuple(model.get_submodule(n).weight.shape) for n in names}
    ranks, total = select_ranks(shapes, args.dense_init_mode, args.lora_rank,
        args.dense_lora_param_budget, args.dense_rank_max, args.seed, args.rank_manifest)
    if args.dense_init_mode in ('shuffled_gid', 'replay_gid') and total != args.dense_lora_param_budget:
        raise ValueError('Source ranks do not match requested actual parameter budget')
    # attach_dense already consumed the same provisional initialization as GID.
    # probe_weight restores RNG; reproducing this second initialization is fair.
    for name in names:
        old = model.get_submodule(name)
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model, child,
                make_dense_layer(old, replace(old.lora_config, lora_rank=ranks[name])))
    freeze_base(model)
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual != total:
        raise RuntimeError('Actual parameter budget mismatch')
    folder = Path(args.output_dir)/'initial_probe'
    folder.mkdir(exist_ok=False)
    summary = dict(mode=args.dense_init_mode, rank_pattern=ranks,
        rank_distribution=dict(Counter(ranks.values())), total_lora_parameters=actual,
        requested_budget=args.dense_lora_param_budget, shape_only=args.dense_init_mode in ('vanilla','budget_matched'),
        gradient_probe_executed=False, svd_executed=False, initializer='existing_kaiming_A_zero_B',
        rng_protocol='same_provisional_then_final_initialization_as_GID',
        rank_manifest_sha256=file_sha256(args.rank_manifest) if args.rank_manifest else None)
    (folder/'rank.json').write_text(json.dumps(ranks, indent=2))
    (folder/'summary.json').write_text(json.dumps(summary, indent=2))
    (folder/'budget.json').write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)
