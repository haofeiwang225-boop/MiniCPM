"""Single second-probe event between two independent optimizers."""
import torch
from .official import blocks, dense_factors
from .gradient_probe import probe_weight
from .subspace_matching import match_subspaces
from .residual_gradient import residual_gradient, residual_energy_ratio, block_tangent_residual
from .residual_gid import svd_with_gid
from .decision import make_adaptive_decision
from .direction_correction import apply_direction_correction
from .rank_expansion import expand_layer
from .diagnostics import write_diagnostics


def adaptive_step(model, target_names, batches, args):
    if not args.adaptive_lora:
        return None  # no probe, RNG consumption, mutation or optimizer reset
    if not target_names:
        raise ValueError('No targets')
    old_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows = []
    # All measurements refer to the same warm-up checkpoint. Defer mutations until
    # every layer is measured, retaining only top factors/edited small A/B on CPU.
    pending = []
    for name in target_names:
        layer = model.get_submodule(name)
        g = probe_weight(model, layer, batches).to(args.analysis_device)
        a, b = dense_factors(layer)
        a, b = a.to(g.device), b.to(g.device)
        u, s, vh, gid = svd_with_gid(g)
        qa, qb, ca, cb = match_subspaces(a, b, u, s, vh)
        residual = residual_gradient(g, qa, qb)
        rho = residual_energy_ratio(g, residual)
        # Roundoff residual is not a physical missing subspace.
        if residual.norm() <= 1e-6 * g.norm():
            residual.zero_()
        ur, sr, vhr, gidr = svd_with_gid(residual)
        old_rank = a.shape[0]
        decision = make_adaptive_decision(
            ca, cb, rho, gid, gidr, old_rank,
            min(args.adaptive_rank_max, min(g.shape)),
            args.adaptive_alignment_threshold, args.adaptive_residual_energy_threshold,
            args.adaptive_rank_step_cap)
        row = dict(module=name, old_rank=old_rank, gid_full=gid, coverage_A=ca,
                   coverage_B=cb, residual_energy_ratio=rho, gid_residual=gidr,
                   n_split=layer.n_split, mini_rank=layer.mini_lora_rank,
                   original_scaling=layer.lora_scaler, **decision)
        pairs = [(aa.detach().float().to(g.device), bb.detach().float().to(g.device)) for aa, bb in blocks(layer)]
        if layer.n_split > 1:
            restricted = block_tangent_residual(g, pairs)
            row['restricted_block_residual_energy_ratio'] = residual_energy_ratio(g, restricted)
            row['tangent_scope'] = 'ambient_dense_relaxation_of_block_factors'
            del restricted
        else:
            row['tangent_scope'] = 'exact_dense_BA_tangent'
        if decision['decision'] == 'expand':
            k = decision['delta_rank']
            pending.append((name, 'expand', (ur[:, :k].cpu().clone(), sr[:k].cpu().clone(), vhr[:k].cpu().clone())))
            row['residual_init_scale'] = args.residual_init_scale
            row['residual_update_sign'] = 'positive_gradient_as_requested'
        elif decision['decision'] == 'direction_mismatch' and args.direction_correction != 'none':
            corrected = []
            ro = co = 0
            for aa, bb in pairs:
                m, n = bb.shape[0], aa.shape[1]
                # Keep MELoRA structural zeros: corrections remain in each block.
                bu, _, bvh, _ = svd_with_gid(g[ro:ro+m, co:co+n])
                aa, bb = apply_direction_correction(aa, bb, bu, bvh, decision['mismatch_side'],
                                                    args.direction_correction, args.direction_blend)
                corrected.append((aa.cpu(), bb.cpu()))
                ro += m
                co += n
            pending.append((name, 'correct', corrected))
        row['direction_correction'] = args.direction_correction if decision['decision'] == 'direction_mismatch' else 'none'
        rows.append(row)
        print(row, flush=True)
        del g, a, b, u, s, vh, qa, qb, residual, ur, sr, vhr, pairs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for name, operation, data in pending:
        layer = model.get_submodule(name)
        if operation == 'expand':
            new_layer = expand_layer(layer, *data, len(data[1]), args.residual_init_scale)
            parent, _, child = name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model, child, new_layer)
        else:
            with torch.no_grad():
                for (a, b), (new_a, new_b) in zip(blocks(layer), data):
                    a.copy_(new_a)
                    b.copy_(new_b)
    new_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return write_diagnostics(args.adaptive_stats_dir, rows, old_params, new_params)
