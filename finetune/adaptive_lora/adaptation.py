"""Legacy rank adaptation and separate fixed-rank geometric correction events."""
import torch
from pathlib import Path
from .official import blocks, dense_factors, LinearWithLoRA
from .gradient_probe import probe_weight
from .subspace_matching import match_subspaces
from .residual_gradient import residual_gradient, residual_energy_ratio, block_tangent_residual
from .residual_gid import svd_with_gid
from .decision import make_adaptive_decision
from .direction_correction import (apply_direction_correction, compute_A_target_given_B,
                                   compute_B_target_given_A)
from .tangent_metrics import tangent_metrics, residual_energy_rank
from .rank_expansion import expand_layer
from .diagnostics import write_diagnostics


def adaptive_step(model, target_names, batches, args, diagnostic_only=False):
    if not args.adaptive_lora and not diagnostic_only:
        return None  # no probe, RNG consumption, mutation or optimizer reset
    if not diagnostic_only and getattr(args, 'direction_correction', 'none') == 'geometric':
        raise ValueError('Geometric correction cannot enter the structural adaptive path')
    if not target_names:
        raise ValueError('No targets')
    old_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows = []
    # All measurements refer to the same warm-up checkpoint. Defer mutations until
    # every layer is measured, retaining only top factors/edited small A/B on CPU.
    pending = []
    for name in target_names:
        layer = model.get_submodule(name)
        is_dense = type(layer) is LinearWithLoRA
        g, batch_losses = probe_weight(model, layer, batches, return_losses=True)
        g = g.to(args.analysis_device)
        a, b = dense_factors(layer)
        a, b = a.to(g.device), b.to(g.device)
        u, s, vh, gid = svd_with_gid(g)
        qa, qb, ca, cb = match_subspaces(a, b, u, s, vh)
        residual = residual_gradient(g, qa, qb)
        # Roundoff residual is not a physical missing subspace.
        if residual.norm() <= 1e-6 * g.norm():
            residual.zero_()
        ur, sr, vhr, gidr = svd_with_gid(residual)
        old_rank = a.shape[0]
        metrics = tangent_metrics(g, residual, s, old_rank, ca, cb)
        from .tangent_metrics import frobenius_metrics
        metrics.update(frobenius_metrics(g, residual, metrics['residual_energy_ratio']))
        k_star = residual_energy_rank(sr, args.adaptive_residual_energy_target)
        decision = make_adaptive_decision(
            metrics['tangent_coverage'], metrics['oracle_tangent_coverage'],
            metrics['direction_efficiency'], ca, cb, metrics['residual_energy_ratio'],
            gidr, old_rank, min(args.adaptive_rank_max, min(g.shape)), k_star,
            metrics['efficiency_A'], metrics['efficiency_B'],
            efficiency_threshold=args.adaptive_direction_efficiency_threshold,
            oracle_threshold=args.adaptive_oracle_coverage_threshold,
            energy_threshold=args.adaptive_residual_energy_threshold,
            step_cap=args.adaptive_rank_step_cap)
        row = dict(module=name, rank=old_rank, old_rank=old_rank, gid_full=gid,
                   coverage_A=ca, coverage_B=cb, gid_residual=gidr,
                   residual_rank_k90=residual_energy_rank(sr, .90),
                   residual_rank_k_star=k_star,
                   residual_energy_target=args.adaptive_residual_energy_target,
                   structure="dense" if is_dense else "block",
                   original_scaling=layer.lora_scaler, diagnostic_only=diagnostic_only,
                   base_weight_unchanged=True, probe_batch_losses=batch_losses,
                   mean_probe_loss=sum(batch_losses)/len(batch_losses),
                   num_calibration_batches=len(batches),
                   calibration_indices_path=str(getattr(args, 'calibration_indices_path', 'in_memory_test')),
                   calibration_indices_sha256=getattr(args, 'calibration_indices_sha256', None),
                   **metrics, **decision)
        pairs = [(a, b)] if is_dense else [(aa.detach().float().to(g.device), bb.detach().float().to(g.device)) for aa, bb in blocks(layer)]
        if not is_dense and layer.n_split > 1:
            restricted = block_tangent_residual(g, pairs)
            row['restricted_block_residual_energy_ratio'] = residual_energy_ratio(g, restricted)
            row['tangent_scope'] = 'ambient_dense_relaxation_of_block_factors'
            del restricted
        else:
            row['tangent_scope'] = 'exact_dense_BA_tangent'
        row['target_side'] = decision['mismatch_side']
        row['target_subspace'] = None
        # Trajectory measures scalars only; correction targets are not consumed there.
        if not diagnostic_only and decision['decision'] == 'direction_mismatch':
            targets = {}
            if decision['mismatch_side'] in ('A', 'both'):
                targets['A_columns'] = compute_A_target_given_B(g, b, old_rank).cpu()
            if decision['mismatch_side'] in ('B', 'both'):
                targets['B_columns'] = compute_B_target_given_A(g, a, old_rank).cpu()
            folder = Path(args.adaptive_stats_dir)/'targets'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/(name+'.pt')
            torch.save(targets, path)
            row['target_subspace'] = dict(path=str(path), kind='complementary_dense_subspace' if is_dense else 'complementary_dense_relaxation',
                                           shapes={k: list(v.shape) for k,v in targets.items()})
            del targets
        if not diagnostic_only and decision['decision'] == 'expand':
            k = decision['delta_rank']
            pending.append((name, 'expand', (ur[:, :k].cpu().clone(), sr[:k].cpu().clone(), vhr[:k].cpu().clone())))
            row['residual_init_scale'] = args.residual_init_scale
            row['adaptive_rank_init'] = args.adaptive_rank_init
            row['residual_update_sign'] = ('zero_function_then_optimizer_descent'
                if args.adaptive_rank_init == 'zero_b_residual_v' else 'positive_gradient_ablation')
        elif not diagnostic_only and decision['decision'] == 'direction_mismatch' and args.direction_correction != 'none':
            corrected = []
            ro = co = 0
            for aa, bb in pairs:
                m, n = bb.shape[0], aa.shape[1]
                # Keep MELoRA structural zeros: corrections remain in each block.
                block_g = g[ro:ro+m, co:co+n]
                bu = compute_B_target_given_A(block_g, aa, aa.shape[0])
                bvh = compute_A_target_given_B(block_g, bb, aa.shape[0]).T
                aa, bb = apply_direction_correction(aa, bb, bu, bvh, decision['mismatch_side'],
                                                    args.direction_correction, args.direction_blend)
                corrected.append((aa.cpu(), bb.cpu()))
                ro += m
                co += n
            pending.append((name, 'correct', corrected))
        row['direction_correction'] = args.direction_correction if not diagnostic_only and decision['decision'] == 'direction_mismatch' else 'none'
        row['applied_delta_rank'] = 0 if diagnostic_only else decision['delta_rank']
        row['applied_new_rank'] = old_rank + row['applied_delta_rank']
        rows.append(row)
        print(row, flush=True)
        del g, a, b, u, s, vh, qa, qb, residual, ur, sr, vhr, pairs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for name, operation, data in pending:
        layer = model.get_submodule(name)
        if operation == 'expand':
            new_layer = expand_layer(layer, *data, len(data[1]), args.residual_init_scale, args.adaptive_rank_init)
            parent, _, child = name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model, child, new_layer)
        else:
            with torch.no_grad():
                pairs = [(layer.weight_a, layer.weight_b)] if type(layer) is LinearWithLoRA else blocks(layer)
                for (a, b), (new_a, new_b) in zip(pairs, data):
                    a.copy_(new_a)
                    b.copy_(new_b)
    new_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return write_diagnostics(args.adaptive_stats_dir, rows, old_params, new_params)


def geometric_correction_step(model, target_names, batches, args, optimizer, scheduler, step):
    """Fixed-rank event: measure every target before applying any mutation."""
    import copy
    from .direction_correction import plan_geometric_direction_correction, apply_geometric_correction
    from .tangent_metrics import frobenius_metrics
    from .diagnostics import write_geometric_diagnostics
    from .gradient_probe import evaluate_calibration_loss
    old_parameters = [(id(p), tuple(p.shape)) for p in model.parameters()]
    old_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    lr_before = [group['lr'] for group in optimizer.param_groups]
    # State tensor versions and identities detect unintended writes, without
    # copying every Adam moment to GPU/CPU a second time.
    state_before = {p: (state, {k:(id(v), v._version) if torch.is_tensor(v) else v
                               for k,v in state.items()}) for p,state in optimizer.state.items()}
    pre_losses, pre_mean = evaluate_calibration_loss(model, batches)
    parameter_names = {p:n for n,p in model.named_parameters()}
    rows, pending = [], []
    for name in target_names:
        layer = model.get_submodule(name)
        if type(layer) is not LinearWithLoRA:
            raise ValueError('Geometric events require ordinary Dense LinearWithLoRA')
        g, losses = probe_weight(model, layer, batches, return_losses=True)
        g = g.to(args.analysis_device)
        a, b = dense_factors(layer)
        a, b = a.to(g.device), b.to(g.device)
        rank = layer.lora_rank
        if a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError('Factor width disagrees with configured rank')
        u,s,vh,gid = svd_with_gid(g)
        qa,qb,ca,cb = match_subspaces(a,b,u,s,vh)
        residual = residual_gradient(g,qa,qb)
        if residual.norm() <= 1e-6*g.norm(): residual.zero_()
        ur,sr,vhr,gidr = svd_with_gid(residual)
        metrics = tangent_metrics(g,residual,s,rank,ca,cb)
        kstar = residual_energy_rank(sr,args.adaptive_residual_energy_target)
        diagnostic = make_adaptive_decision(metrics['tangent_coverage'],metrics['oracle_tangent_coverage'],
            metrics['direction_efficiency'],ca,cb,metrics['residual_energy_ratio'],gidr,rank,
            max(rank,args.adaptive_rank_max),kstar,metrics['efficiency_A'],metrics['efficiency_B'],
            efficiency_threshold=args.adaptive_direction_efficiency_threshold,
            oracle_threshold=args.adaptive_oracle_coverage_threshold,
            energy_threshold=args.adaptive_residual_energy_threshold,step_cap=args.adaptive_rank_step_cap)
        plan,factors = plan_geometric_direction_correction(g,a,b,metrics['oracle_tangent_coverage'],
            layer.lora_scaler, residual_threshold=args.adaptive_residual_energy_threshold,
            function_tolerance=args.direction_function_tolerance,min_gain=args.direction_min_gain,
            eta_start=args.direction_eta_start,eta_decay=args.direction_eta_decay,eta_min=args.direction_eta_min,
            diagnostic_mismatch_side=diagnostic['mismatch_side'])
        row = dict(module=name,step=step,coverage_A=ca,coverage_B=cb,gid_full=gid,gid_residual=gidr,
                   residual_rank_k90=residual_energy_rank(sr),residual_rank_k_star=kstar,
                   probe_batch_losses=losses,mean_probe_loss=sum(losses)/len(losses),
                   num_calibration_batches=len(batches),calibration_indices_path=str(args.calibration_indices_path),
                   calibration_indices_sha256=args.calibration_indices_sha256,
                   base_weight_unchanged=True,diagnostic_decision=diagnostic['decision'],
                   **metrics)
        row.update(frobenius_metrics(g,residual,metrics['residual_energy_ratio']))
        row.update(plan)
        rows.append(row)
        if factors is not None:
            pending.append((name,tuple(f.detach().cpu().clone() for f in factors),row))
        print(f'Geometric step {step}: {name}, side={row["applied_side"]}, gain={row["applied_gain"]:.6g}',flush=True)
        del g,a,b,u,s,vh,qa,qb,residual,ur,sr,vhr,factors
    changed = set()
    for name,factors,row in pending:
        layer = model.get_submodule(name)
        base_before = layer.weight.detach().cpu().clone()
        side = row['applied_side']
        untouched = layer.weight_b if side == 'A' else layer.weight_a
        untouched_before = untouched.detach().clone()
        row.update(apply_geometric_correction(layer,factors,row,optimizer))
        changed.add(layer.weight_a if side == 'A' else layer.weight_b)
        if not torch.equal(untouched_before,untouched.detach()):
            raise RuntimeError(f'Geometric correction unexpectedly changed untouched factor: {name}')
        if not torch.equal(base_before,layer.weight.detach().cpu()):
            raise RuntimeError(f'Geometric correction unexpectedly changed base weight: {name}')
        if layer.lora_scaler != row['old_scaling']:
            raise RuntimeError(f'Geometric correction unexpectedly changed scaling: {name}')
        if layer.lora_rank != row['rank_before']:
            raise RuntimeError(f'Geometric correction unexpectedly changed rank: {name}')
    for row in rows:
        row.setdefault('optimizer_state_reset_side','none')
        row.setdefault('optimizer_state_reset_count',0)
        row['scheduler_preserved'] = True
    post_losses, post_mean = evaluate_calibration_loss(model, batches)
    if old_parameters != [(id(p),tuple(p.shape)) for p in model.parameters()]:
        raise RuntimeError('Geometric correction changed Parameter identity or shape')
    new_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if old_count != new_count:
        raise RuntimeError(f'Geometric correction changed trainable parameter count: {old_count} -> {new_count}')
    if scheduler.state_dict() != scheduler_before:
        raise RuntimeError('Geometric correction changed scheduler state')
    if lr_before != [group['lr'] for group in optimizer.param_groups]:
        raise RuntimeError('Geometric correction changed optimizer param-group learning rates')
    for p in changed:
        if p in optimizer.state:
            raise RuntimeError(f'Geometric correction did not clear modified optimizer state: {parameter_names[p]}')
    for p,(state,versions) in state_before.items():
        if p in changed:
            continue
        else:
            if p not in optimizer.state or optimizer.state[p] is not state:
                raise RuntimeError(f'Geometric correction replaced untouched optimizer state: {parameter_names[p]}')
            if state.keys() != versions.keys():
                raise RuntimeError(f'Geometric correction changed untouched optimizer state keys: {parameter_names[p]}')
            for k,v in state.items():
                if ((id(v),v._version) if torch.is_tensor(v) else v) != versions[k]:
                    raise RuntimeError(f'Geometric correction changed untouched optimizer state: {parameter_names[p]}.{k}')
    summary = write_geometric_diagnostics(args.output_dir/'corrections'/f'step_{step}',rows,
        dict(step=step,old_total_lora_params=old_count,new_total_lora_params=new_count,
             pre_correction_batch_losses=pre_losses,post_correction_batch_losses=post_losses,
             pre_correction_mean_loss=pre_mean,post_correction_mean_loss=post_mean,
             immediate_loss_change=post_mean-pre_mean,
             relative_immediate_loss_change=(post_mean-pre_mean)/(abs(pre_mean)+1e-30),
             optimizer_object_preserved=True,scheduler_preserved=True,
             scheduler_last_epoch=scheduler.last_epoch,learning_rates=lr_before,
             untouched_optimizer_states_preserved=True,parameter_objects_preserved=True,
             calibration_indices_path=str(args.calibration_indices_path),
             calibration_indices_sha256=args.calibration_indices_sha256,num_calibration_batches=len(batches)))
    return summary
