"""Opt-in GID-RSC: PRE planning, two training gradients, transactional joint trial."""
from collections import Counter
import copy
import json
from pathlib import Path
import time
import torch
from .official import LinearWithLoRA, dense_factors
from .gradient_probe import probe_weight
from .residual_gid import svd_with_gid
from .subspace_matching import match_subspaces
from .residual_gradient import residual_gradient
from .tangent_metrics import tangent_metrics
from .direction_correction import plan_geometric_direction_correction
from .direction_correction_v2 import (preserved_training_state, factor_directional_derivatives,
                                     passes_direction_check, joint_loss_search, trial_factors)
from .rsc_calibration import load_rsc_samples
from .data_provenance import git_version, file_sha256


def plan_layer(model, layer, proposal, args):
    """Use precisely the v1 full-gradient geometry; no rank allocation here."""
    g = probe_weight(model, layer, proposal).to(args.analysis_device)
    a, b = (x.to(g.device) for x in dense_factors(layer))
    u, s, vh, gid = svd_with_gid(g)
    qa, qb, ca, cb = match_subspaces(a, b, u, s, vh)
    residual = residual_gradient(g, qa, qb)
    if residual.norm() <= 1e-6*g.norm():
        residual.zero_()
    metrics = tangent_metrics(g, residual, s, layer.lora_rank, ca, cb)
    plan, factors = plan_geometric_direction_correction(g, a, b, metrics['oracle_tangent_coverage'],
        layer.lora_scaler, residual_threshold=args.adaptive_residual_energy_threshold,
        function_tolerance=args.direction_function_tolerance, min_gain=args.direction_min_gain,
        eta_start=args.direction_eta_start, eta_decay=args.direction_eta_decay, eta_min=args.direction_eta_min)
    plan.update(gid_full=gid, coverage_A=ca, coverage_B=cb)
    return plan, factors


def compute_rsc_candidates(model, layers, proposal, selection, args, rows=None, candidates=None):
    """Shared computation only: no optimizer, scheduler or permanent factor mutation.

    Joint forward trials restore PRE values in finally. Tensor versions may advance.
    Callers can retain partially built rows when computation raises.
    """
    rows = [] if rows is None else rows
    candidates = [] if candidates is None else candidates
    timings = {}
    devices = {p.device for p in model.parameters() if p.is_cuda}
    analysis = torch.device(args.analysis_device)
    if analysis.type == 'cuda':
        devices.add(analysis)

    def synchronize():
        for device in devices:
            torch.cuda.synchronize(device)

    synchronize()
    last = time.perf_counter()

    def mark(name):
        nonlocal last
        synchronize()
        now = time.perf_counter()
        timings[name] = now - last
        last = now

    for name, layer in layers.items():
        plan, factors = plan_layer(model, layer, proposal, args)
        row = dict(module=name, v1_plan=plan, original_geometric_gain=plan['applied_gain'],
            proposal_derivative=None, selection_derivative=None, direction_accepted=False,
            applied_side='none', applied_strength=0., optimizer_state_reset_count=0,
            reason=plan['correction_skip_reason'] or 'pending_direction_check')
        rows.append(row)
        if factors is not None:
            side = plan['applied_side']
            if side not in ('A','B'):
                raise ValueError('V1 candidate must select exactly one factor')
            p = layer.weight_a if side=='A' else layer.weight_b
            target = factors[0 if side=='A' else 1].detach().to(device='cpu', dtype=p.dtype).clone()
            candidates.append(dict(module=name, layer=layer, parameter=p, side=side, target=target, row=row))
    mark('candidate_generation')
    # No factor mutation has occurred: both groups see the identical PRE model.
    pd = factor_directional_derivatives(model, candidates, proposal)
    sd = factor_directional_derivatives(model, candidates, selection)
    qualified = []
    for c in candidates:
        name, row = c['module'], c['row']
        row.update(proposal_derivative=pd[name], selection_derivative=sd[name])
        passed = passes_direction_check(pd[name]['mean'], sd[name]['mean'],
            args.rsc_proposal_descent_margin, args.rsc_selection_descent_margin)
        row['direction_accepted'] = passed
        row['reason'] = 'pending_joint_loss_check' if passed else 'cross_sample_descent_threshold_failed'
        if passed:
            qualified.append(c)
    mark('gradient_gating')
    accepted, joint = joint_loss_search(model, qualified, proposal, selection,
        args.rsc_trial_strengths, args.direction_function_tolerance, args.rsc_loss_min_improvement)
    mark('joint_loss_search')
    return qualified, accepted, joint, timings


def _state_signature(state):
    return {k: (id(v), v._version) if torch.is_tensor(v) else copy.deepcopy(v) for k, v in state.items()}


def geometric_v2_step(model, target_names, dataset, collator, args, optimizer, scheduler, step):
    # Includes manifest/dataset access in the RNG transaction as well as forwards.
    with preserved_training_state(model):
        return _geometric_v2_step(model, target_names, dataset, collator, args, optimizer, scheduler, step)


def _geometric_v2_step(model, target_names, dataset, collator, args, optimizer, scheduler, step):
    if args.direction_correction != 'geometric_v2' or args.adaptive_lora or args.lora_structure != 'dense':
        raise ValueError('RSC requires geometric_v2, Dense LoRA, adaptive_lora=false')
    if not target_names or len(set(target_names)) != len(target_names):
        raise ValueError('RSC needs unique target modules')
    layers = {n: model.get_submodule(n) for n in target_names}
    if any(type(layer) is not LinearWithLoRA for layer in layers.values()):
        raise ValueError('RSC only supports ordinary Dense LinearWithLoRA')
    proposal, selection, manifest = load_rsc_samples(dataset, collator, args)
    out = Path(args.output_dir)/'corrections_v2'/f'step_{step}'
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    params = list(model.parameters())
    param_meta = [(id(p), tuple(p.shape), p.requires_grad, p.grad, p._version) for p in params]
    trainable = sum(p.numel() for p in params if p.requires_grad)
    originals = {p: p.detach().cpu().clone() for layer in layers.values() for p in (layer.weight_a, layer.weight_b)}
    ranks = {n: (layer.lora_rank, layer.lora_scaler) for n, layer in layers.items()}
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    groups = [(group, list(group['params']), copy.deepcopy({k:v for k,v in group.items() if k!='params'}))
              for group in optimizer.param_groups]
    states = {p: (state, _state_signature(state)) for p, state in optimizer.state.items()}
    rows, candidates, accepted, changed = [], [], None, set()
    summary = dict(step=step, mode='geometric_v2', prototype='GID-RSC-v2-first', **git_version(),
        training_samples=manifest, manifest_sha256=args.rsc_manifest_sha256,
        model=args.model_name_or_path, model_class=type(model).__name__,
        model_config=model.config.to_dict() if hasattr(model,'config') else None,
        parameter_dtypes=sorted({str(p.dtype) for p in params}), torch_version=str(torch.__version__),
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_sha256={name:file_sha256(Path(__file__).with_name(name)) for name in (
            'adaptation_v2.py','direction_correction_v2.py','rsc_calibration.py','direction_correction.py','gradient_probe.py')},
        old_total_lora_params=trainable, optimizer_state_reset_count=0,
        interpolation='factor_linear_not_grassmann', optimizer_was_reconstructed=False)

    def require(condition, text):
        if not condition:
            raise RuntimeError('RSC invariant failed: '+text)

    try:
        with preserved_training_state(model):
            qualified, accepted, joint, timings = compute_rsc_candidates(
                model, layers, proposal, selection, args, rows, candidates)
            summary['stage_timings_seconds'] = timings
            summary['joint_loss_check'] = joint
            if accepted is not None:
                # Revalidate casted factors at the exact accepted chord strength.
                values, checks = trial_factors(qualified, joint['accepted_strength'], args.direction_function_tolerance)
                require(all(c['feasible'] for c in checks), 'accepted candidate rank/trust')
                require(all(torch.equal(a,b) for a,b in zip(accepted,values)), 'trial/commit value mismatch')
                for c, value, check in zip(qualified, accepted, checks):
                    row, p = c['row'], c['parameter']
                    if not check['changed']:
                        row['reason'] = 'rounded_to_no_change'
                        continue
                    with torch.no_grad():
                        p.copy_(value)
                    changed.add(p)
                    row.update(applied_side=c['side'], applied_strength=joint['accepted_strength'],
                               applied_checks=check, reason='accepted')
            else:
                for c in qualified:
                    c['row']['reason'] = joint['reason']
        # State audit before touching Adam. Original grad objects/modes have been restored.
        require(len(list(model.parameters())) == len(params), 'parameter list length')
        for current, p, (identity, shape, flag, grad, version) in zip(model.parameters(),params,param_meta):
            require(current is p and id(p)==identity and tuple(p.shape)==shape and p.requires_grad==flag and p.grad is grad,
                    'parameter identity/shape/trainability/grad object')
            if p not in originals:
                require(p._version==version, 'base or unrelated parameter changed')
        for p, value in originals.items():
            require(torch.isfinite(p).all(), 'nonfinite factor')
            if p not in changed:
                require(torch.equal(p.detach().cpu(),value), 'unaccepted factor changed')
        require(all((layers[n].lora_rank,layers[n].lora_scaler)==ranks[n] for n in layers), 'rank/scaling')
        require(sum(p.numel() for p in params if p.requires_grad)==trainable,'parameter count')
        require(scheduler.state_dict()==scheduler_before,'scheduler state')
        require(len(groups)==len(optimizer.param_groups),'optimizer group count')
        for actual,(old, ps, options) in zip(optimizer.param_groups,groups):
            require(actual is old and len(actual['params'])==len(ps) and all(a is b for a,b in zip(actual['params'],ps))
                    and {k:v for k,v in actual.items() if k!='params'}==options, 'optimizer param groups/LR')
        require(set(optimizer.state)==set(states),'optimizer keys before commit')
        for p,(state,signature) in states.items():
            require(optimizer.state[p] is state and _state_signature(state)==signature,'optimizer state changed during trial')
        for c in candidates:
            if c['parameter'] in changed:
                existed = c['parameter'] in optimizer.state
                optimizer.state.pop(c['parameter'],None)
                c['row']['optimizer_state_reset_count'] = int(existed)
        require(set(optimizer.state)==set(states)-changed,'only accepted states removed')
        summary.update(accepted=bool(changed), num_modules=len(rows), num_direction_qualified=len(qualified),
            applied_sides=dict(Counter(r['applied_side'] for r in rows)),
            reasons=dict(Counter(r['reason'] for r in rows)),
            optimizer_state_reset_count=sum(r['optimizer_state_reset_count'] for r in rows),
            new_total_lora_params=trainable, parameter_objects_preserved=True, rank_scaling_preserved=True,
            base_weights_unchanged=True, optimizer_object_preserved=True, untouched_optimizer_states_preserved=True,
            scheduler_preserved=True, scheduler_last_epoch=scheduler.last_epoch,
            learning_rates=[g['lr'] for g in optimizer.param_groups], training_state_restored=True,
            elapsed_seconds=time.perf_counter()-start)
        with (out/'modules.jsonl').open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    except BaseException as error:
        # Trial/commit/logging failure is an abort, never a partial accepted event.
        with torch.no_grad():
            for p,value in originals.items():
                if not torch.equal(p.detach().cpu(),value):
                    p.copy_(value.to(p))
        for p,(state,_) in states.items():
            if p not in optimizer.state:
                optimizer.state[p] = state
        summary.update(error=repr(error), aborted=True, accepted=False, factors_rolled_back=True,
                       optimizer_state_not_committed=True, optimizer_state_reset_count=0)
        for row in rows:
            row.update(applied_side='none', applied_strength=0., optimizer_state_reset_count=0,
                       reason='event_aborted: '+repr(error))
        summary['module_abort_diagnostics'] = rows
        try:
            (out/'error.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        except Exception:
            # A failed filesystem must not mask the original transactional failure.
            pass
        raise
    print(f'RSC step {step}: qualified={len(qualified)}, committed={len(changed)}',flush=True)
    return summary
