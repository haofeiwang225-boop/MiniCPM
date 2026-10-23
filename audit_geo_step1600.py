#!/usr/bin/env python3
"""Forensic audit of a saved fixed-rank geometric LoRA correction event.

CPU mode: exact checkpoint-difference audit, NO base model / gradients required.
GPU mode: additionally replay pre/post eval losses and pre-event LoRA-factor
          directional derivatives on original proposal and disjoint train batches.
          NO optimizer, NO weight updates, NO re-training.

Run at MiniCPM repository root with PYTHONPATH=$PWD:$PWD/finetune.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_torch():
    import torch
    return torch


def norm_product(b, a):
    """Frobenius ||B A|| without allocating the ambient matrix (FP64)."""
    import torch
    b, a = b.double(), a.double()
    # tr((B^T B)(A A^T)); both Gramians are r x r.
    bb = b.T @ b
    aa = a @ a.T
    v = torch.sum(bb * aa.T)
    return float(torch.clamp(v, min=0).sqrt())


def validate_rank_config(pre, post, name):
    for field in ('config', 'structure', 'scaling'):
        if pre.get(field) != post.get(field):
            raise ValueError(f'{name}: immutable {field} differs pre/post')
    if pre.get('structure') != 'dense':
        raise ValueError(f'{name}: only ordinary Dense LoRA supported')


def event_cpu(run: Path, step: int):
    torch = load_torch()
    before = run / 'correction_checkpoints' / f'step_{step}_pre.pt'
    after = run / 'correction_checkpoints' / f'step_{step}_post.pt'
    rows_path = run / 'corrections' / f'step_{step}' / 'modules.jsonl'
    summary_path = rows_path.with_name('summary.json')
    config_path = run / 'run_config.json'
    for p in (before, after, rows_path, summary_path, config_path):
        if not p.is_file():
            raise FileNotFoundError(f'Required original artifact not found: {p}')
    pre = torch.load(before, map_location='cpu', weights_only=True)
    post = torch.load(after, map_location='cpu', weights_only=True)
    if pre['format_version'] != post['format_version'] or pre['base_model'] != post['base_model']:
        raise ValueError('Base model / checkpoint format changed')
    a_layers, b_layers = pre['layers'], post['layers']
    if not a_layers or a_layers.keys() != b_layers.keys():
        raise ValueError('Pre/post adapter module sets differ')
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    by_name = {r['module']: r for r in rows}
    if len(by_name) != len(rows) or by_name.keys() != a_layers.keys():
        raise ValueError('Event diagnostics and checkpoint module names disagree')
    saved_summary = read_json(summary_path)
    run_config = read_json(config_path)
    if run_config.get('direction_correction') != 'geometric':
        raise ValueError('Run is not geometric correction')
    if step not in run_config.get('direction_correction_steps', []):
        raise ValueError('Event step not present in run configuration')
    issues, enriched = [], []
    delta_factors = {}
    side_counts = Counter()
    absolute_energy = []
    slope_gains = []
    total_parameters = 0
    for name in a_layers:
        old, new, diag = a_layers[name], b_layers[name], by_name[name]
        validate_rank_config(old, new, name)
        if old['state'].keys() != new['state'].keys():
            raise ValueError(f'{name}: adapter state keys differ')
        if set(old['state']) != {'weight_a', 'weight_b'}:
            raise ValueError(f'{name}: unexpected adapter state keys: {list(old["state"])}')
        a0, b0 = old['state']['weight_a'], old['state']['weight_b']
        a1, b1 = new['state']['weight_a'], new['state']['weight_b']
        if a0.shape != a1.shape or b0.shape != b1.shape:
            raise ValueError(f'{name}: factor shape differs')
        r = a0.shape[0]
        if a0.ndim != 2 or b0.ndim != 2 or b0.shape[1] != r:
            raise ValueError(f'{name}: invalid factors')
        if not all(torch.isfinite(t).all().item() for t in (a0, a1, b0, b1)):
            raise ValueError(f'{name}: nonfinite factors')
        changed_a = not torch.equal(a0, a1)
        changed_b = not torch.equal(b0, b1)
        actual_side = 'both' if changed_a and changed_b else 'A' if changed_a else 'B' if changed_b else 'none'
        planned_side = diag['applied_side']
        side_counts[actual_side] += 1
        if planned_side != actual_side:
            issues.append(f'{name}: planned {planned_side}, actual {actual_side}')
        if actual_side == 'both':
            issues.append(f'{name}: BOTH factors changed (violates fixed-side policy)')
            # Exact difference B1 A1 - B0 A0 represented by concatenated rank-2r factors.
            db = torch.cat([b1.double(), -b0.double()], dim=1)
            da = torch.cat([a1.double(), a0.double()], dim=0)
        elif actual_side == 'A':
            db, da = b0, a1.double()-a0.double()
        elif actual_side == 'B':
            db, da = b1.double()-b0.double(), a0
        else:
            db, da = b0, torch.zeros_like(a0)
        dnorm = norm_product(db, da)
        base_norm = norm_product(b0, a0)
        relative = dnorm/(base_norm+1e-12)
        scaling = float(old['scaling'])
        changed_norm = abs(scaling)*dnorm
        log_relative = float(diag.get('applied_relative_update_change', 0.0))
        log_absolute = float(diag.get('absolute_scaled_delta_change', 0.0))
        # Float32 persisted factors may differ slightly from float64 planning tensors.
        if not math.isclose(relative, log_relative, rel_tol=0.03, abs_tol=3e-5):
            issues.append(f'{name}: relative update mismatch checkpoint={relative:.9g} log={log_relative:.9g}')
        if not math.isclose(changed_norm, log_absolute, rel_tol=0.03, abs_tol=3e-5):
            issues.append(f'{name}: absolute scaled norm mismatch checkpoint={changed_norm:.9g} log={log_absolute:.9g}')
        if relative > float(run_config['direction_function_tolerance']) + 2e-4:
            issues.append(f'{name}: trust region violated: relative={relative}')
        if diag.get('rank_after') != r or diag.get('rank_before') != r:
            issues.append(f'{name}: logged rank disagrees with factor rank {r}')
        if diag.get('optimizer_state_reset_count') not in ((1,) if actual_side != 'none' else (0,)):
            issues.append(f'{name}: optimizer reset log inconsistent with actual side')
        gain = float(diag['tangent_coverage_after'])-float(diag['tangent_coverage_before'])
        gradient_energy = float(diag['gradient_fro_norm'])**2
        total_parameters += a0.numel()+b0.numel()
        row = {
            'module':name, 'rank':r, 'planned_side':planned_side, 'actual_side':actual_side,
            'gradient_fro_norm':float(diag['gradient_fro_norm']),
            'gradient_energy':gradient_energy,
            'direction_gap':float(diag['direction_gap']),
            'capacity_gap':float(diag['capacity_gap']),
            'tangent_coverage_before':float(diag['tangent_coverage_before']),
            'tangent_coverage_after':float(diag['tangent_coverage_after']),
            'tangent_gain':gain,
            'eta':diag.get('applied_eta'),
            'relative_weight_update':relative,
            'scaled_delta_weight_norm':changed_norm,
            'old_scaled_weight_norm':abs(scaling)*base_norm,
            'optimizer_state_reset_count':diag.get('optimizer_state_reset_count'),
        }
        enriched.append(row)
        absolute_energy.append(changed_norm**2)
        slope_gains.append(gradient_energy*gain)
        if actual_side in ('A','B'):
            delta_factors[name] = dict(side=actual_side,
                                       delta=(a1-a0 if actual_side == 'A' else b1-b0))
    etotal = math.fsum(r['gradient_energy'] for r in enriched)
    global_gain = math.fsum(slope_gains)/(etotal+1e-30) if etotal else 0.0
    logged_gain = float(saved_summary['global_tangent_coverage_gain'])
    if not math.isclose(global_gain, logged_gain, rel_tol=1e-5, abs_tol=1e-7):
        issues.append(f'Global coverage gain mismatch: {global_gain} vs {logged_gain}')
    if saved_summary['applied_sides'] != dict(Counter(r['planned_side'] for r in enriched)):
        issues.append('Logged applied side counter inconsistent with modules')
    if saved_summary['old_total_lora_params'] != total_parameters or saved_summary['new_total_lora_params'] != total_parameters:
        issues.append('Logged total trainable parameters mismatch actual factors')
    if saved_summary['optimizer_state_reset_count'] != sum(r['optimizer_state_reset_count'] for r in enriched):
        issues.append('Logged optimizer-state reset total inconsistent')
    sorted_gradient = sorted(enriched, key=lambda x:x['gradient_energy'],reverse=True)
    sorted_delta = sorted(enriched,key=lambda x:x['scaled_delta_weight_norm']**2,reverse=True)
    delta_energy = math.fsum(absolute_energy)
    audit = {
        'run_dir':str(run.resolve()), 'step':step, 'mode':'cpu',
        'checkpoint_pre_sha256':sha256(before), 'checkpoint_post_sha256':sha256(after),
        'diagnostic_modules_sha256':sha256(rows_path),
        'num_modules':len(enriched), 'total_lora_parameters':total_parameters,
        'actual_changed_sides':dict(side_counts),
        'logged_changed_sides':saved_summary['applied_sides'],
        'logged_immediate_loss_change':saved_summary.get('immediate_loss_change'),
        'global_tangent_coverage_gain_recomputed':global_gain,
        'global_tangent_coverage_gain_logged':logged_gain,
        'update_energy_top10_share':sum(r['scaled_delta_weight_norm']**2 for r in sorted_delta[:10])/(delta_energy+1e-30),
        'gradient_energy_top10_share':sum(r['gradient_energy'] for r in sorted_gradient[:10])/(etotal+1e-30),
        'weighted_mean_relative_weight_update':math.fsum(r['gradient_energy']*r['relative_weight_update'] for r in enriched)/(etotal+1e-30),
        'max_relative_weight_update':max(r['relative_weight_update'] for r in enriched),
        'top10_by_delta_weight_norm':[r['module'] for r in sorted_delta[:10]],
        'top10_by_gradient_energy':[r['module'] for r in sorted_gradient[:10]],
        'invariant_issues':issues, 'invariants_passed':not issues,
        'first_order_sign_known_from_existing_logs':False,
        'why_first_order_sign_unknown':'G direction / LoRA-factor gradients were not persisted in step_1600 artifacts',
    }
    return audit, enriched, delta_factors, pre, post, run_config


def batch_dataset_indices(manifest_path, expected_sha, size):
    if expected_sha is not None and sha256(manifest_path) != expected_sha:
        raise ValueError('Original calibration manifest SHA-256 mismatch')
    payload = read_json(manifest_path)
    if payload['split'] != 'train' or payload['dataset_length'] != size:
        raise ValueError('Incorrect calibration dataset provenance')
    groups = payload['batch_indices']
    if not groups or any(len(group) != 1 for group in groups):
        raise ValueError('Replay currently supports original batch_size=1 only')
    flat = [i for group in groups for i in group]
    if len(set(flat)) != len(flat) or any(not 0 <= i < size for i in flat):
        raise ValueError('Invalid calibration indices')
    return flat


def event_gpu(run: Path, step: int, audit, details, deltas, pre, post, config, heldout_start, heldout_count, device):
    import torch
    from transformers import AutoTokenizer, default_data_collator
    from adaptive_lora.model_loading import load_training_model
    from adaptive_lora.checkpoint import load_adapter
    from adaptive_lora.training_data import ChatTrainingDataset

    if not torch.cuda.is_available() and device.startswith('cuda'):
        raise RuntimeError('GPU replay requires CUDA on this node; CPU checkpoint audit already works')
    data_path = Path(config['train_data_path'])
    if not data_path.is_file():
        raise FileNotFoundError(f'Train data unavailable; invoke from repo root: {data_path}')
    expected_data_sha = config.get('train_data_sha256')
    if expected_data_sha and sha256(data_path) != expected_data_sha:
        raise ValueError('Train data SHA-256 differs from original run')
    model_path = config['model_name_or_path']
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (Path(model_path)/'chat_template.jinja').read_text()
    tokenizer.pad_token = tokenizer.eos_token
    dataset = ChatTrainingDataset(data_path,tokenizer,max_length=int(config['model_max_length']),dynamic_length=True)
    calibration_path = run/'calibration_indices.json'
    proposal = batch_dataset_indices(calibration_path,config.get('calibration_indices_sha256'),len(dataset))
    acceptance = list(range(heldout_start,heldout_start+heldout_count))
    if not acceptance or max(acceptance)>=len(dataset) or set(proposal)&set(acceptance):
        raise ValueError('Held-out indices unavailable or overlap original proposal samples')
    if len(set(acceptance)) != len(acceptance):
        raise ValueError('Duplicate acceptance indices')
    dtype = getattr(torch,config['dtype'])
    print('Loading original model and PRE adapter (no training)...',flush=True)
    model = load_training_model(model_path,dtype,config['model_family']).to(device)
    model.config.use_cache = False
    load_adapter(model,run/'correction_checkpoints'/f'step_{step}_pre.pt')
    model.eval()
    changes = {}
    parameters = {}
    for name, item in deltas.items():
        layer = model.get_submodule(name)
        param = layer.weight_a if item['side']=='A' else layer.weight_b
        parameters[name] = param
        changes[name] = item['delta'].to(device=param.device,dtype=param.dtype)
        if not param.requires_grad:
            raise RuntimeError(f'Missing LoRA gradient flag: {name}')

    def one_batch(index, backward):
        batch = {k:v.unsqueeze(0).to(device) for k,v in dataset[index].items()}
        model.zero_grad(set_to_none=True)
        if backward:
            with torch.enable_grad():
                loss = model(**batch).loss
                loss.backward()
                signed = {}
                for name,p in parameters.items():
                    if p.grad is None:
                        raise RuntimeError(f'Gradient missing for {name}')
                    signed[name] = float((p.grad.double()*changes[name].double()).sum())
            slope = math.fsum(signed.values())
        else:
            with torch.no_grad():
                loss = model(**batch).loss
            signed, slope = {}, None
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite loss at train index {index}')
        value = float(loss.detach())
        model.zero_grad(set_to_none=True)
        return value,slope,signed

    out = {}
    before = {}
    per_module = {}
    for group, ids in [('proposal',proposal),('acceptance',acceptance)]:
        losses, slopes = [], []
        all_per_layer = {n:[] for n in parameters}
        for i in ids:
            loss, slope, signed = one_batch(i,True)
            losses.append(loss);slopes.append(slope)
            for name,value in signed.items():
                all_per_layer[name].append(value)
            print(f'  PRE {group} row {i}: loss={loss:.8g}, directional_dL/dt={slope:.8g}',flush=True)
        before[group] = losses
        per_module[group] = {n:sum(values)/len(values) for n,values in all_per_layer.items()}
        out[group] = dict(indices=ids,pre_losses=losses,pre_mean_loss=sum(losses)/len(losses),
                          per_sample_directional_derivatives=slopes,
                          first_order_global_prediction=sum(slopes)/len(slopes),
                          number_layers_with_negative_slope=sum(v<0 for v in per_module[group].values()),
                          number_layers_with_positive_slope=sum(v>0 for v in per_module[group].values()))

    print('Applying POST factors in-memory (optimizer does not exist)...',flush=True)
    with torch.no_grad():
        for name,spec in post['layers'].items():
            layer = model.get_submodule(name)
            for key in ('weight_a','weight_b'):
                p = getattr(layer,key)
                val = spec['state'][key]
                p.copy_(val.to(device=p.device,dtype=p.dtype))
    for group,ids in [('proposal',proposal),('acceptance',acceptance)]:
        post_losses = []
        for i in ids:
            loss,_,_=one_batch(i,False)
            post_losses.append(loss)
            print(f'  POST {group} row {i}: loss={loss:.8g}',flush=True)
        out[group]['post_losses'] = post_losses
        out[group]['post_mean_loss'] = sum(post_losses)/len(post_losses)
        out[group]['actual_loss_change'] = out[group]['post_mean_loss']-out[group]['pre_mean_loss']
        out[group]['first_order_prediction_matches_actual_change_sign'] = ((out[group]['first_order_global_prediction']<0)==(out[group]['actual_loss_change']<0))
    out.update(model_path=model_path,train_sha256=sha256(data_path),
               gradient_interpretation='exact first derivative along pre->post LoRA-factor interpolation at the PRE checkpoint, dropout off; no full base-weight gradients needed',
               interpolation_note='first-order slope is local; actual post-pre loss includes nonlinear and multi-layer interactions',
               optimizer_state='not loaded or modified',training_steps_run=0)
    for row in details:
        name=row['module']
        row['proposal_first_order_dL_dt'] = per_module['proposal'].get(name,0.0)
        row['acceptance_first_order_dL_dt'] = per_module['acceptance'].get(name,0.0)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True,help='e.g. finetune/adaptive_lora/outputs/metamath_pilot_geo1600_2000_seed42')
    parser.add_argument('--step',type=int,default=1600)
    parser.add_argument('--output',type=Path,required=True,help='NEW output dir; never overwrite previous audits')
    parser.add_argument('--gpu',action='store_true',help='Also replay pre/post eval loss and first-order derivatives on GPU')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--heldout-start',type=int,default=8)
    parser.add_argument('--heldout-count',type=int,default=8)
    args=parser.parse_args()
    audit, details, delta, pre, post,config=event_cpu(args.run,args.step)
    print('\n=== CHECKPOINT AUDIT (CPU) ===',flush=True)
    for key in ('num_modules','actual_changed_sides','global_tangent_coverage_gain_recomputed',
                'logged_immediate_loss_change','max_relative_weight_update','update_energy_top10_share',
                'invariants_passed','invariant_issues'):
        print(f'{key}: {audit[key]}',flush=True)
    replay=None
    if args.gpu:
        replay=event_gpu(args.run,args.step,audit,details,delta,pre,post,config,
                         args.heldout_start,args.heldout_count,args.device)
        print('\n=== GPU DIRECTIONAL REPLAY ===',flush=True)
        for k in ('proposal','acceptance'):
            print(k,{x:replay[k][x] for x in ('pre_mean_loss','post_mean_loss',
                   'actual_loss_change','first_order_global_prediction',
                   'first_order_prediction_matches_actual_change_sign')},flush=True)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'cpu_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if replay is not None:
        (args.output/'gpu_replay.json').write_text(json.dumps(replay,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (args.output/'modules.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(details[0]))
        writer.writeheader();writer.writerows(details)
    print(f'\nAudit saved to: {args.output}',flush=True)
    if audit['invariant_issues']:
        print('WARNING: checkpoint invariants failed; inspect audit before causal interpretation.',file=sys.stderr)
        sys.exit(2)

if __name__=='__main__':
    main()
