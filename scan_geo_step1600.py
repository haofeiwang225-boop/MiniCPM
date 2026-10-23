#!/usr/bin/env python3
"""Read-only loss-path scan for a saved geometric LoRA correction.

Run from the MiniCPM repo root, alongside audit_geo_step1600.py.
No training, optimizer, gradients, or modifications to existing checkpoints.
Uses train-only proposal/acceptance indices verified against the original run.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--step', type=int, default=1600)
    p.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite')
    p.add_argument('--heldout-start', type=int, default=8)
    p.add_argument('--heldout-count', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--t-values', nargs='+', type=float,
                   default=[0, .03125, .0625, .125, .25, .5, .75, 1.0])
    args = p.parse_args()
    if args.output.exists():
        p.error(f'Output directory already exists: {args.output}')
    ts = args.t_values
    if not ts or ts != sorted(set(ts)) or ts[0] != 0 or ts[-1] != 1 or any(not 0 <= t <= 1 for t in ts):
        p.error('t-values must be unique ascending in [0,1], including 0 and 1')
    if args.heldout_count < 1:
        p.error('heldout-count must be positive')

    from audit_geo_step1600 import event_cpu, batch_dataset_indices, sha256
    import torch
    from transformers import AutoTokenizer
    from adaptive_lora.model_loading import load_training_model
    from adaptive_lora.checkpoint import load_adapter
    from adaptive_lora.training_data import ChatTrainingDataset

    audit, details, deltas, pre, post, cfg = event_cpu(args.run, args.step)
    if not audit['invariants_passed']:
        raise RuntimeError(f'Existing CPU audit invariant violations: {audit["invariant_issues"]}')
    dataset_path = Path(cfg['train_data_path'])
    if sha256(dataset_path) != cfg['train_data_sha256']:
        raise ValueError('Training dataset SHA256 differs from original run')
    model_path = Path(cfg['model_name_or_path'])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (model_path/'chat_template.jinja').read_text()
    tokenizer.pad_token = tokenizer.eos_token
    dataset = ChatTrainingDataset(dataset_path, tokenizer,
                                  max_length=int(cfg['model_max_length']), dynamic_length=True)
    proposal = batch_dataset_indices(args.run/'calibration_indices.json',
                                     cfg.get('calibration_indices_sha256'), len(dataset))
    acceptance = list(range(args.heldout_start, args.heldout_start+args.heldout_count))
    if len(set(acceptance)) != len(acceptance) or min(acceptance)<0 or max(acceptance)>=len(dataset):
        raise ValueError('Invalid acceptance indices')
    if set(proposal) & set(acceptance):
        raise ValueError('Acceptance overlaps proposal; refusing to scan')
    groups = {'proposal':proposal, 'acceptance':acceptance}
    print('Loading PRE checkpoint and base model (forward only)...',flush=True)
    model = load_training_model(str(model_path),getattr(torch,cfg['dtype']),cfg['model_family']).to(args.device)
    model.config.use_cache = False
    load_adapter(model,args.run/'correction_checkpoints'/f'step_{args.step}_pre.pt')
    model.eval()

    changed=[]
    for name in pre['layers']:
        s0=pre['layers'][name]['state']
        s1=post['layers'][name]['state']
        layer=model.get_submodule(name)
        for key in ('weight_a','weight_b'):
            p0,p1=s0[key],s1[key]
            if p0.shape!=p1.shape:
                raise ValueError('Factor shape mismatch: '+name)
            if not torch.equal(p0,p1):
                param=getattr(layer,key)
                original=p0.to(device=param.device,dtype=param.dtype)
                delta=(p1-p0).to(device=param.device,dtype=param.dtype)
                exact_post=p1.to(device=param.device,dtype=param.dtype)
                changed.append((param,original,delta,exact_post))
    if len(changed)!=sum(count for side,count in audit['actual_changed_sides'].items() if side != 'none'):
        raise RuntimeError('Unexpected factor modification count')

    def set_t(t):
        with torch.no_grad():
            for param,original,delta,exact_post in changed:
                if t == 0:
                    param.copy_(original)
                elif t == 1:
                    param.copy_(exact_post)
                else:
                    param.copy_(torch.add(original,delta,alpha=t))

    def score(i):
        row={k:v.unsqueeze(0).to(args.device) for k,v in dataset[i].items()}
        with torch.inference_mode():
            loss=model(**row).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at row {i}')
        return float(loss)

    records=[]
    try:
        for t in ts:
            set_t(t)
            for group, ids in groups.items():
                values=[score(i) for i in ids]
                m=math.fsum(values)/len(values)
                records.append(dict(t=t,group=group,mean_loss=m,indices=ids,losses=values))
                print(f't={t:0.5f} {group:10s} mean_loss={m:.9f}',flush=True)
    finally:
        set_t(0)
    baseline={group:next(row['mean_loss'] for row in records if row['t']==0 and row['group']==group)
              for group in groups}
    for row in records:
        row['delta_from_pre']=row['mean_loss']-baseline[row['group']]
    table={group:[r for r in records if r['group']==group] for group in groups}
    minima={group:min(table[group],key=lambda r:r['mean_loss']) for group in groups}
    print('\n=== SCAN SUMMARY ===')
    for group in groups:
        full=table[group][-1]
        best=minima[group]
        print(f'{group}: best grid t={best["t"]}, delta={best["delta_from_pre"]:+.9f}; '
              f'full-step delta={full["delta_from_pre"]:+.9f}',flush=True)
    report=dict(step=args.step,run=str(args.run.resolve()),model_path=str(model_path),
                calibration_indices=proposal,acceptance_indices=acceptance,
                dataset_sha256=cfg['train_data_sha256'],
                pre_sha256=audit['checkpoint_pre_sha256'],post_sha256=audit['checkpoint_post_sha256'],
                method='linear interpolation of changed LoRA factor tensors; one side per layer',
                evaluation='eval-mode no-grad train-only loss, batch size 1, per-sample arithmetic mean',
                diagnostics_only=True,optimizer=None,training_updates=0,
                best_grid_t={k:v['t'] for k,v in minima.items()},records=records)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'loss_path.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (args.output/'loss_path.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['t','group','mean_loss','delta_from_pre'])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in records)
    print('\nSaved:',args.output,flush=True)


if __name__=='__main__':
    main()
