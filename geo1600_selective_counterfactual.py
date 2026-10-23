#!/usr/bin/env python3
"""Read-only, fixed-policy counterfactual at the saved step-1600 LoRA event.

CPU-only by default: choose predetermined layer subsets from a prior BF16
first-order diagnostic; do not train or touch the original checkpoints.
Optional GPU: load the exact PRE checkpoint, apply selected exact POST factors,
and compare eval-mode loss on proposal (0..7), selection (8..15) and fresh
train-only audit indices randomly drawn from the remaining training examples.
The selection set is NOT an independent validation set once used to gate layers.
The audit set must never be used to adapt policies or thresholds.

Requires audit_geo_step1600.py alongside this script, and original MiniCPM
repository on PYTHONPATH. Outputs to a NEW directory only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

POLICIES = ('none','all','both_down','proposal_down','selection_down','a_only','b_only')


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):
            h.update(buf)
    return h.hexdigest()


def select_policies(rows):
    active={r['module']:r for r in rows if r['actual_side'] in ('A','B')}
    return {
        'none':set(),
        'all':set(active),
        'both_down':{n for n,r in active.items() if r['proposal_slope']<0 and r['selection_slope']<0},
        'proposal_down':{n for n,r in active.items() if r['proposal_slope']<0},
        'selection_down':{n for n,r in active.items() if r['selection_slope']<0},
        'a_only':{n for n,r in active.items() if r['actual_side']=='A'},
        'b_only':{n for n,r in active.items() if r['actual_side']=='B'},
    }


def load_slope_rows(path, expected_names):
    with path.open(newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=len(expected_names):
        raise ValueError('Slope CSV does not cover the event module count')
    seen=set(); parsed=[]
    for row in rows:
        name=row.get('module')
        if name not in expected_names or name in seen:
            raise ValueError('Unexpected or duplicate layer in slope CSV: '+str(name))
        seen.add(name)
        side=row['actual_side']
        if side not in ('A','B','none'):
            raise ValueError('Unexpected actual_side: '+str(side))
        proposal=float(row['proposal_first_order_dL_dt'])
        selection=float(row['acceptance_first_order_dL_dt'])
        if not all(math.isfinite(x) for x in (proposal,selection)):
            raise ValueError('Non-finite gradient contribution')
        parsed.append(dict(module=name,actual_side=side,proposal_slope=proposal,
                           selection_slope=selection,rank=int(row['rank']),
                           tangent_gain=float(row['tangent_gain'])))
    if seen != set(expected_names):
        raise ValueError('Missing slope rows')
    return parsed


def tabulate_policies(rows, selections):
    index={r['module']:r for r in rows}
    result={}
    for name,selected in selections.items():
        included=[index[k] for k in sorted(selected)]
        result[name]=dict(
            n_layers=len(included),
            sides=dict(Counter(r['actual_side'] for r in included)),
            proposal_first_order_sum=math.fsum(r['proposal_slope'] for r in included),
            selection_first_order_sum=math.fsum(r['selection_slope'] for r in included),
            module_names=sorted(selected))
    return result


def self_test():
    fake=[{'module':n,'actual_side':side,'proposal_slope':p,'selection_slope':s,
           'rank':1,'tangent_gain':0.1} for n,side,p,s in [
             ('a','A',-1.,-1.), ('b','B',-1.,+1.),
             ('c','A',+1.,-1.),('d','B',+1.,+1.), ('e','none',0.,0.)]]
    p=select_policies(fake)
    assert p['none']==set() and p['all']=={'a','b','c','d'}
    assert p['both_down']=={'a'} and p['proposal_down']=={'a','b'}
    assert p['selection_down']=={'a','c'}
    assert p['a_only']=={'a','c'} and p['b_only']=={'b','d'}
    t=tabulate_policies(fake,p)
    assert t['both_down']['proposal_first_order_sum']==-1
    print('SELF-TEST PASS: all 7 policies, deterministic counts and slopes')


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--run',required=True,type=Path)
    pa.add_argument('--step',type=int,default=1600)
    pa.add_argument('--slopes',required=True,type=Path,help='modules.csv from the GPU audit')
    pa.add_argument('--output',required=True,type=Path,help='New folder, never overwritten')
    pa.add_argument('--gpu',action='store_true',help='Run no-grad forward-only counterfactual')
    pa.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    pa.add_argument('--device',default='cuda')
    pa.add_argument('--selection-start',type=int,default=8)
    pa.add_argument('--selection-count',type=int,default=8)
    pa.add_argument('--audit-count',type=int,default=32)
    pa.add_argument('--audit-seed',type=int,default=20261013)
    pa.add_argument('--policies',nargs='+',choices=POLICIES,default=list(POLICIES))
    pa.add_argument('--self-test',action='store_true')
    args=pa.parse_args()
    if args.self_test:
        self_test();return
    if args.output.exists():
        pa.error('Output already exists: '+str(args.output))
    if args.selection_count<1 or args.audit_count<1 or args.selection_start<0:
        pa.error('Selection / audit indices must be valid positive counts')
    if len(args.policies)!=len(set(args.policies)) or 'none' not in args.policies or 'all' not in args.policies:
        pa.error('Policies must be unique and contain both none and all')

    from audit_geo_step1600 import event_cpu, batch_dataset_indices, sha256
    cpu,detail,delta,pre,post,cfg=event_cpu(args.run,args.step)
    if not cpu['invariants_passed']:
        raise RuntimeError('CPU invariants failed: '+str(cpu['invariant_issues']))
    rows=load_slope_rows(args.slopes,set(pre['layers']))
    by_module={r['module']:r for r in rows}
    for name in pre['layers']:
        if (name in delta)!=(by_module[name]['actual_side'] in ('A','B')):
            raise RuntimeError('Slope CSV inconsistent with saved checkpoint sides: '+name)
        if name in delta and delta[name]['side']!=by_module[name]['actual_side']:
            raise RuntimeError('Slope CSV mismatched factor side for '+name)
    policies=select_policies(rows)
    counts=tabulate_policies(rows,policies)
    print('=== CPU COUNTERFACTUAL MASKS ===',flush=True)
    for name in args.policies:
        info=counts[name]
        print(f'{name:17s} layers={info["n_layers"]:2d} sides={info["sides"]} '
              f'proposal_dL={info["proposal_first_order_sum"]:+.9f} '
              f'selection_dL={info["selection_first_order_sum"]:+.9f}',flush=True)
    report=dict(step=args.step,run_dir=str(args.run.resolve()),
        pre_sha256=cpu['checkpoint_pre_sha256'],post_sha256=cpu['checkpoint_post_sha256'],
        slope_csv_sha256=file_hash(args.slopes),source_grads='BF16 PRE checkpoint, '
        'proposal samples from the original first-eight calibration; selection samples 8..15',
        selection_caution='The eight selection examples are now used by the mask selection policy, '
        'so they are not an independent evaluation set.',
        audit_caution='Additional audit examples are training data, not a dev/test set. '
        'Do not use their results to pick a new mask or tune a threshold.',
        policies={k:counts[k] for k in args.policies},gpu=False)

    if args.gpu:
        import torch
        from transformers import AutoTokenizer
        from adaptive_lora.model_loading import load_training_model
        from adaptive_lora.checkpoint import load_adapter
        from adaptive_lora.training_data import ChatTrainingDataset
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        if args.dtype=='float32' and args.device.startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
        data=Path(cfg['train_data_path'])
        if sha256(data)!=cfg['train_data_sha256']:
            raise ValueError('Training dataset has changed from original run')
        model_path=Path(cfg['model_name_or_path'])
        tokenizer=AutoTokenizer.from_pretrained(str(model_path),trust_remote_code=True,local_files_only=True)
        if tokenizer.chat_template is None:
            tokenizer.chat_template=(model_path/'chat_template.jinja').read_text()
        tokenizer.pad_token=tokenizer.eos_token
        dataset=ChatTrainingDataset(data,tokenizer,max_length=int(cfg['model_max_length']),dynamic_length=True)
        original=batch_dataset_indices(args.run/'calibration_indices.json',
                                       cfg.get('calibration_indices_sha256'),len(dataset))
        selection=list(range(args.selection_start,args.selection_start+args.selection_count))
        if max(selection)>=len(dataset) or set(selection)&set(original):
            raise ValueError('Selection examples overlap proposal or out of range')
        exclusion=set(original)|set(selection)
        candidates=[i for i in range(len(dataset)) if i not in exclusion]
        if args.audit_count>len(candidates):
            raise ValueError('Not enough disjoint training examples for audit')
        audit_ids=sorted(random.Random(args.audit_seed).sample(candidates,args.audit_count))
        assert not (set(audit_ids)&exclusion)
        groups={'proposal':original,'selection':selection,'fresh_training_audit':audit_ids}
        print('Fresh train-only audit indices:',audit_ids,flush=True)
        print(f'Loading PRE base model, dtype={args.dtype}, forward-only...',flush=True)
        model=load_training_model(str(model_path),getattr(torch,args.dtype),cfg['model_family']).to(args.device)
        model.config.use_cache=False
        load_adapter(model,args.run/'correction_checkpoints'/f'step_{args.step}_pre.pt')
        model.eval()
        # Only the changed factor is ever written. No optimizer, no backward, no training.
        changed={}
        for name in delta:
            side=delta[name]['side']
            key='weight_a' if side=='A' else 'weight_b'
            parameter=getattr(model.get_submodule(name),key)
            old=pre['layers'][name]['state'][key].to(device=parameter.device,dtype=parameter.dtype)
            new=post['layers'][name]['state'][key].to(device=parameter.device,dtype=parameter.dtype)
            if torch.equal(old,new):
                raise RuntimeError('Marked changed factor identical: '+name)
            changed[name]=(parameter,old,new)
        def activate(selected):
            with torch.no_grad():
                for name,(param,old,new) in changed.items():
                    param.copy_(new if name in selected else old)
        def score_index(i):
            inp={k:v.unsqueeze(0).to(args.device) for k,v in dataset[i].items()}
            with torch.no_grad():
                loss=model(**inp).loss
            if loss.numel()!=1 or not torch.isfinite(loss).all():
                raise ValueError('Invalid loss at sample '+str(i))
            return float(loss)
        all_outputs={}
        activate(set())
        try:
            for policy in args.policies:
                activate(policies[policy])
                group_result={}
                for group,ids in groups.items():
                    losses=[score_index(i) for i in ids]
                    avg=math.fsum(losses)/len(losses)
                    group_result[group]=dict(indices=ids,mean_loss=avg,losses=losses)
                    print(f'{policy:17s} {group:20s} mean_loss={avg:.9f}',flush=True)
                all_outputs[policy]=group_result
        finally:
            activate(set())
        baseline=all_outputs['none']
        repeat={group: [score_index(i) for i in ids] for group,ids in groups.items()}
        repeat_max={group:max(abs(x-y) for x,y in zip(baseline[group]['losses'],repeat[group]))
                    for group in groups}
        if any(x>1e-7 for x in repeat_max.values()):
            raise RuntimeError('Baseline changed after masks: '+str(repeat_max))
        for policy in all_outputs:
            for group in groups:
                s=all_outputs[policy][group]
                s['delta_from_none']=s['mean_loss']-baseline[group]['mean_loss']
        report.update(gpu=True,dtype=args.dtype,original_train_dtype=cfg['dtype'],
            dataset_sha256=cfg['train_data_sha256'],
            evaluation='eval-mode no-grad, mean of per-row teacher-forced training loss',
            no_optimizer=True,no_training_steps=True,
            group_indices=groups,repeat_none_max_abs_loss_difference=repeat_max,
            losses=all_outputs)
        print('\n=== INDEPENDENT TRAINING AUDIT (not dev) ===')
        for policy in args.policies:
            print(f'{policy:17s} '+ ' '.join(f'{grp}={all_outputs[policy][grp]["delta_from_none"]:+.8f}'
                for grp in groups),flush=True)
        print('Repeated none max abs differences:',repeat_max,flush=True)

    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'counterfactual.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.gpu:
        with (args.output/'losses.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['policy','group','index','loss','mean_loss','delta_from_none'])
            writer.writeheader()
            for policy,policy_groups in report['losses'].items():
                for group,r in policy_groups.items():
                    for i,loss in zip(r['indices'],r['losses']):
                        writer.writerow(dict(policy=policy,group=group,index=i,loss=loss,
                                             mean_loss=r['mean_loss'],delta_from_none=r['delta_from_none']))
    print('\nSaved:',args.output,flush=True)

if __name__=='__main__':
    main()
