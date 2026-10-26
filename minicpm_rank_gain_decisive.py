#!/usr/bin/env python3
"""Fixed, preregistered rank-gain audit. GPU execution requires probe-audit/train.

No imports change the original rank-gain experiment. dry-run uses CPU only.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time
import traceback

import torch
import minicpm_single_layer_rank_gain as old

ROOT = old.ROOT
SEEDS = [42, 43, 44, 45, 46]
PROTOCOL = dict(version='rank_gain_decisive_v1', steps=200, train_samples=200,
    construction_samples=32, audit_samples=32, dev_samples=400, batch_size=1,
    lr=5e-5, weight_decay=.01, dropout=False, scheduler='none', max_length=2048,
    dtype='bfloat16', added_dtype='float32', added_A_norm=1., added_scaling=1.,
    objective='sum masked token NLL / total effective tokens',
    step_objective='sample_mean_NLL * sample_tokens / mean_train_tokens',
    optimizer='fresh AdamW, default betas=(0.9,0.999), eps=1e-8; no historical state',
    endpoint='400 Dev token-weighted NLL at exactly step 200',
    min_C_minus_B_improvement=.0005, C_minus_A_noninferiority_margin=.0005,
    bootstrap_repeats=10000, bootstrap_seed=42)


def require(value, message):
    if not bool(value): raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def seed_stream(seed, name):
    return int(hashlib.sha256(f'rank_gain_decisive_v1:{seed}:{name}'.encode()).hexdigest()[:8], 16)


def split_data(train_meta, dev_meta, split_seed=20261026):
    require(len(dev_meta['samples']) == 400, 'Exactly the existing 400 Dev rows are required')
    split = old.isolated_split(train_meta, dev_meta, 200, 400, 64, split_seed)
    # Do not reshuffle or subset Dev; same source order for all ten coordinates.
    result = dict(train=split['train'], construction=split['probe'][:32], audit=split['probe'][32:], dev=list(range(400)))
    require(sum(r['family']=='GSM' for r in dev_meta['samples']) == 200 and
            sum(r['family']=='MATH' for r in dev_meta['samples']) == 200, 'Expected GSM200/MATH200 Dev')
    return result


def effective_tokens(example):
    n = int((example['labels'][1:] != -100).sum())
    require(n > 0, 'No effective shifted labels')
    return n


def weighted_probe(model, layer, examples, device):
    """Full frozen-W gradient of sum_i n_i loss_i / sum_i n_i; restore state."""
    params = list(model.parameters()); flags = [p.requires_grad for p in params]
    grads = [p.grad for p in params]; modes = [(m,m.training) for m in model.modules()]
    saved = layer.weight.detach().clone()
    counts = [effective_tokens(e) for e in examples]; total = sum(counts)
    require(total > 0, 'Empty probe')
    gradient = torch.zeros_like(layer.weight, dtype=torch.float32)
    devices = sorted({p.device.index for p in params if p.is_cuda}); rows = []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            for p in params: p.requires_grad_(False)
            layer.weight.requires_grad_(True)
            for i,(example,n) in enumerate(zip(examples,counts)):
                loss = model(**old.batch(example,device),use_cache=False).loss
                require(torch.isfinite(loss), 'Nonfinite probe loss')
                g, = torch.autograd.grad(loss,layer.weight)
                require(torch.isfinite(g).all(), 'Nonfinite full gradient')
                gradient.add_(g.float(),alpha=n/total)
                rows.append(dict(index=i,effective_tokens=n,nll=float(loss.detach()),nll_sum=float(loss.detach())*n))
    finally:
        for p,f,g in zip(params,flags,grads): p.requires_grad_(f); p.grad=g
        for m,mode in modes: m.training=mode
        require(old.tensor_exact(saved,layer.weight),'Probe changed frozen base')
    return gradient, rows


def audit_resource_gate(c_residual_score, random_p95, half_cosine):
    """Fixed heuristic resource screen; never a claim of generalization."""
    require(all(math.isfinite(x) for x in (c_residual_score,random_p95,half_cosine)), 'Nonfinite gate input')
    require(c_residual_score>=0 and random_p95>=0 and 0<=half_cosine<=1, 'Invalid gate input')
    score_pass=c_residual_score>random_p95
    cosine_pass=half_cosine>=.20
    passed=score_pass and cosine_pass
    return dict(audit_gate_pass=passed,
        audit_gate_checks=dict(C_residual_above_random_p95=score_pass,construction_half_cosine_at_least_020=cosine_pass),
        audit_gate_thresholds=dict(random_reference_count=128,quantile=.95,score_comparison='strict >',
                                   construction_half_abs_cosine_min=.20,cosine_comparison='>='),
        audit_gate_interpretation='heuristic resource screen, not a mathematical guarantee',
        audit_gate_recommendation='可供人工审核是否继续训练' if passed else '方向证据不足、暂停本方案训练',
        audit_gate_failed_checks=[name for name,ok in [('residual_score',score_pass),('half_cosine',cosine_pass)] if not ok])


def audit_reference_directions(a):
    """128 fixed independent CPU-RNG draws; no audit-gradient input."""
    reference_seed=seed_stream(20261026,'audit_random_reference_v1')
    require(old.orth(a.T).shape[1]==a.shape[0], 'Reference projection requires full-rank old A')
    qa=torch.linalg.qr(a.detach().double().T,mode='reduced')[0]
    require(qa.shape[1]<a.shape[1], 'No orthogonal complement for random references')
    generator=torch.Generator(device='cpu').manual_seed(reference_seed)
    directions=torch.randn(128,a.shape[1],dtype=torch.float64,generator=generator).to(a.device)
    directions=directions-(directions@qa)@qa.T
    norms=directions.norm(dim=1,keepdim=True)
    require(torch.isfinite(norms).all() and (norms>1e-12).all(), 'Invalid random reference projection')
    directions=directions/norms
    require(float((directions@qa).abs().max())<1e-10, 'Random references not orthogonal')
    return directions,reference_seed


def direction_audit(a,b,construction_halves,audit_gradient,random_seed):
    """Only construction data generates C. Audit scores never select candidates."""
    (g1,n1),(g2,n2) = construction_halves
    require(n1>0 and n2>0,'Empty construction half')
    full = (g1*n1+g2*n2)/(n1+n2)
    random_a,residual_a,diag = old.new_directions(a,b,full,random_seed)
    _,v1,_ = old.new_directions(a,b,g1,random_seed)
    _,v2,_ = old.new_directions(a,b,g2,random_seed)
    def score(v): return float((audit_gradient.double() @ v.double().T).square().sum())
    sb,sc=score(random_a),score(residual_a)
    cosine=float((v1.double() @ v2.double().T).abs().item()/(v1.double().norm()*v2.double().norm()))
    require(all(math.isfinite(x) for x in (sb,sc,cosine)), 'Nonfinite audit')
    cosine=min(1.,cosine)
    qa=torch.linalg.qr(a.detach().double().T,mode='reduced')[0]
    qb=torch.linalg.qr(b.detach().double(),mode='reduced')[0]
    residual_audit=old.residual_gradient(audit_gradient.double(),qa,qb)
    rb=float((residual_audit@random_a.double().T).square().sum())
    rc=float((residual_audit@residual_a.double().T).square().sum())
    references,reference_seed=audit_reference_directions(a)
    scores=(residual_audit@references.T).square().sum(dim=0)
    require(torch.isfinite(scores).all() and math.isfinite(rb) and math.isfinite(rc), 'Nonfinite residual audit score')
    median=float(torch.quantile(scores,.5)); p95=float(torch.quantile(scores,.95))
    greater=int((scores>rc).sum()); tied=int((scores==rc).sum()); lower=int((scores<rc).sum())
    gate=audit_resource_gate(rc,p95,cosine)
    return random_a,residual_a,dict(**diag,**gate,
        full_gradient_scores=dict(B=sb,C=sc,C_over_B=sc/sb if sb>0 else None,
                                  ratio_undefined_reason=None if sb>0 else 'zero_B_score'),
        residual_gradient_scores=dict(B=rb,C=rc,C_over_B=rc/rb if rb>0 else None,
                                      ratio_undefined_reason=None if rb>0 else 'zero_B_score',
                                      definition='||R_audit @ a.T||_F^2; R_audit=(I-P_B)G_audit(I-P_A)'),
        audit_random_reference=dict(seed=reference_seed,count=128,scores=scores.cpu().tolist(),
            mean=float(scores.mean()),median=median,p95=p95,quantile_interpolation='linear',
            C_descending_rank=1+greater,C_descending_rank_range=[1+greater,1+greater+tied],
            reference_scores_above_C=greater,reference_scores_equal_C=tied,reference_scores_below_C=lower,
            C_midrank_percentile=100*(lower+.5*tied)/128,
            rank_definition='C among 128 references plus C; rank 1 is best, tie range inclusive',
            max_unit_norm_error=float((references.norm(dim=1)-1).abs().max()),
            max_old_A_orthogonality_error=float((references@qa).abs().max())),
        random_score=sb,residual_score=sc,
        C_over_B_score=sc/sb if sb>0 else None,ratio_undefined_reason=None if sb>0 else 'zero_random_score',
        construction_half_abs_cosine=min(1.,cosine),half_effective_tokens=[n1,n2],
        score_definition='||G_audit @ a.T||_F^2; full audit gradient, not residual',
        audit_used_for_candidate_selection=False)


def train_arm(model,target,examples,order,device,output,source_indices):
    """One fixed permutation, 200 updates; equal-example draws estimate token risk."""
    require(len(order)==len(examples) and sorted(order)==list(range(len(examples))), 'Training must visit each example once')
    base=target.base_layer if isinstance(target,old.ExpandedLoRA) else target
    factors=dict(old_A=base.weight_a,old_B=base.weight_b)
    if isinstance(target,old.ExpandedLoRA): factors.update(new_A=target.add_a,new_B=target.add_b)
    for p in model.parameters(): p.requires_grad_(False); p.grad=None
    for p in factors.values(): p.requires_grad_(True)
    initial={k:p.detach().clone() for k,p in factors.items()}
    counts=[effective_tokens(e) for e in examples]; mean_tokens=sum(counts)/len(counts)
    optimizer=torch.optim.AdamW(list(factors.values()),lr=PROTOCOL['lr'],weight_decay=PROTOCOL['weight_decay'])
    require(not optimizer.state,'Optimizer must be fresh')
    model.eval(); rows=[]
    for step,index in enumerate(order,1):
        optimizer.zero_grad(set_to_none=True)
        loss=model(**old.batch(examples[index],device),use_cache=False).loss
        objective=loss*(counts[index]/mean_tokens)
        require(torch.isfinite(objective),'Nonfinite train objective')
        objective.backward(); norms={}
        for k,p in factors.items():
            require(p.grad is not None and torch.isfinite(p.grad).all(),f'Invalid gradient {k}')
            norms[k]=float(p.grad.float().norm())
        optimizer.step()
        require(all(torch.isfinite(p).all() for p in factors.values()),'Nonfinite factors')
        row=dict(step=step,source_index=source_indices[index],nll=float(loss.detach()),effective_tokens=counts[index],
                 nll_sum=float(loss.detach())*counts[index],backward_objective=float(objective.detach()),gradient_norms=norms,lr=PROTOCOL['lr'])
        rows.append(row)
        if output is not None:
            with Path(output).open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
    optimizer.zero_grad(set_to_none=True)
    changes={k:dict(initial_norm=float(initial[k].float().norm()),final_norm=float(p.detach().float().norm()),
                   change_norm=float((p.detach().float()-initial[k].float()).norm()),changed=not old.tensor_exact(initial[k],p)) for k,p in factors.items()}
    if 'new_A' in factors:
        require(changes['new_A']['changed'] and changes['new_B']['changed'],'New factors did not both change')
        require(any(r['gradient_norms']['new_A']>0 for r in rows[1:]) and any(r['gradient_norms']['new_B']>0 for r in rows),
                'New factors need actual LM gradients, not just weight decay')
        # Rank-one Gram identity, no d_out x d_in allocation.
        update_norm=float(factors['new_A'].detach().double().norm()*factors['new_B'].detach().double().norm())
    else: update_norm=0.
    return dict(completed_steps=len(rows),steps=rows,parameter_changes=changes,added_BA_fro_norm=update_norm,
                trainable_parameters=sum(p.numel() for p in factors.values()),mean_train_tokens=mean_tokens,
                optimizer_defaults=optimizer.defaults,historical_AdamW_restored=False)


def enriched_eval(model,examples,device,metadata,reference=None):
    result=old.evaluate(model,examples,device,reference)
    for r,m in zip(result['samples'],metadata): r['source']=m
    result['mean_per_question_nll']=math.fsum(r['nll'] for r in result['samples'])/len(result['samples'])
    return result


def prepare(args):
    """CPU-only provenance/tokenization/adapter structure checks, no model forward."""
    tm=old.build_source_metadata(args.train_data,args.source_data)
    dm=old.build_source_metadata(args.dev_data,args.source_data)
    splits=split_data(tm,dm,args.split_seed)
    streams={k:seed_stream(args.seed,k) for k in ('training_order','random_direction','probe_order')}
    order=list(range(200)); random.Random(streams['training_order']).shuffle(order)
    probe_order=list(range(32)); random.Random(streams['probe_order']).shuffle(probe_order)
    inputs=[args.adapter,args.train_data,args.dev_data,args.source_data,Path(args.model)/'config.json']
    inputs+=sorted(Path(args.model).glob('*.safetensors'))+sorted(Path(args.model).glob('*.index.json'))
    require(any(p.suffix=='.safetensors' for p in inputs),'No model safetensors')
    input_hashes={str(p.resolve()):old.sha256(p) for p in inputs}
    code_paths=[Path(__file__),ROOT/'minicpm_rank_gain_statistics.py',ROOT/'minicpm_single_layer_rank_gain.py',ROOT/'rank_gain_probe_support.py']
    code_paths += [ROOT/'finetune/adaptive_lora'/name for name in ('rank_expansion.py','subspace_matching.py','residual_gradient.py','training_data.py','data_provenance.py')]
    code_paths += [ROOT/'MyTransformers_RaLoRA_official/common/lora_modules/lora.py']
    payload=torch.load(args.adapter,map_location='cpu',weights_only=True)
    require(Path(payload['base_model']).resolve()==Path(args.model).resolve(),'Wrong adapter base')
    expected={f'model.layers.{i}.self_attn.{k}' for i in range(42) for k in ('q_proj','v_proj')}
    require(set(payload['layers'])==expected,'Expected all 84 adapters')
    for name,saved in payload['layers'].items():
        require(saved['structure']=='dense' and set(saved['state'])=={'weight_a','weight_b'},'Only original Dense adapter supported')
        a,b=saved['state']['weight_a'],saved['state']['weight_b']; cfg=saved['config']
        require(a.shape==(cfg['lora_rank'],cfg['in_features']) and b.shape==(cfg['out_features'],cfg['lora_rank']),f'Bad factors {name}')
        require(torch.isfinite(a).all() and torch.isfinite(b).all() and math.isfinite(saved['scaling']),f'Nonfinite adapter {name}')
    require(payload['layers'][args.module]['config']['lora_rank']==4,'Target must have saved rank 4')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=True)
    if tokenizer.chat_template is None: tokenizer.chat_template=(Path(args.model)/'chat_template.jinja').read_text()
    tokenizer.pad_token=tokenizer.eos_token
    datasets={}; selected_meta={}; tokens={}
    for role,indices in splits.items():
        source=args.dev_data if role=='dev' else args.train_data
        rows=json.loads(source.read_text()); selected=[rows[i] for i in indices]
        path=args.output_dir/f'{role}_selected.json'; old.write_json(path,selected)
        datasets[role]=old.ChatTrainingDataset(path,tokenizer,PROTOCOL['max_length'],dynamic_length=True)
        selected_meta[role]=[(dm if role=='dev' else tm)['samples'][i] for i in indices]
        tokens[role]=[effective_tokens(e) for e in datasets[role]]
    versions={p:importlib.metadata.version(p) for p in ('torch','transformers','safetensors','accelerate')}
    tokenizer_files={p.name:old.sha256(p) for p in Path(args.model).iterdir()
                     if p.is_file() and ('token' in p.name or 'chat_template' in p.name)}
    common=dict(protocol=PROTOCOL,split_seed=args.split_seed,indices=splits,metadata=selected_meta,
                effective_tokens=tokens,input_sha256=input_hashes,tokenizer_sha256=tokenizer_files,
                code_sha256={str(p.relative_to(ROOT)):old.sha256(p) for p in code_paths},versions=versions)
    coordinate=dict(common_fingerprint=digest(common),module=args.module,seed=args.seed,streams=streams,
                    training_order=order,probe_order=probe_order)
    plan=dict(common=common,coordinate=coordinate,fingerprint=digest(coordinate),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        caveat='Within-experiment group isolation only. PRE saw historical training data; existing Dev is not a blind test.')
    old.write_json(args.output_dir/'plan.json',plan)
    return plan,datasets,payload


def reusable_arms(source,plan):
    """Only complete, fingerprint-matched, artifact-verified arms may be skipped."""
    if source is None: return {}
    old_plan=json.loads((source/'plan.json').read_text())
    require(old_plan['fingerprint']==plan['fingerprint'] and old_plan['common']==plan['common'],'Resume protocol/data/code mismatch')
    results={}
    for arm in ('A','B','C'):
        path=source/arm/'result.json'
        if not path.exists() or not (source/arm/'artifacts.json').exists(): continue
        r=json.loads(path.read_text())
        if r.get('status')!='PASS': continue
        require(r['fingerprint']==plan['fingerprint'] and r['arm']==arm,'Resume arm mismatch')
        require(r['training']['completed_steps']==200 and r['after']['num_samples']==400,'Incomplete endpoint')
        require(r['invariants']['base_tensors_verified']>0 and all(v for k,v in r['invariants'].items()),'Failed invariants')
        artifact=json.loads((source/arm/'artifacts.json').read_text())
        require(set(artifact)=={'result.json','train.jsonl','target_factors.pt'},'Missing resume artifacts')
        require(all(old.sha256(source/arm/k)==v for k,v in artifact.items()),'Corrupted resume artifacts')
        results[arm]=r
    return results


def gpu_run(args,plan,datasets,payload,report):
    completed=reusable_arms(args.resume_from,plan) if args.mode=='train' else {}
    for arm,r in completed.items():
        shutil.copytree(args.resume_from/arm,args.output_dir/arm)
        report.setdefault('reused',{})[arm]=dict(source=str((args.resume_from/arm).resolve()),result_sha256=old.sha256(args.resume_from/arm/'result.json'))
    if len(completed)==3:
        report.update(status='PASS',all_arms_reused=True); return
    require(torch.cuda.is_available(),'CUDA unavailable')
    free,total=torch.cuda.mem_get_info(); require(free>=args.min_free_gib*2**30,'Insufficient free GPU memory')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    report['gpu']=dict(name=torch.cuda.get_device_name(),free_bytes=free,total_bytes=total,
        matmul_allow_tf32=False,cudnn_allow_tf32=False,deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    model,report['loading']=old.load_probe_model(args.model,torch.bfloat16)
    model.to('cuda'); model.config.use_cache=False
    base_params=dict(model.named_parameters())
    names=old.restore_dense_adapter_on_full_model(model,payload,ROOT)
    require(len(names)==84 and set(names)==set(payload['layers']),'Adapter restoration incomplete')
    layers={n:model.get_submodule(n) for n in names}
    factors={n:{k:getattr(l,k) for k in ('weight_a','weight_b')} for n,l in layers.items()}
    saved={n:{k:p.detach().clone() for k,p in fs.items()} for n,fs in factors.items()}
    scalings={n:l.lora_scaler for n,l in layers.items()}; ranks={n:l.lora_rank for n,l in layers.items()}
    for n in names:
        for k,v in saved[n].items(): require(old.tensor_exact(v.cpu(),payload['layers'][n]['state'][k]),f'Inexact adapter {n}.{k}')
    target=layers[args.module]; parent,_,child=args.module.rpartition('.')
    def check(frozen_target=False):
        for n,l in layers.items():
            require(l.lora_scaler==scalings[n] and l.lora_rank==ranks[n],f'Rank/scaling changed: {n}')
            for k,p in factors[n].items():
                require(getattr(l,k) is p,f'Parameter replaced: {n}.{k}')
                if n!=args.module or frozen_target: require(old.tensor_exact(p,saved[n][k]),f'Frozen factor changed: {n}.{k}')
            if n!=args.module: require(model.get_submodule(n) is l,f'Layer replaced: {n}')
        return dict(base_tensors_verified=old.verify_disk_weights(base_params,args.model),other_83_adapters_unchanged=True,
                    original_parameter_identity_preserved=True,old_rank_scaling_unchanged=True)
    try:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
        order=plan['coordinate']['probe_order']; ds=datasets['construction']
        halves=[[ds[i] for i in order[:16]],[ds[i] for i in order[16:]]]
        g1,loss1=weighted_probe(model,target,halves[0],'cuda')
        g2,loss2=weighted_probe(model,target,halves[1],'cuda')
        ga,audit_losses=weighted_probe(model,target,datasets['audit'],'cuda')
        ra,ca,audit=direction_audit(target.weight_a,target.weight_b,
            [(g1,sum(r['effective_tokens'] for r in loss1)),(g2,sum(r['effective_tokens'] for r in loss2))],ga,plan['coordinate']['streams']['random_direction'])
        del g1,g2,ga
        audit.update(construction_half_losses=[loss1,loss2],audit_losses=audit_losses,
                     construction_source_order=[plan['common']['indices']['construction'][i] for i in order],
                     audit_source_order=plan['common']['indices']['audit'],invariants=check(True))
        torch.cuda.synchronize(); audit.update(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        old.write_json(args.output_dir/'probe_audit.json',audit)
        torch.save(dict(random_A=ra.cpu(),residual_A=ca.cpu(),fingerprint=plan['fingerprint']),args.output_dir/'directions.pt')
        report['probe_audit']=audit
        if args.mode=='probe-audit': report['status']='PASS'; return
        baseline=enriched_eval(model,datasets['dev'],'cuda',plan['common']['metadata']['dev'])
        for arm,direction in [('A',None),('B',ra),('C',ca)]:
            if arm in completed: continue
            folder=args.output_dir/arm; folder.mkdir(exist_ok=False)
            setattr(model.get_submodule(parent),child,target)
            with torch.no_grad():
                for k,v in saved[args.module].items(): getattr(target,k).copy_(v)
            for p in model.parameters(): p.grad=None
            selected=target if direction is None else old.ExpandedLoRA(target,direction,torch.zeros(target.out_features,1,device='cuda'))
            setattr(model.get_submodule(parent),child,selected)
            check(True)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
            before=enriched_eval(model,datasets['dev'],'cuda',plan['common']['metadata']['dev'],baseline['samples'])
            torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
            train_start=time.perf_counter()
            training=train_arm(model,selected,datasets['train'],plan['coordinate']['training_order'],'cuda',folder/'train.jsonl',plan['common']['indices']['train'])
            torch.cuda.synchronize(); train_seconds=time.perf_counter()-train_start
            require(training['completed_steps']==200,'Primary endpoint is exactly step 200')
            after=enriched_eval(model,datasets['dev'],'cuda',plan['common']['metadata']['dev'])
            inv=check(); inv['insertion_full_logits_equal_PRE']=True
            saved_target={k:p.detach().cpu().clone() for k,p in factors[args.module].items()}
            if direction is not None: saved_target.update(add_a=selected.add_a.detach().cpu(),add_b=selected.add_b.detach().cpu())
            torch.save(dict(factors=saved_target,format='selected factors only, not PEFT/full adapter',old_scaling=scalings[args.module],added_scaling=1.),folder/'target_factors.pt')
            torch.cuda.synchronize()
            result=dict(status='PASS',arm=arm,module=args.module,seed=args.seed,fingerprint=plan['fingerprint'],
                before=before,after=after,training=training,invariants=inv,seconds=time.perf_counter()-start,train_seconds=train_seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                total_model_parameters=sum(p.numel() for p in model.parameters()),
                total_adapter_parameters=sum(v.numel() for fs in saved.values() for v in fs.values())+(0 if direction is None else selected.add_a.numel()+selected.add_b.numel()),
                old_rank=4,effective_capacity=4 if direction is None else 5,old_scaling=scalings[args.module],added_scaling=None if direction is None else 1.)
            old.write_json(folder/'result.json',result)
            old.write_json(folder/'artifacts.json',{name:old.sha256(folder/name) for name in ('result.json','train.jsonl','target_factors.pt')})
            report.setdefault('completed',[]).append(arm); old.write_json(args.output_dir/'report.json',report)
            print(f'{arm}: step 200 Dev NLL={after["token_weighted_nll"]:.9f}',flush=True)
            del selected,training; gc.collect()
        report['status']='PASS'
    finally:
        # Nothing is written to PRE or base files. Restore even an interrupted arm.
        setattr(model.get_submodule(parent),child,target)
        with torch.no_grad():
            for k,v in saved[args.module].items(): getattr(target,k).copy_(v)
        check(True)
        report['in_memory_PRE_restored']=True


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['dry-run','probe-audit','train'])
    p.add_argument('--module',choices=old.MODULES,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--split-seed',type=int,default=20261026)
    p.add_argument('--model',default=old.MODEL)
    p.add_argument('--adapter',type=Path,default=old.ADAPTER)
    p.add_argument('--train-data',type=Path,default=old.DATA/'train.json')
    p.add_argument('--dev-data',type=Path,default=old.DATA/'dev.json')
    p.add_argument('--source-data',type=Path,default=Path('/home/hongchang/whf/datasets/MetaMathQA/MetaMathQA-395K.jsonl'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resume-from',type=Path,help='Read completed arms from an old directory; new output directory required')
    p.add_argument('--min-free-gib',type=float,default=30.)
    return p


def main():
    args=parser().parse_args()
    require(args.resume_from is None or args.mode=='train','Resume is only valid in train mode')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',mode=args.mode,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},historical_AdamW_restored=False)
    try:
        plan,datasets,payload=prepare(args)
        report['fingerprint']=plan['fingerprint']
        if args.mode=='dry-run': report.update(status='PASS',cpu_only=True,checks='provenance, group isolation, labels, lengths, adapter structure; no model loading/forward')
        else: gpu_run(args,plan,datasets,payload,report)
        require(all(old.sha256(p)==h for p,h in plan['common']['input_sha256'].items()),'Input file changed')
        report['input_files_unchanged']=True
    except BaseException as exc:
        report.update(status='FAIL',error=str(exc),traceback=traceback.format_exc()); raise
    finally: old.write_json(args.output_dir/'report.json',report)


if __name__=='__main__': main()
