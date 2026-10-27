#!/usr/bin/env python3
"""Independent global84 dry-run / gpu-smoke / probe / benchmark / allocate / train."""
from __future__ import annotations
import argparse
import copy
import gc
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
import global84_rank_direction as core
import global84_rank_math as methods
import minicpm_single_layer_rank_gain as shared
from minicpm_rank_gain_decisive import digest, effective_tokens

require=core.require
ROOT=core.ROOT
PROTOCOL=dict(version='global84_stratified_v2_alignment',seeds=[42,43,44],split_seed=20261027,
    adapter_sha256='d79b1b86a36a9809f465ef8c2d0a3251cf18f176832e0e3a48bb719e065d6769',
    train_samples=400,construction_samples=32,alignment_samples=32,selection_samples=32,audit_samples=32,dev_samples=400,
    blocks_per_probe_group=4,samples_per_block=8,steps=400,batch_size=1,lr=5e-5,weight_decay=.01,
    dropout=False,scheduler='none',dtype='bfloat16',max_length=2048,
    minimum_meaningful_nll_improvement=.0005,primary_familywise_alpha=.05,
    primary_interval='Bonferroni two primary comparisons: two-sided 97.5% intervals',
    added_q_ranks=4,added_v_ranks=4,added_parameters=25600,added_scaling=1.,added_A_norm=1.,
    uncertainty_penalty=1.,max_candidates_per_layer=4,krylov_basis=128,eigen_tolerance=1e-4,
    objective='token weighted: per-step mean NLL * sample_tokens / mean_training_tokens',
    C_score='dot(R_construct v, R_selection v), prefix mean minus 1*block SE',
    DE_score='dot(G_construct v_optimized, G_selection v_optimized), prefix mean minus 1*block SE',
    DE_same_allocation=True,endpoint='Dev400 token-weighted NLL at step 400',
    scope='q/v-stratified budget; not freely optimized across q/v',
    prior_method='single-layer residual-SVD did not meet its success criteria; not validated effective')
ROLES=('construction','alignment','selection','audit')
ARMS=('A','B','C','D','E')


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def source_hashes(args):
    paths=[args.adapter,args.train_data,args.dev_data,args.source_data]
    paths += [p for p in Path(args.model).iterdir() if p.is_file() and
              (p.suffix in ('.json','.safetensors','.jinja') or 'token' in p.name)]
    return {str(p.resolve()):shared.sha256(p) for p in paths}


def code_hashes():
    names=['global84_experiment.py','global84_rank_direction.py','global84_rank_math.py','global84_statistics.py',
           'rank_gain_probe_support.py','minicpm_single_layer_rank_gain.py','minicpm_rank_gain_decisive.py','minicpm_rank_gain_statistics.py',
           'finetune/adaptive_lora/rank_expansion.py','finetune/adaptive_lora/residual_gradient.py',
           'finetune/adaptive_lora/subspace_matching.py','finetune/adaptive_lora/training_data.py',
           'finetune/adaptive_lora/data_provenance.py','MyTransformers_RaLoRA_official/common/lora_modules/lora.py']
    return {n:shared.sha256(ROOT/n) for n in names}


def check_hashes(mapping,root=None):
    for path,expected in mapping.items():
        p=Path(path) if root is None else Path(root)/path
        require(p.is_file() and shared.sha256(p)==expected,f'Artifact changed or missing: {p}')


def prepare(args):
    inventory=core.adapter_inventory(args.adapter,args.model)
    require(inventory['old_total_parameters']==480000,'Unexpected PRE adapter budget')
    require(inventory['adapter_sha256']==PROTOCOL['adapter_sha256'],'Not the approved step1600 PRE adapter')
    tm=shared.build_source_metadata(args.train_data,args.source_data)
    dm=shared.build_source_metadata(args.dev_data,args.source_data)
    require(len(dm['samples'])==400,'Fixed Dev400 required')
    split=shared.isolated_split(tm,dm,400,400,128,PROTOCOL['split_seed'])
    indices=dict(train=split['train'],construction=split['probe'][:32],alignment=split['probe'][32:64],selection=split['probe'][64:96],audit=split['probe'][96:],dev=list(range(400)))
    require(sum(x['family']=='GSM' for x in dm['samples'])==200 and sum(x['family']=='MATH' for x in dm['samples'])==200,'Dev must be GSM200/MATH200')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=True)
    if tokenizer.chat_template is None: tokenizer.chat_template=(args.model/'chat_template.jinja').read_text()
    tokenizer.pad_token=tokenizer.eos_token
    datasets={}; metadata={}; counts={}
    for role,ids in indices.items():
        path=args.dev_data if role=='dev' else args.train_data
        rows=json.loads(path.read_text()); selected=[rows[i] for i in ids]
        dest=args.output_dir/f'{role}.json'; write(dest,selected)
        datasets[role]=shared.ChatTrainingDataset(dest,tokenizer,2048,dynamic_length=True)
        metadata[role]=[(dm if role=='dev' else tm)['samples'][i] for i in ids]
        counts[role]=[effective_tokens(row) for row in datasets[role]]
    common=dict(protocol=PROTOCOL,inventory=inventory,indices=indices,metadata=metadata,tokens=counts,
        input_sha256=source_hashes(args),code_sha256=code_hashes(),
        versions={k:importlib.metadata.version(k) for k in ('torch','transformers','safetensors','accelerate','numpy')})
    plan=dict(common=common,fingerprint=digest(common),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
              note='Existing Dev has prior use; not independent blind evaluation. PRE saw historical training data.')
    write(args.output_dir/'plan.json',plan)
    return plan,datasets


def load_plan(folder):
    plan=json.loads((Path(folder)/'plan.json').read_text())
    require(plan['fingerprint']==digest(plan['common']),'Corrupt plan')
    require(plan['common']['protocol']==PROTOCOL,'Protocol mismatch')
    require(plan['common']['code_sha256']==code_hashes(),'Code differs from frozen probe protocol')
    check_hashes(plan['common']['input_sha256'])
    return plan


def load_full(plan):
    inv=plan['common']['inventory']; model,info=shared.load_probe_model(inv['model'],torch.bfloat16)
    model.to('cuda'); model.config.use_cache=False
    base=dict(model.named_parameters())
    payload=torch.load(inv['adapter'],map_location='cpu',weights_only=True)
    names=shared.restore_dense_adapter_on_full_model(model,payload,ROOT)
    require(set(names)=={r['module'] for r in inv['modules']} and len(names)==84,'Incomplete adapter restore')
    layers={n:model.get_submodule(n) for n in names}
    for n,l in layers.items():
        for k in ('weight_a','weight_b'):
            require(shared.tensor_exact(getattr(l,k).detach().cpu(),payload['layers'][n]['state'][k]),f'Inexact load: {n}.{k}')
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    return model,layers,base,info


def gpu_admission(min_free_gib):
    require(torch.cuda.is_available(),'Real CUDA device required')
    free,total=torch.cuda.mem_get_info(); require(free>=min_free_gib*2**30,'Insufficient free GPU memory; no sample reduction permitted')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    return dict(device=torch.cuda.get_device_name(),free_bytes=free,total_bytes=total,tf32=False)


def probe(args,report):
    plan,datasets=prepare(args)
    cache_bytes=sum(r['d_in']*r['d_out']*4 for r in plan['common']['inventory']['modules'])*16
    require(shutil.disk_usage(args.output_dir).free>cache_bytes*1.1,'Insufficient disk space for all gradients; no omissions allowed')
    report['expected_gradient_cache_bytes']=cache_bytes
    print(f'Full probe: 128 samples, 1344 FP32 matrices, {cache_bytes} raw tensor bytes; requires 10% disk headroom',flush=True)
    report['gpu']=gpu_admission(args.min_free_gib)
    model,layers,base,report['loading']=load_full(plan)
    snapshot={n:{k:core.tensor_hash(getattr(l,k)) for k in ('weight_a','weight_b')} for n,l in layers.items()}
    report['blocks']={}; artifacts={}
    for role in ROLES:
        report['blocks'][role]=[]
        for block in range(4):
            torch.cuda.reset_peak_memory_stats()
            items=[shared.batch(datasets[role][i],'cpu') for i in range(block*8,(block+1)*8)]
            gradients,profile=core.collect_weight_gradients(model,layers,items)
            require(profile['num_target_matrices']==84 and profile['forward_calls']==8 and profile['backward_calls']==8,'Probe did not collect all targets together')
            path=args.output_dir/'gradients'/role/str(block); path.mkdir(parents=True,exist_ok=False)
            for i,row in enumerate(plan['common']['inventory']['modules']):
                file=path/f'{i:02d}.pt'; torch.save(gradients[row['module']],file)
                artifacts[str(file.relative_to(args.output_dir))]=shared.sha256(file)
            del gradients; gc.collect()
            report['blocks'][role].append(profile)
            print(f'Probe {role} block {block+1}/4 completed: 8 backward calls, all 84 matrices',flush=True)
            write(args.output_dir/'report.json',report)
    report['base_tensors_verified']=shared.verify_disk_weights(base,plan['common']['inventory']['model'])
    require(all(core.tensor_hash(getattr(layers[n],k))==h for n,fs in snapshot.items() for k,h in fs.items()),'Probe changed adapter')
    report.update(adapter_unchanged=True,total_forward_calls=128,total_backward_calls=128,
        probe_peak_allocated_bytes=max(p['cuda_memory_after']['peak_allocated'] for ps in report['blocks'].values() for p in ps))
    write(args.output_dir/'gradient_artifacts.json',artifacts)
    report['actual_gradient_file_bytes']=sum((args.output_dir/n).stat().st_size for n in artifacts)
    report['disk_free_bytes_after']=shutil.disk_usage(args.output_dir).free
    report['status']='PASS'


def load_gradient_blocks(folder,artifacts,role,index,shape,device):
    require(role in ROLES,'Dev gradients forbidden')
    out=[]
    for block in range(4):
        name=f'gradients/{role}/{block}/{index:02d}.pt'; p=folder/name
        require(name in artifacts and shared.sha256(p)==artifacts[name],f'Gradient hash mismatch: {name}')
        g=torch.load(p,map_location='cpu',weights_only=True)
        require(g.dtype==torch.float32 and tuple(g.shape)==tuple(shape) and torch.isfinite(g).all(),f'Invalid gradient: {name}')
        out.append(g.to(device=device,dtype=torch.float64))
    return out


def block_tokens(plan,role):
    n=plan['common']['tokens'][role]
    return [sum(n[i:i+8]) for i in range(0,32,8)]


def audit_scores(gc,construction_blocks,selection_blocks,audit_blocks,counts,V):
    """Reporting only; never consumed by allocation DP."""
    gs=methods.weighted_mean(selection_blocks,counts['selection']); ga=methods.weighted_mean(audit_blocks,counts['audit'])
    responses=gc@V
    selection=(responses*(gs@V)).sum(dim=0)
    transfer=(responses*(ga@V)).sum(dim=0)
    sa=torch.stack([(responses*(g@V)).sum(dim=0) for g in audit_blocks])
    ss=torch.stack([(responses*(g@V)).sum(dim=0) for g in selection_blocks])
    gc1=methods.weighted_mean(construction_blocks[:2],counts['construction'][:2])@V
    gc2=methods.weighted_mean(construction_blocks[2:],counts['construction'][2:])@V
    denom=gc1.norm(dim=0)*gc2.norm(dim=0)
    cosine=(gc1*gc2).sum(dim=0)/denom.clamp_min(1e-30)
    return dict(selection_U=selection.cpu().tolist(),audit_U=transfer.cpu().tolist(),
        audit_full_score=(ga@V).square().sum(dim=0).cpu().tolist(),
        selection_positive_blocks=(ss>0).sum(dim=0).cpu().tolist(),audit_positive_blocks=(sa>0).sum(dim=0).cpu().tolist(),
        selection_block_U=ss.cpu().tolist(),audit_block_U=sa.cpu().tolist(),audit_block_std=sa.std(dim=0).cpu().tolist(),
        construction_half_response_cosine=[None if float(d)==0 else float(c.clamp(-1,1)) for d,c in zip(denom,cosine)],
        stability_definition='response cosine for FIXED v, not cosine of separately fitted eigenvectors',
        audit_used_for_selection=False,claim='predictive proxy, not a true post-training gain guarantee')


def gpu_smoke(args,report):
    """One real sample, all targets. Deliberately produces NO full-probe artifacts."""
    plan,datasets=prepare(args)
    report['gpu']=gpu_admission(args.min_free_gib)
    model,layers,base,report['loading']=load_full(plan)
    snapshot={n:core.tensor_hash(p) for n,p in model.named_parameters()}
    torch.cuda.reset_peak_memory_stats()
    gradients,profile=core.collect_weight_gradients(model,layers,[shared.batch(datasets['construction'][0],'cpu')])
    require(len(gradients)==84 and profile['forward_calls']==profile['backward_calls']==1,'Incomplete smoke backward')
    require(all(torch.isfinite(g).all() for g in gradients.values()),'Nonfinite smoke gradient')
    require(all(core.tensor_hash(p)==snapshot[n] for n,p in model.named_parameters()),'Smoke changed original weights')
    report.update(status='PASS',profile=profile,all_original_weights_unchanged=True,
        base_tensors_verified=shared.verify_disk_weights(base,plan['common']['inventory']['model']),
        gradient_norms={n:float(g.norm()) for n,g in gradients.items()},
        allocation_eligible=False,sample_index=plan['common']['indices']['construction'][0],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())


def analysis_cache_key(plan,artifacts,index,seed,device):
    names={f'gradients/{role}/{block}/{index:02d}.pt' for role in ROLES for block in range(4)}
    return digest(dict(plan=plan['fingerprint'],module_index=index,seed=seed,device=device,
                       gradients={n:artifacts[n] for n in sorted(names)}))


def reuse_analysis(folder,index,key):
    """Incomplete entries may be recomputed; completed but inconsistent entries fail."""
    marker=Path(folder)/'module_cache'/f'{index:02d}.json'
    if not marker.exists(): return None
    cache=json.loads(marker.read_text())
    require(cache['status']=='PASS' and cache['key']==key,'Module cache protocol/input/seed mismatch')
    require(set(cache['files'])=={f'directions/{index:02d}.pt',f'module_cache/{index:02d}.record.json'},'Incomplete module cache hashes')
    check_hashes(cache['files'],folder)
    record=json.loads((Path(folder)/'module_cache'/f'{index:02d}.record.json').read_text())
    require(set(record['solver'])==set(methods.FAMILIES)-{'random'},'Incomplete cached solver families')
    require(all(d['converged'] and len(d['relative_ritz_residuals'])==4 and
                all(math.isfinite(x) and 0<=x<=PROTOCOL['eigen_tolerance'] for x in d['relative_ritz_residuals'])
                for d in record['solver'].values()),'Unconverged cached solver')
    return record


def analyze_module(source,artifacts,plan,payload,i,row,seed,device):
    counts={role:block_tokens(plan,role) for role in ROLES}
    name=row['module']; state=payload['layers'][name]['state']
    a=state['weight_a'].to(device); b=state['weight_b'].to(device)
    shape=(row['d_out'],row['d_in'])
    cb=load_gradient_blocks(source,artifacts,'construction',i,shape,device)
    alignment=load_gradient_blocks(source,artifacts,'alignment',i,shape,device)
    gc_=methods.weighted_mean(cb,counts['construction'])
    ga=methods.weighted_mean(alignment,counts['alignment'])
    frames,solver=methods.candidate_frames(a,b,gc_,ga,4,methods.stream(seed,name),PROTOCOL['krylov_basis'],PROTOCOL['eigen_tolerance'])
    del alignment,ga
    # Allocation scoring data is only read AFTER direction fitting finishes.
    sb=load_gradient_blocks(source,artifacts,'selection',i,shape,device)
    qa,qb=methods.basis(a.T),methods.basis(b)
    rc=shared.residual_gradient(gc_,qa,qb)
    rs=[shared.residual_gradient(g,qa,qb) for g in sb]
    c,_=methods.proxy_prefixes(rc,rs,counts['selection'],frames['residual_svd'],row['cost_per_added_rank'],PROTOCOL['uncertainty_penalty'])
    d,_=methods.proxy_prefixes(gc_,sb,counts['selection'],frames['cross_alignment'],row['cost_per_added_rank'],PROTOCOL['uncertainty_penalty'])
    ab=load_gradient_blocks(source,artifacts,'audit',i,shape,device)
    scores={f:audit_scores(gc_,cb,sb,ab,counts,V) for f,V in frames.items()}
    record=dict(module=name,cost_per_added_rank=row['cost_per_added_rank'],solver=solver,candidates=scores,
                direction_fit_roles=dict(random=[],residual_svd=['construction'],projected_full_svd=['construction'],
                                        cross_alignment=['construction','alignment']),allocation_score_role='selection',
                C_prefix_options=c,DE_prefix_options=d)
    return {f:V.float().cpu() for f,V in frames.items()},record


def allocate(args,report,benchmark=False):
    require(args.probe_dir is not None,'Analysis requires --probe-dir')
    source=args.probe_dir; plan=load_plan(source)
    old_report=json.loads((source/'report.json').read_text())
    require(old_report['status']=='PASS' and old_report['mode']=='probe','Completed full probe required; smoke is ineligible')
    require(old_report.get('total_backward_calls')==128 and old_report.get('total_forward_calls')==128,'Full 128-sample probe required')
    artifacts=json.loads((source/'gradient_artifacts.json').read_text())
    expected={f'gradients/{role}/{block}/{i:02d}.pt' for role in ROLES for block in range(4) for i in range(84)}
    require(set(artifacts)==expected,'Expected every target/group/block gradient')
    write(args.output_dir/'plan.json',plan)
    payload=torch.load(plan['common']['inventory']['adapter'],map_location='cpu',weights_only=True)
    if benchmark: require(args.analysis_device=='cpu' and args.module is not None,'Benchmark requires --module and CPU')
    if args.analysis_device=='cuda': report['gpu']=gpu_admission(args.min_free_gib)
    rows=plan['common']['inventory']['modules']; options_C={}; options_DE={}; files={}
    selected=[(i,r) for i,r in enumerate(rows) if not benchmark or r['module']==args.module]
    require(bool(selected),'Unknown benchmark module')
    frame_dir=args.output_dir/'directions'; frame_dir.mkdir()
    cache_dir=args.output_dir/'module_cache'; cache_dir.mkdir()
    started=time.perf_counter(); reused=[]
    for i,row in selected:
        name=row['module']; tick=time.perf_counter()
        key=analysis_cache_key(plan,artifacts,i,args.seed,args.analysis_device)
        record=reuse_analysis(args.reuse_dir,i,key) if args.reuse_dir else None
        file=frame_dir/f'{i:02d}.pt'
        if record is None:
            frames,record=analyze_module(source,artifacts,plan,payload,i,row,args.seed,args.analysis_device)
            torch.save(frames,file); del frames
        else:
            # Verify the actual source gradients as well as the reusable cache.
            subset={n:h for n,h in artifacts.items() if n.endswith(f'/{i:02d}.pt')}
            check_hashes(subset,source)
            shutil.copyfile(args.reuse_dir/'directions'/f'{i:02d}.pt',file); reused.append(name)
        record_path=cache_dir/f'{i:02d}.record.json'; write(record_path,record)
        marker_files={str(p.relative_to(args.output_dir)):shared.sha256(p) for p in (file,record_path)}
        # Completion marker is written last; a failed module is never reusable.
        write(cache_dir/f'{i:02d}.json',dict(status='PASS',key=key,files=marker_files))
        files[str(file.relative_to(args.output_dir))]=shared.sha256(file)
        options_C[name]=record['C_prefix_options']; options_DE[name]=record['DE_prefix_options']
        with (args.output_dir/'candidates.jsonl').open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
        print(f'Analysis {i+1}/84 {name}: {time.perf_counter()-tick:.2f}s, reused={name in reused}',flush=True)
        gc.collect()
    report['reused_modules']=reused
    if benchmark:
        report.update(status='PASS',cpu_only=True,module=args.module,seconds=time.perf_counter()-started,
                      solver=record['solver'],allocation_eligible=False)
        return
    c,cr=methods.exact_stratified_allocation(options_C,rows)
    d,dr=methods.exact_stratified_allocation(options_DE,rows)
    b=methods.random_allocation(rows,args.seed)
    allocations=dict(A={r['module']:0 for r in rows},B=b,C=c,D=d,E=copy.deepcopy(d))
    for arm,allocation in allocations.items():
        budget=sum(allocation[r['module']]*r['cost_per_added_rank'] for r in rows)
        require(budget==(0 if arm=='A' else 25600),'Wrong allocated budget')
    require(allocations['D']==allocations['E'],'D/E allocations differ')
    result=dict(seed=args.seed,plan_fingerprint=plan['fingerprint'],allocations=allocations,
                direction_family=dict(A=None,B='random',C='residual_svd',D='random',E='cross_alignment'),
                C_optimization=cr,DE_optimization=dr,audit_used_for_allocation=False,
                scoring_bias='Directions fitted on construction/alignment, allocation scored on independent selection; four-block SE remains heuristic',
                frames_sha256=files,source_probe=str(source.resolve()),source_probe_report_sha256=shared.sha256(source/'report.json'))
    write(args.output_dir/'allocation.json',result)
    files['allocation.json']=shared.sha256(args.output_dir/'allocation.json'); files['candidates.jsonl']=shared.sha256(args.output_dir/'candidates.jsonl')
    write(args.output_dir/'artifacts.json',files)
    report.update(status='PASS',seed=args.seed,seconds=time.perf_counter()-started,added_parameters=25600,scope=PROTOCOL['scope'])


def restore_layers(model,layers,snapshot):
    for name,layer in layers.items():
        parent,_,child=name.rpartition('.'); setattr(model.get_submodule(parent),child,layer)
        with torch.no_grad():
            for key,value in snapshot[name].items(): getattr(layer,key).copy_(value)
    for p in model.parameters(): p.grad=None; p.requires_grad_(False)


def attach_branches(model,layers,allocation,frames,family):
    selected={}; added=0
    for name,layer in layers.items():
        k=allocation[name]
        if not k: continue
        V=frames[name][family][:,:k].to(layer.weight.device)
        qa=methods.basis(layer.weight_a.T)
        require(V.shape==(layer.in_features,k) and torch.isfinite(V).all(),'Invalid selected frame')
        require(float((V.double().T@V.double()-torch.eye(k,device=V.device,dtype=torch.float64)).abs().max())<2e-5,'Nonorthogonal selected directions')
        require(float((qa.T@V.double()).abs().max())<2e-5,'Selected directions overlap old A')
        wrap=shared.ExpandedLoRA(layer,V.T,torch.zeros(layer.out_features,k,device=layer.weight.device))
        parent,_,child=name.rpartition('.'); setattr(model.get_submodule(parent),child,wrap)
        selected[name]=wrap; added+=wrap.add_a.numel()+wrap.add_b.numel()
    return selected,added


def train_adapters(model,layers,expanded,examples,order,device,log_path=None):
    require(sorted(order)==list(range(len(examples))),'Expected one permutation of training set')
    factors={}
    for p in model.parameters(): p.requires_grad_(False); p.grad=None
    for name,layer in layers.items():
        for key in ('weight_a','weight_b'): factors[name+'.'+key]=getattr(layer,key)
    for name,wrapper in expanded.items():
        factors[name+'.add_a']=wrapper.add_a; factors[name+'.add_b']=wrapper.add_b
    initial={k:p.detach().clone() for k,p in factors.items()}
    for p in factors.values(): p.requires_grad_(True)
    require(sum(p.numel() for p in model.parameters() if p.requires_grad)==sum(p.numel() for p in factors.values()),'Unexpected trainable params')
    optimizer=torch.optim.AdamW(list(factors.values()),lr=5e-5,weight_decay=.01)
    require(not optimizer.state,'Fresh AdamW required')
    model.eval(); counts=[effective_tokens(x) for x in examples]; mean_tokens=sum(counts)/len(counts)
    records=[]
    for step,index in enumerate(order,1):
        optimizer.zero_grad(set_to_none=True)
        loss=model(**shared.batch(examples[index],device),use_cache=False).loss
        objective=loss*(counts[index]/mean_tokens)
        require(torch.isfinite(objective),'Nonfinite train loss')
        objective.backward()
        norms={}
        for key,p in factors.items():
            require(p.grad is not None and torch.isfinite(p.grad).all(),f'Invalid factor gradient: {key}')
            norms[key]=float(p.grad.float().norm())
        optimizer.step()
        require(all(torch.isfinite(p).all() for p in factors.values()),'Nonfinite adapter')
        record=dict(step=step,pool_index=index,nll=float(loss.detach()),effective_tokens=counts[index],
            nll_sum=float(loss.detach())*counts[index],backward_objective=float(objective.detach()),lr=5e-5,gradient_norms=norms)
        records.append(record)
        if log_path:
            with Path(log_path).open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
    optimizer.zero_grad(set_to_none=True)
    changes={key:dict(initial_norm=float(initial[key].float().norm()),final_norm=float(p.detach().float().norm()),
        change_norm=float((p.detach().float()-initial[key].float()).norm())) for key,p in factors.items()}
    modules={}
    for name in layers:
        keys=[k for k in factors if k.rsplit('.',1)[0]==name]
        details={k.rsplit('.',1)[1]:dict(**changes[k],
            max_gradient_norm=max(r['gradient_norms'][k] for r in records),
            nonzero_gradient_steps=sum(r['gradient_norms'][k]>0 for r in records)) for k in keys}
        for key,d in details.items():
            require(d['max_gradient_norm']>0 and d['change_norm']>0,
                    f'Inactive factor (no effective gradient or no actual update): {name}.{key}')
        ba=None
        if name in expanded:
            w=expanded[name]; aa=w.add_a.detach().double(); bb=w.add_b.detach().double()
            ba=float(torch.trace((bb.T@bb)@(aa@aa.T)).clamp_min(0).sqrt())
            require(math.isfinite(ba) and ba>0,f'Inactive added BA branch: {name}')
        modules[name]=dict(factors=details,added_BA_fro_norm=ba,effective=True)
    return dict(module_diagnostics=modules,all_factors_effective=True,
                completed_steps=len(records),steps=records,trainable_parameters=sum(p.numel() for p in factors.values()),
                factor_changes=changes,fresh_AdamW=True,historical_moments_restored=False,optimizer_defaults=optimizer.defaults,
                mean_train_tokens=mean_tokens)


def eval_dev(model,data,meta,reference=None):
    result=shared.evaluate(model,data,'cuda',reference)
    for row,source in zip(result['samples'],meta): row['source']=source
    result['mean_question_nll']=sum(r['nll'] for r in result['samples'])/len(result['samples'])
    result['by_family']={family:sum(r['nll_sum'] for r in result['samples'] if r['source']['family']==family)/sum(r['effective_tokens'] for r in result['samples'] if r['source']['family']==family) for family in ('GSM','MATH')}
    return result


def train(args,report):
    require(args.allocation_dir is not None,'train requires --allocation-dir')
    folder=args.allocation_dir; plan=load_plan(folder)
    require(json.loads((folder/'report.json').read_text())['status']=='PASS','Allocation failed')
    check_hashes(json.loads((folder/'artifacts.json').read_text()),folder)
    allocation=json.loads((folder/'allocation.json').read_text())
    require(allocation['seed']==args.seed and allocation['plan_fingerprint']==plan['fingerprint'],'Allocation seed/data mismatch')
    require(allocation['allocations']['D']==allocation['allocations']['E'],'D/E mismatch')
    write(args.output_dir/'plan.json',plan); write(args.output_dir/'allocation.json',allocation)
    from transformers import AutoTokenizer
    model_path=plan['common']['inventory']['model']
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=True)
    if tok.chat_template is None: tok.chat_template=(Path(model_path)/'chat_template.jinja').read_text()
    tok.pad_token=tok.eos_token
    # Recover selected rows from original sources, never trust a replacement split.
    input_paths=list(plan['common']['input_sha256'])
    train_path=next(Path(p) for p in input_paths if Path(p).resolve()==args.train_data.resolve())
    dev_path=next(Path(p) for p in input_paths if Path(p).resolve()==args.dev_data.resolve())
    data={}
    for role,path in [('train',train_path),('dev',dev_path)]:
        rows=json.loads(path.read_text()); chosen=[rows[i] for i in plan['common']['indices'][role]]
        dest=args.output_dir/f'{role}.json'; write(dest,chosen)
        data[role]=shared.ChatTrainingDataset(dest,tok,2048,dynamic_length=True)
        require([effective_tokens(x) for x in data[role]]==plan['common']['tokens'][role],'Tokenization changed')
    frames={r['module']:torch.load(folder/'directions'/f'{i:02d}.pt',map_location='cpu',weights_only=True) for i,r in enumerate(plan['common']['inventory']['modules'])}
    order=list(range(400)); random.Random(methods.stream(args.seed,'train_order')).shuffle(order)
    report.update(seed=args.seed,plan_fingerprint=plan['fingerprint'],train_order=order,
                  source_order=[plan['common']['indices']['train'][i] for i in order],allocation_sha256=shared.sha256(folder/'allocation.json'),
                  arms_requested=args.arms)
    report['gpu']=gpu_admission(args.min_free_gib)
    model,layers,base,report['loading']=load_full(plan)
    snapshot={n:{k:getattr(l,k).detach().clone() for k in ('weight_a','weight_b')} for n,l in layers.items()}
    original_objects={n:{k:getattr(l,k) for k in ('weight_a','weight_b')} for n,l in layers.items()}
    scaling={n:l.lora_scaler for n,l in layers.items()}; ranks={n:l.lora_rank for n,l in layers.items()}
    baseline=eval_dev(model,data['dev'],plan['common']['metadata']['dev'])
    try:
        for arm in args.arms:
            out=args.output_dir/arm; out.mkdir(exist_ok=False)
            restore_layers(model,layers,snapshot)
            chosen=allocation['allocations'][arm]
            require(set(chosen)==set(layers) and all(type(k)==int and 0<=k<=4 for k in chosen.values()),'Invalid rank manifest')
            for kind in ('q_proj','v_proj'): require(sum(k for n,k in chosen.items() if n.endswith(kind))==(0 if arm=='A' else 4),'Incorrect type rank budget')
            expanded,added=attach_branches(model,layers,chosen,frames,allocation['direction_family'][arm])
            require(added==(0 if arm=='A' else 25600),'Actual added parameter count differs')
            require(all(shared.tensor_exact(getattr(layers[n],k),v) for n,fs in snapshot.items() for k,v in fs.items()),'Insertion changed original factors')
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
            before=eval_dev(model,data['dev'],plan['common']['metadata']['dev'],baseline['samples'])
            torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
            torch.cuda.synchronize(); train_started=time.perf_counter()
            training=train_adapters(model,layers,expanded,data['train'],order,'cuda',out/'train.jsonl')
            torch.cuda.synchronize(); training_seconds=time.perf_counter()-train_started
            require(training['completed_steps']==400 and training['trainable_parameters']==480000+added,'Incorrect training endpoint/count')
            after=eval_dev(model,data['dev'],plan['common']['metadata']['dev'])
            base_verified=shared.verify_disk_weights(base,model_path)
            require(all(layers[n].lora_scaler==scaling[n] and layers[n].lora_rank==ranks[n] for n in layers),'Original scaling/rank changed')
            require(all(getattr(layers[n],k) is p for n,fs in original_objects.items() for k,p in fs.items()),'Original factor Parameter replaced')
            updated={n:{k:getattr(l,k).detach().cpu().clone() for k in ('weight_a','weight_b')} for n,l in layers.items()}
            branch_norms={}
            for n,w in expanded.items():
                updated[n].update(add_a=w.add_a.detach().cpu(),add_b=w.add_b.detach().cpu())
                aa=w.add_a.detach().double(); bb=w.add_b.detach().double()
                branch_norms[n]=float(torch.trace((bb.T@bb)@(aa@aa.T)).clamp_min(0).sqrt())
            torch.save(dict(format='global84 research state; NOT PEFT',layers=updated,allocation=chosen,scaling=scaling),out/'adapter.pt')
            torch.cuda.synchronize()
            result=dict(status='PASS',arm=arm,seed=args.seed,plan_fingerprint=plan['fingerprint'],before=before,after=after,
                training=training,allocation=chosen,added_parameters=added,branch_BA_fro_norms=branch_norms,
                seconds=time.perf_counter()-start,training_seconds=training_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                invariants=dict(base_weights_verified=base_verified,original_factor_objects_preserved=True,original_scaling_preserved=True,
                                insertion_full_logits_equal_PRE=True),
                note='All original 84 adapters are trained. Unexpanded adapters are NOT claimed frozen.')
            write(out/'result.json',result)
            write(out/'artifacts.json',{name:shared.sha256(out/name) for name in ('result.json','train.jsonl','adapter.pt')})
            report.setdefault('completed_arms',[]).append(arm); write(args.output_dir/'report.json',report)
            print(f'{arm} seed {args.seed}: Dev400 NLL={after["token_weighted_nll"]:.9f}',flush=True)
            del training,expanded; gc.collect()
        report['status']='PASS'
    finally:
        restore_layers(model,layers,snapshot)
        require(all(shared.tensor_exact(getattr(layers[n],k),v) for n,fs in snapshot.items() for k,v in fs.items()),'PRE restore failed')
        report['in_memory_PRE_restored']=True


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['dry-run','gpu-smoke','probe','benchmark','allocate','train'])
    p.add_argument('--model',type=Path,default=core.DEFAULT_MODEL)
    p.add_argument('--adapter',type=Path,default=core.DEFAULT_ADAPTER)
    p.add_argument('--train-data',type=Path,default=shared.DATA/'train.json')
    p.add_argument('--dev-data',type=Path,default=shared.DATA/'dev.json')
    p.add_argument('--source-data',type=Path,default=Path('/home/hongchang/whf/datasets/MetaMathQA/MetaMathQA-395K.jsonl'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--probe-dir',type=Path)
    p.add_argument('--allocation-dir',type=Path)
    p.add_argument('--reuse-dir',type=Path,help='Read-only completed per-module analysis cache; hash mismatches fail')
    p.add_argument('--module',help='Exact module name for single-layer CPU benchmark')
    p.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    p.add_argument('--analysis-device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--min-free-gib',type=float,default=30.)
    return p


def main():
    args=parser().parse_args(); require(len(args.arms)==len(set(args.arms)),'Duplicate arms')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',mode=args.mode,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    started=time.perf_counter()
    try:
        if args.mode=='dry-run':
            plan,_=prepare(args); report.update(status='PASS',cpu_only=True,plan_fingerprint=plan['fingerprint'],inventory=plan['common']['inventory'])
        elif args.mode=='gpu-smoke': gpu_smoke(args,report)
        elif args.mode=='probe': probe(args,report)
        elif args.mode=='benchmark': allocate(args,report,benchmark=True)
        elif args.mode=='allocate': allocate(args,report)
        else: train(args,report)
        plan=json.loads((args.output_dir/'plan.json').read_text()); check_hashes(plan['common']['input_sha256'])
        report['input_files_unchanged']=True
    except BaseException as error:
        report.update(status='FAIL',error=str(error),traceback=traceback.format_exc()); raise
    finally:
        report['wall_seconds']=time.perf_counter()-started
        write(args.output_dir/'report.json',report)


if __name__=='__main__': main()
