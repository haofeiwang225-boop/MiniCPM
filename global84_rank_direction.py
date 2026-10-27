#!/usr/bin/env python3
"""Global-84 research foundations: immutable inventory and simultaneous probe.

Experiment stages live in global84_experiment.py. This module keeps the
standalone CPU inventory CLI and simultaneous-probe implementation.
The preceding residual-SVD study did not meet its success criteria.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import rank_gain_probe_support as support

ROOT=Path(__file__).resolve().parent
DEFAULT_ADAPTER=ROOT/'finetune/adaptive_lora/outputs/metamath_pilot_geo1600_rebuilt_20261013_221250_seed42/correction_checkpoints/step_1600_pre.pt'
DEFAULT_MODEL=Path('/home/hongchang/whf/models/MiniCPM5-2B')


def require(condition,message):
    if not bool(condition): raise ValueError(message)


def tensor_hash(value):
    """One-at-a-time host transfer, preserving raw dtype bits (including BF16)."""
    x=value.detach().contiguous().cpu()
    h=hashlib.sha256()
    h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode())
    h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def adapter_inventory(adapter,model_dir):
    adapter,model_dir=Path(adapter),Path(model_dir)
    before=support.sha256(adapter)
    payload=torch.load(adapter,map_location='cpu',weights_only=True)
    config=json.loads((model_dir/'config.json').read_text())
    require(Path(payload['base_model']).resolve()==model_dir.resolve(),'Adapter/base path mismatch')
    require(config['num_hidden_layers']==42,'Expected actual MiniCPM5-2B 42 layers')
    expected={f'model.layers.{i}.self_attn.{k}' for i in range(config['num_hidden_layers']) for k in ('q_proj','v_proj')}
    require(set(payload['layers'])==expected and len(expected)==84,'Expected all 84 saved q/v modules')
    from safetensors import safe_open
    weight_map=json.loads((model_dir/'model.safetensors.index.json').read_text())['weight_map']
    rows=[]
    for name in sorted(expected,key=lambda n:(int(n.split('.')[2]),n.split('.')[-1])):
        saved=payload['layers'][name]; state=saved['state']; cfg=saved['config']
        require(saved['structure']=='dense' and set(state)=={'weight_a','weight_b'},f'Not plain Dense: {name}')
        a,b=state['weight_a'],state['weight_b']
        require(a.ndim==b.ndim==2 and a.shape[0]==b.shape[1],f'Invalid factor shapes: {name}')
        rank,dim_in=a.shape; dim_out=b.shape[0]
        require((cfg['lora_rank'],cfg['in_features'],cfg['out_features'])==(rank,dim_in,dim_out),f'Config/shape mismatch: {name}')
        scaling=float(saved['scaling'])
        require(torch.isfinite(a).all() and torch.isfinite(b).all() and np.isfinite(scaling),f'Nonfinite adapter: {name}')
        key=name+'.weight'
        with safe_open(model_dir/weight_map[key],framework='pt',device='cpu') as handle:
            require(handle.get_slice(key).get_shape()==[dim_out,dim_in],f'Base tensor shape mismatch: {name}')
        rows.append(dict(module=name,A_shape=list(a.shape),B_shape=list(b.shape),rank=rank,scaling=scaling,
                         A_dtype=str(a.dtype),B_dtype=str(b.dtype),d_in=dim_in,d_out=dim_out,
                         cost_per_added_rank=dim_in+dim_out,old_parameters=a.numel()+b.numel()))
    require(support.sha256(adapter)==before,'Source adapter changed during read')
    return dict(adapter=str(adapter.resolve()),adapter_sha256=before,model=str(model_dir.resolve()),
                model_config_sha256=support.sha256(model_dir/'config.json'),num_modules=len(rows),
                old_total_parameters=sum(r['old_parameters'] for r in rows),modules=rows,
                base_tensor_shapes_verified=True,source_adapter_unchanged=True,
                prior_SVD_result='Not validated effective; previous study missed preregistered success criteria')


def collect_weight_gradients(model,targets,batches):
    """One forward/backward per batch for ALL targets, no layerwise reruns.

    targets: ordered name -> original Dense-LoRA layer mapping.
    batches: already masked/collated dicts, batch size one (may live on CPU).
    Native leaf gradients are copied to CPU and accumulated in FP32 by
    post-accumulate hooks, then ordinary .grad is immediately cleared.
    Restores preexisting grad objects, flags, all modes and Python/NumPy/Torch
    RNG on both success and exception. Never creates or steps an optimizer.
    All target gradients are required on every batch, including zero gradients.
    """
    require(bool(targets) and bool(batches),'Nonempty targets and batches required')
    params=list(model.parameters()); param_ids={id(p) for p in params}
    weights={name:layer.weight for name,layer in targets.items()}
    require(len({id(p) for p in weights.values()})==len(weights),'Aliased target weights are unsupported')
    require(all(id(p) in param_ids and p.is_leaf and p.ndim==2 for p in weights.values()),'Targets must be model leaf matrices')
    device=next(iter(weights.values())).device
    require(all(p.device==device for p in params),'Single-device probe required')
    counts=[]
    for batch in batches:
        require(batch['labels'].ndim==2 and batch['labels'].shape[0]==1,'Probe requires explicit batch size one')
        count=int((batch['labels'][:,1:]!=-100).sum())
        require(count>0,'Empty shifted supervised labels; never skip a sample')
        counts.append(count)
    flags=[p.requires_grad for p in params]; grads=[p.grad for p in params]
    modes=[(module,module.training) for module in model.modules()]
    python_rng=random.getstate(); numpy_rng=np.random.get_state()
    devices=sorted({p.device.index for p in params if p.is_cuda})
    weight_hashes={name:tensor_hash(p) for name,p in weights.items()}
    # Never retain all 84 full gradients on GPU; no per-layer forward loop.
    accum={name:torch.zeros(p.shape,dtype=torch.float32,device='cpu') for name,p in weights.items()}
    hooks=[]; rows=[]; seen={}; alpha=0.; current_batch=-1
    calls={name:0 for name in targets}
    started=time.perf_counter()
    cuda_before=None
    if device.type=='cuda':
        torch.cuda.synchronize(device)
        cuda_before=dict(allocated=torch.cuda.memory_allocated(device),reserved=torch.cuda.memory_reserved(device),
                         peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device))
    def make_hook(name):
        def hook(parameter):
            require(name not in seen,f'Multiple gradient accumulations for {name} in batch {current_batch}')
            require(parameter.grad is not None,f'Missing gradient: {name}')
            host=parameter.grad.detach().to(device='cpu',dtype=torch.float32)
            require(torch.isfinite(host).all(),f'Nonfinite gradient: {name}, batch {current_batch}')
            accum[name].add_(host,alpha=alpha)
            parameter.grad=None
            seen[name]=True; calls[name]+=1
        return hook
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            for p in params: p.requires_grad_(False); p.grad=None
            for name,p in weights.items():
                p.requires_grad_(True)
                hooks.append(p.register_post_accumulate_grad_hook(make_hook(name)))
            for i,(batch,count) in enumerate(zip(batches,counts)):
                current_batch=i; seen={}; alpha=count/sum(counts)
                inputs={k:v.to(device) for k,v in batch.items()}
                output=model(**inputs,use_cache=False)
                loss=output.loss
                require(loss.numel()==1 and torch.isfinite(loss),f'Nonfinite/nonscalar loss in batch {i}')
                loss.backward()
                require(set(seen)==set(weights),f'Missing target gradients in batch {i}: {sorted(set(weights)-set(seen))}')
                rows.append(dict(batch_index=i,effective_tokens=count,nll=float(loss.detach()),nll_sum=float(loss.detach())*count))
                del loss,output,inputs
            require(all(n==len(batches) for n in calls.values()),'Incomplete gradient counts')
            require(all(torch.isfinite(g).all() for g in accum.values()),'Nonfinite accumulated gradient')
    finally:
        for hook in hooks: hook.remove()
        for p,flag,grad in zip(params,flags,grads): p.requires_grad_(flag); p.grad=grad
        for module,mode in modes: module.training=mode
        random.setstate(python_rng); np.random.set_state(numpy_rng)
        require(all(tensor_hash(p)==weight_hashes[name] for name,p in weights.items()),'Probe changed a target base weight')
    if device.type=='cuda': torch.cuda.synchronize(device)
    profile=dict(seconds=time.perf_counter()-started,num_batches=len(batches),forward_calls=len(batches),
        backward_calls=len(batches),num_target_matrices=len(targets),gradient_calls=calls,
        effective_tokens=sum(counts),normalization='sum(n_i * grad(mean_masked_token_NLL_i)) / sum(n_i)',
        accumulation_dtype='float32',accumulation_device='cpu',
        cpu_gradient_bytes=sum(g.numel()*g.element_size() for g in accum.values()),
        target_base_weights_unchanged=True,training_state_restored=True,batch_losses=rows,
        cuda_memory_before=cuda_before,
        cuda_memory_after=None if device.type!='cuda' else dict(allocated=torch.cuda.memory_allocated(device),
            reserved=torch.cuda.memory_reserved(device),peak_allocated=torch.cuda.max_memory_allocated(device),
            peak_reserved=torch.cuda.max_memory_reserved(device)),
        memory_note='CUDA peaks are process high-water marks, not reset by this helper; caller may reset before probe.')
    return accum,profile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',type=Path,default=DEFAULT_ADAPTER)
    p.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',mode='cpu_inventory_only')
    try:
        report.update(adapter_inventory(args.adapter,args.model),status='PASS')
    except BaseException as error:
        import traceback
        report.update(status='FAIL',error=str(error),traceback=traceback.format_exc()); raise
    finally: (args.output_dir/'inventory.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
