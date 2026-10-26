#!/usr/bin/env python3
"""Isolated A/B/C single-layer rank-4 -> capacity-5 experiment; never resume Adam.

Only this entry point can run the real model. Importing helpers/tests is CPU-only.
All arms use eval-mode (dropout disabled) with autograd enabled during training,
constant LR, fresh AdamW, batch size one, identical ordered supervised examples.
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
import subprocess
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'finetune'))
from rank_gain_probe_support import (sha256, must, tensor_exact, load_probe_model,
    restore_dense_adapter_on_full_model, verify_disk_weights)
from adaptive_lora.rank_expansion import ExpandedLoRA
from adaptive_lora.subspace_matching import orth
from adaptive_lora.residual_gradient import residual_gradient
from adaptive_lora.training_data import ChatTrainingDataset
from adaptive_lora.data_provenance import build_source_metadata

MODEL = '/home/hongchang/whf/models/MiniCPM5-2B'
ADAPTER = ROOT / 'finetune/adaptive_lora/outputs/metamath_pilot_geo1600_rebuilt_20261013_221250_seed42/correction_checkpoints/step_1600_pre.pt'
DATA = ROOT / 'finetune/adaptive_lora/data/metamath_pilot_v1'
MODULES = ['model.layers.16.self_attn.q_proj', 'model.layers.18.self_attn.q_proj']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def isolated_split(train_meta, dev_meta, ntrain, ndev, nprobe, seed):
    """Require full source-pool group isolation; then select distinct train groups."""
    train = train_meta['samples']; dev = dev_meta['samples']
    for key in ('sample_sha256', 'original_question_sha256'):
        must(not ({r[key] for r in train} & {r[key] for r in dev}), f'Train/dev overlap: {key}')
    rng = random.Random(seed)
    def unique_indices(rows):
        indices = list(range(len(rows))); rng.shuffle(indices)
        groups = set(); hashes = set(); result = []
        for i in indices:
            r = rows[i]
            if r['original_question_sha256'] in groups or r['sample_sha256'] in hashes:
                continue
            groups.add(r['original_question_sha256']); hashes.add(r['sample_sha256']); result.append(i)
        return result
    ti, vi = unique_indices(train), unique_indices(dev)
    must(len(ti) >= ntrain+nprobe and len(vi) >= ndev, 'Insufficient independent source groups')
    result = dict(train=ti[:ntrain], probe=ti[ntrain:ntrain+nprobe], validation=vi[:ndev])
    must(not set(result['train']) & set(result['probe']), 'Probe/train sample overlap')
    return result


def batch(example, device):
    return {k: v.unsqueeze(0).to(device) for k, v in example.items()}


def averaged_gradient(model, layer, batches):
    """Equal-example masked LM loss gradients; no optimizer or adapter update.

    Local autograd implementation avoids importing the official DeepSpeed registry.
    The same frozen-W full-gradient definition as the existing probe is retained.
    """
    params = list(model.parameters()); flags = [p.requires_grad for p in params]
    grads = [p.grad for p in params]; modes = [(m, m.training) for m in model.modules()]
    before = layer.weight.detach().clone()
    devices = sorted({p.device.index for p in params if p.is_cuda})
    losses = []; total = torch.zeros_like(layer.weight, dtype=torch.float32)
    must(bool(batches), 'Empty probe')
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            for p in params: p.requires_grad_(False)
            layer.weight.requires_grad_(True)
            for item in batches:
                loss = model(**item, use_cache=False).loss
                must(torch.isfinite(loss), 'Nonfinite probe loss')
                g, = torch.autograd.grad(loss, layer.weight)
                must(torch.isfinite(g).all(), 'Nonfinite probe gradient')
                total.add_(g.float()); losses.append(float(loss.detach()))
    finally:
        for p, flag, grad in zip(params, flags, grads):
            p.requires_grad_(flag); p.grad = grad
        for m, mode in modes: m.training = mode
        must(tensor_exact(before, layer.weight), 'Probe mutated base weight')
    return total / len(batches), losses


def new_directions(a, b, gradient, seed):
    """Unit-norm projected random direction and leading residual right vector."""
    qa, qb = orth(a.T), orth(b)
    must(qa.shape[1] == a.shape[0] and qb.shape[1] == b.shape[1], 'Rank-deficient old factors')
    r = residual_gradient(gradient.float(), qa, qb)
    _, s, vh = torch.linalg.svd(r, full_matrices=False)
    must(torch.isfinite(s).all() and float(s[0]) > max(1e-12, float(gradient.norm())*1e-7),
         'No reliable nonzero tangent residual direction')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    random_v = torch.randn(a.shape[1], generator=rng).to(a.device)
    def normalize(v):
        v = v.float() - qa @ (qa.T @ v.float())
        must(float(v.norm()) > 1e-7, 'Projected direction vanished')
        v = v / v.norm()
        # Canonical sign: both unit vectors remain equivalent subspaces.
        if v[v.abs().argmax()] < 0: v = -v
        must(float((qa.T @ v).abs().max()) < 2e-5, 'New A not orthogonal to old A')
        return v.unsqueeze(0)
    rand, residual = normalize(random_v), normalize(vh[0])
    must(torch.allclose(rand.norm(), residual.norm(), atol=1e-6, rtol=1e-6), 'New A norms differ')
    return rand, residual, dict(gradient_fro_norm=float(gradient.norm()), residual_fro_norm=float(r.norm()),
        residual_top_singular_value=float(s[0]), random_A_norm=float(rand.norm()), residual_A_norm=float(residual.norm()),
        random_orthogonality_error=float((qa.T @ rand.T).abs().max()),
        residual_orthogonality_error=float((qa.T @ residual.T).abs().max()))


def evaluate(model, examples, device, reference=None):
    """Real causal label masking; token-weighted NLL, full-logit byte digests."""
    modes = [(m, m.training) for m in model.modules()]; rows = []
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            for i, example in enumerate(examples):
                item = batch(example, device)
                count = int((item['labels'][:, 1:] != -100).sum())
                must(count > 0, 'No shifted supervised tokens')
                out = model(**item, use_cache=False)
                loss = float(out.loss.double())
                must(math.isfinite(loss) and torch.isfinite(out.logits).all(), 'Nonfinite evaluation')
                logits = out.logits.detach().contiguous().cpu()
                digest = hashlib.sha256(logits.view(torch.uint8).numpy().tobytes()).hexdigest()
                row = dict(index=i, effective_tokens=count, nll_sum=loss*count, nll=loss,
                           logits_sha256=digest, logits_shape=list(logits.shape), logits_dtype=str(logits.dtype))
                if reference is not None:
                    must(all(row[k] == reference[i][k] for k in ('logits_sha256','logits_shape','logits_dtype')),
                         f'Full-model insertion logits differ at validation sample {i}')
                rows.append(row)
    finally:
        for m, mode in modes: m.training = mode
    must(bool(rows), 'Empty validation')
    return dict(token_weighted_nll=math.fsum(r['nll_sum'] for r in rows)/sum(r['effective_tokens'] for r in rows),
                effective_tokens=sum(r['effective_tokens'] for r in rows), num_samples=len(rows), samples=rows)


def optimize(model, target, examples, order, device, lr, weight_decay, output):
    """Fresh optimizer, only target factors; constant LR and no dropout in every arm."""
    for p in model.parameters(): p.requires_grad_(False)
    base = target.base_layer if isinstance(target, ExpandedLoRA) else target
    factors = dict(old_A=base.weight_a, old_B=base.weight_b)
    if isinstance(target, ExpandedLoRA): factors.update(new_A=target.add_a, new_B=target.add_b)
    for p in factors.values(): p.requires_grad_(True)
    must(sum(p.numel() for p in model.parameters() if p.requires_grad) == sum(p.numel() for p in factors.values()),
         'Unexpected trainable parameters')
    opt = torch.optim.AdamW(list(factors.values()), lr=lr, weight_decay=weight_decay)
    initial = {k: p.detach().clone() for k, p in factors.items()}
    model.eval()  # autograd enabled; eliminate different extra-branch dropout RNG consumption
    rows = []
    for step, index in enumerate(order, 1):
        opt.zero_grad(set_to_none=True)
        item = batch(examples[index], device)
        count = int((item['labels'][:, 1:] != -100).sum())
        must(count > 0, 'No training supervised tokens')
        loss = model(**item, use_cache=False).loss
        must(torch.isfinite(loss), f'Nonfinite training loss at step {step}')
        loss.backward()
        norms = {}
        for key, p in factors.items():
            must(p.grad is not None and torch.isfinite(p.grad).all(), f'Invalid gradient {key}')
            norms[key] = float(p.grad.float().norm())
        opt.step()
        must(all(torch.isfinite(p).all() for p in factors.values()), 'Nonfinite trained factor')
        row = dict(step=step, train_pool_index=index, loss=float(loss.detach()), effective_tokens=count, gradient_norms=norms, lr=lr)
        rows.append(row)
        if output is not None:
            with Path(output).open('a') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
    opt.zero_grad(set_to_none=True)
    changes = {k: not tensor_exact(initial[k], p) for k, p in factors.items()}
    if isinstance(target, ExpandedLoRA):
        must(changes['new_A'] and changes['new_B'], 'New rank factors did not both update')
        must(any(r['gradient_norms']['new_A'] > 0 for r in rows[1:])
             and any(r['gradient_norms']['new_B'] > 0 for r in rows),
             'New factor changes must include real LM gradients, not just weight decay')
    return dict(steps=rows, parameter_changes=changes, trainable_parameters=sum(p.numel() for p in factors.values()),
                optimizer='fresh AdamW; NOT historical step1600 state', scheduler='none; constant LR',
                dropout='disabled for all arms', optimizer_config=opt.defaults)


def run(args, report):
    must(torch.cuda.is_available(), 'CUDA unavailable; full experiment requires real GPU')
    free, total = torch.cuda.mem_get_info()
    must(free >= args.min_free_gib * 2**30, f'Insufficient free GPU memory: {free/2**30:.2f} GiB')
    report['gpu'] = dict(name=torch.cuda.get_device_name(), free_bytes=free, total_bytes=total)
    report['versions'] = {k: importlib.metadata.version(k) for k in ('torch','transformers','safetensors','accelerate')}
    report['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    report['git_status'] = subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True)
    inputs = [args.adapter, args.train_data, args.dev_data, args.source_data, Path(args.model)/'config.json']
    inputs += sorted(Path(args.model).glob('*.safetensors')) + sorted(Path(args.model).glob('*.index.json'))
    report['input_sha256'] = {str(p): sha256(p) for p in inputs}
    report['code_sha256'] = {str(p.relative_to(ROOT)):sha256(p) for p in
        (Path(__file__), ROOT/'rank_gain_probe_support.py', ROOT/'finetune/adaptive_lora/rank_expansion.py',
         ROOT/'finetune/adaptive_lora/training_data.py')}
    tm = build_source_metadata(args.train_data, args.source_data)
    vm = build_source_metadata(args.dev_data, args.source_data)
    splits = isolated_split(tm, vm, args.train_samples, args.validation_samples, args.probe_batches, args.seed)
    provenance = dict(indices=splits, train_metadata=tm, dev_metadata=vm,
                      boundary='Disjoint within this experiment; PRE adapter has already seen historical training data. Dev is not blind.')
    write_json(args.output_dir/'data_manifest.json', provenance)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    if tokenizer.chat_template is None: tokenizer.chat_template = (Path(args.model)/'chat_template.jinja').read_text()
    tokenizer.pad_token = tokenizer.eos_token
    datasets = {}
    for role, indices in splits.items():
        source = args.dev_data if role == 'validation' else args.train_data
        source_rows = json.loads(source.read_text())
        path = args.output_dir/f'{role}_selected.json'
        write_json(path, [source_rows[i] for i in indices])
        datasets[role] = ChatTrainingDataset(path, tokenizer, args.max_length, dynamic_length=True)
    order = []; rng = random.Random(args.seed)
    while len(order) < args.steps:
        epoch = list(range(args.train_samples)); rng.shuffle(epoch); order.extend(epoch)
    order = order[:args.steps]
    report['training_order_source_indices'] = [splits['train'][i] for i in order]
    payload = torch.load(args.adapter, map_location='cpu', weights_only=True)
    must(Path(payload['base_model']).resolve() == Path(args.model).resolve(), 'Adapter base mismatch')
    model, report['loading'] = load_probe_model(args.model, torch.bfloat16)
    model.to('cuda'); model.config.use_cache = False
    base_parameters = dict(model.named_parameters())
    names = restore_dense_adapter_on_full_model(model, payload, ROOT)
    expected = {f'model.layers.{i}.self_attn.{k}' for i in range(42) for k in ('q_proj','v_proj')}
    must(set(names) == expected and len(names) == 84, 'Expected all 84 Dense adapters')
    layers = {n: model.get_submodule(n) for n in names}
    factor_objects = {n: {k: getattr(l,k) for k in ('weight_a','weight_b')} for n,l in layers.items()}
    snapshots = {n: {k:getattr(l,k).detach().clone() for k in ('weight_a','weight_b')} for n,l in layers.items()}
    for n in names:
        for k in snapshots[n]:
            must(tensor_exact(snapshots[n][k].cpu(), payload['layers'][n]['state'][k]), f'Inexact adapter load: {n}.{k}')
    target = layers[args.module]; scaling = target.lora_scaler
    must(target.lora_rank == 4 and target.weight_a.shape[0] == target.weight_b.shape[1] == 4, 'Expected old rank 4')
    report['num_loaded_adapters'] = len(names)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    gradient, probe_losses = averaged_gradient(model, target, [batch(x,'cuda') for x in datasets['probe']])
    random_a, residual_a, report['direction_diagnostics'] = new_directions(target.weight_a, target.weight_b, gradient, args.seed)
    report['probe'] = dict(batch_losses=probe_losses, seconds=time.perf_counter()-start,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(), gradient_average='equal per-example masked loss')
    del gradient
    parent, _, child = args.module.rpartition('.')
    report['arms'] = {}
    baseline = None
    for arm, direction in [('A', None), ('B', random_a), ('C', residual_a)]:
        setattr(model.get_submodule(parent), child, target)
        with torch.no_grad():
            for k,v in snapshots[args.module].items(): getattr(target,k).copy_(v)
        for p in model.parameters(): p.grad = None
        selected = target if direction is None else ExpandedLoRA(target, direction, torch.zeros(target.out_features,1,device='cuda'))
        setattr(model.get_submodule(parent), child, selected)
        must(target.lora_rank == 4 and target.lora_scaler == scaling, 'Old rank/scaling changed')
        for k,v in snapshots[args.module].items(): must(tensor_exact(getattr(target,k),v), 'Old factor changed at insertion')
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
        before = evaluate(model, datasets['validation'], 'cuda', baseline)
        if arm == 'A': baseline = before['samples']
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); random.seed(args.seed)
        train_start = time.perf_counter()
        training = optimize(model, selected, datasets['train'], order, 'cuda', args.lr, args.weight_decay,
                            args.output_dir/f'{arm}_train.jsonl')
        torch.cuda.synchronize(); training_seconds = time.perf_counter()-train_start
        after = evaluate(model, datasets['validation'], 'cuda')
        torch.cuda.synchronize()
        for n,l in layers.items():
            for k,p in factor_objects[n].items():
                must(getattr(l,k) is p, f'Original factor Parameter replaced: {n}.{k}')
            if n != args.module:
                must(model.get_submodule(n) is l, f'Other layer replaced: {n}')
                for k,v in snapshots[n].items(): must(tensor_exact(getattr(l,k), v), f'Other factor changed: {n}.{k}')
        must(target.lora_rank == 4 and target.lora_scaler == scaling, 'Old rank/scaling changed during training')
        verified = verify_disk_weights(base_parameters, args.model)
        factors = {k:getattr(target,k).detach().cpu().clone() for k in ('weight_a','weight_b')}
        if direction is not None: factors.update(add_a=selected.add_a.detach().cpu(), add_b=selected.add_b.detach().cpu())
        torch.save(dict(module=args.module, arm=arm, factors=factors, old_scaling=scaling,
                        new_branch_scaling=1.0, format='isolated selected factors; NOT full adapter or PEFT'),
                   args.output_dir/f'{arm}_target_factors.pt')
        result = dict(status='PASS', capacity=4 if arm=='A' else 5, old_scaling=scaling, new_branch_scaling=None if arm=='A' else 1.0,
            before=before, after=after, training=training, seconds=time.perf_counter()-start,
            training_seconds=training_seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), total_model_parameters=sum(p.numel() for p in model.parameters()),
            total_adapter_parameters=sum(v.numel() for x in snapshots.values() for v in x.values())+(0 if arm=='A' else selected.add_a.numel()+selected.add_b.numel()),
            invariants=dict(insertion_full_logits_match_A=True, other_83_adapters_unchanged=True,
                            base_tensors_verified=verified, original_factor_objects_preserved=True, old_scaling_unchanged=True))
        report['arms'][arm] = result
        write_json(args.output_dir/f'{arm}_results.json', result)
        print(f'{arm}: validation NLL {before["token_weighted_nll"]:.8f} -> {after["token_weighted_nll"]:.8f}', flush=True)
        write_json(args.output_dir/'report.json', report)
        del training, selected
        gc.collect()
    must(all(sha256(p) == h for p,h in report['input_sha256'].items()), 'Input files changed during experiment')
    report['input_files_unchanged'] = True
    report['historical_AdamW_verified'] = False
    report['status'] = 'PASS'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=MODEL)
    p.add_argument('--adapter', type=Path, default=ADAPTER)
    p.add_argument('--module', choices=MODULES, default=MODULES[0])
    p.add_argument('--train-data', type=Path, default=DATA/'train.json')
    p.add_argument('--dev-data', type=Path, default=DATA/'dev.json')
    p.add_argument('--source-data', type=Path, default=Path('/home/hongchang/whf/datasets/MetaMathQA/MetaMathQA-395K.jsonl'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--train-samples', type=int, default=64)
    p.add_argument('--validation-samples', type=int, default=64)
    p.add_argument('--probe-batches', type=int, default=4)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--min-free-gib', type=float, default=30.0)
    return p


def main():
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  historical_AdamW_verified=False)
    try:
        must(args.steps >= 2 and min(args.train_samples,args.validation_samples,args.probe_batches)>0, 'Invalid sample/step counts')
        must(math.isfinite(args.lr) and args.lr>0 and math.isfinite(args.weight_decay) and args.weight_decay>=0, 'Invalid optimizer config')
        run(args, report)
    except BaseException as exc:
        report.update(status='FAIL', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.output_dir/'report.json', report)


if __name__ == '__main__':
    main()
