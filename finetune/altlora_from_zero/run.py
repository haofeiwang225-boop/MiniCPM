"""Explicit commands: prepare once, then three independent 2000-update arms.
No model import/loading occurs until prepare/train is explicitly requested.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import traceback
import uuid
import time
from .protocol import *

ROOT=Path(__file__).resolve().parents[2]


def code_hashes():
    paths=list((ROOT/'finetune/altlora_from_zero').rglob('*.py'))
    paths+=list((ROOT/'finetune/altlora').glob('*.py'))
    paths += [ROOT/p for p in ('global84_rank_direction.py','rank_gain_probe_support.py',
        'finetune/adaptive_lora/training_data.py','finetune/adaptive_lora/data_provenance.py',
        'finetune/adaptive_lora/gid_validation.py','finetune/adaptive_lora/subspace_matching.py',
        'finetune/adaptive_lora/tangent_metrics.py','finetune/adaptive_lora/residual_gid.py',
        'third_party/altlora_official/AltLoRA-main/optimizer_new.py',
        'third_party/altlora_official/AltLoRA-main/logtrainer.py',
        'benchmarks/math500/evaluate.py','benchmarks/math500/answer_parser.py',
        'benchmarks/math500/grader.py','benchmarks/math500/prompt.py')]
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(set(paths))}


def environment():
    import torch,numpy,transformers
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,
                transformers=transformers.__version__,cuda=torch.version.cuda)


def load_model(config):
    import torch
    from rank_gain_probe_support import load_probe_model
    from .inputs import verify_model
    verify_model(config['model'],config['base_hashes'])
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Exactly one visible CUDA GPU required')
    free,_=torch.cuda.mem_get_info()
    if free < config.get('minimum_free_gpu_gib',28)*1024**3:raise RuntimeError('Insufficient free GPU memory for requested preflight floor')
    model,info=load_probe_model(Path(config['model']),torch.bfloat16)
    verify_model(config['model'],config['base_hashes'])
    model.to('cuda');model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model,info


def datasets(config):
    from transformers import AutoTokenizer
    from adaptive_lora.training_data import ChatTrainingDataset
    from .inputs import verify_model
    tok=AutoTokenizer.from_pretrained(config['model'],local_files_only=True,trust_remote_code=True)
    tok.pad_token=tok.eos_token
    if tok.chat_template is None:tok.chat_template=(Path(config['model'])/'chat_template.jinja').read_text()
    verify_model(config['model'],config['base_hashes'])
    return (ChatTrainingDataset(config[k]['path'],tok,2048,dynamic_length=True) for k in ('train','probe'))


def batch(data,index,device='cuda'):
    return {k:v.unsqueeze(0).to(device) for k,v in data[index].items()}


def prepare(config, output):
    import torch
    from altlora.adapters import adapter_state
    from .inputs import audit
    from .initialization import attach_pattern,assert_zero_b
    from .state import write_json,atomic_save,preserved,set_step_rng
    from .diagnostics import gradients,coverage_rows
    require_fresh_output(output)
    import shutil
    parent=output.parent
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free < config.get('minimum_free_disk_gib',40)*1024**3:raise RuntimeError('Insufficient disk preflight floor')
    audited=audit(config)  # BEFORE any model loading or output creation
    order=training_order(config['train']['path'])
    output.mkdir(parents=True);shared=output/'shared';shared.mkdir()
    try:
        config=dict(config,version=VERSION,environment=environment(),code_hashes=code_hashes(),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            protocol=dict(steps=2000,seed=42,batch_size=1,accumulation=1,lr=5e-5,
                scheduler='LambdaLR 1-completed/2000',warmup=0,weight_decay=.01,dropout=.1,
                loss='per-question masked mean',rng='paired per-step reseeding v1',stages=list(STAGES)))
        write_json(shared/'config.json',config);write_json(shared/'rank_map.json',audited['ranks'])
        write_json(shared/'train_order.json',order);write_json(shared/'data_audit.json',audited)
        model,loading=load_model(config);train,probe_data=datasets(config)
        with preserved(model),torch.no_grad():
            model.eval();base_logits=model(**batch(train,order['samples'][0]['index'])).logits.detach().cpu()
        set_step_rng(42,0)
        layers=attach_pattern(model,audited['ranks']['pattern']);assert_zero_b(layers)
        with preserved(model),torch.no_grad():
            model.eval();after=model(**batch(train,order['samples'][0]['index'])).logits.detach().cpu()
        if not torch.equal(base_logits,after):raise ValueError('Initial adapter changed complete Base logits')
        atomic_save(shared/'initial_adapter.pt',adapter_state(layers))
        atomic_save(shared/'base_logits.pt',base_logits)
        del base_logits,after
        for split,ids in config['probe_splits'].items():
            gs,profile=gradients(model,layers,[batch(probe_data,i,'cpu') for i in ids])
            atomic_save(shared/f'G0_{split}.pt',gs)
            write_json(shared/f'G0_{split}_profile.json',profile)
            write_json(shared/f'G0_{split}_metrics.json',coverage_rows(layers,gs,gs,0,split));del gs
        identity=dict(rank_map_sha256=audited['ranks']['rank_map_sha256'],
            initial_adapter_sha256=file_hash(shared/'initial_adapter.pt'),
            train_order_sha256=order['manifest_sha256'],base_hashes=config['base_hashes'],
            code_hashes=config['code_hashes'],protocol=config['protocol'],environment=config['environment'],
            config_sha256=file_hash(shared/'config.json'),
            shared_hashes={p.name:file_hash(p) for p in shared.iterdir() if p.is_file()})
        identity['pair_id']=canonical_hash(identity)
        write_json(shared/'identity.json',identity)
        write_json(shared/'report.json',dict(status='PASS',loading=loading,base_logits_equal=True,
            base_logits_sha256=file_hash(shared/'base_logits.pt'),initial_B_zero=True))
    except BaseException:
        write_json(output/'prepare_failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise


def train_arm(output,arm,resume=None):
    import torch
    from altlora.adapters import factor_pairs,adapter_state
    from altlora.optimizer import OfficialAltLoRA,OfficialSchedule
    from rank_gain_probe_support import verify_disk_weights
    from .initialization import attach_pattern,restore_initial
    from .inputs import verify_model
    from .state import (write_json,atomic_save,preserved,set_step_rng,rng_fingerprint,
                        save_checkpoint,load_checkpoint)
    from .diagnostics import gradients,coverage_rows,update_rows
    shared=output/'shared';config=json.loads((shared/'config.json').read_text())
    identity=json.loads((shared/'identity.json').read_text())
    if json.loads((shared/'report.json').read_text())['status']!='PASS':raise ValueError('Prepare not complete')
    if identity['code_hashes']!=code_hashes() or identity['environment']!=environment():raise ValueError('Code/runtime changed')
    for name,digest in identity['shared_hashes'].items():
        if file_hash(shared/name)!=digest:raise ValueError('Shared artifact changed: '+name)
    for key in ('train','probe','source','gsm8k','math500','rank_plan','historical_config'):
        if file_hash(config[key]['path'])!=config[key]['sha256']:raise ValueError('Input changed: '+key)
    identity=dict(identity,arm=arm)
    directory=output/arm
    if resume is None:directory.mkdir(exist_ok=False)
    else:
        if not directory.is_dir() or not resume.resolve().is_relative_to(directory.resolve()):raise ValueError('Resume must belong to this arm')
        if (directory/'report.json').exists():raise ValueError('Completed arm cannot resume')
    attempt=directory/('attempt_'+uuid.uuid4().hex);attempt.mkdir()
    try:
        model,loading=load_model(config);base=dict(model.named_parameters())
        ranks=json.loads((shared/'rank_map.json').read_text())
        layers=attach_pattern(model,ranks['pattern'])
        initial=torch.load(shared/'initial_adapter.pt',map_location='cpu',weights_only=True)
        restore_initial(layers,initial);del initial
        train,probe_data=datasets(config);order=json.loads((shared/'train_order.json').read_text())
        with preserved(model),torch.no_grad():
            model.eval();actual=model(**batch(train,order['samples'][0]['index'])).logits.detach().cpu()
            expected=torch.load(shared/'base_logits.pt',map_location='cpu',weights_only=True)
            if not torch.equal(actual,expected):raise ValueError('Arm Step0 logits differ from Base')
        del actual,expected
        pairs=factor_pairs(layers);params=[p for _,a,b in pairs for p in (a,b)]
        opt=(torch.optim.AdamW(params,lr=5e-5,weight_decay=.01) if arm=='vanilla'
             else OfficialAltLoRA(pairs,lr=5e-5,weight_decay=.01))
        if len({id(p) for p in params})!=168 or sum(p.numel() for p in params)!=480000:raise ValueError('Invalid optimizer factors')
        if arm=='vanilla' and (opt.defaults['betas']!=(.9,.999) or opt.defaults['eps']!=1e-8):raise ValueError('Unexpected AdamW defaults')
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda k:max(0,1-k/2000))
        phase=OfficialSchedule(pairs,'official-trainer' if arm=='altlora_true' else 'official-main')
        write_json(attempt/'optimizer_spec.json',dict(arm=arm,defaults=opt.defaults,loading=loading,
            official=arm!='vanilla',fresh_state=resume is None))
        start=load_checkpoint(resume,layers,opt,scheduler,phase,order,identity) if resume else 0
        history=[] if not resume else torch.load(resume,map_location='cpu',weights_only=True)['extra']['history']
        records=[] if not resume else torch.load(resume,map_location='cpu',weights_only=True)['extra']['records']
        reference_step=0
        refs=None
        reference_file=None
        fixed_construction=torch.load(shared/'G0_construction.pt',map_location='cpu',weights_only=True)
        def measure(step):
            nonlocal refs,reference_step,reference_file
            for split,ids in config['probe_splits'].items():
                fixed=(fixed_construction if split=='construction' else torch.load(shared/f'G0_{split}.pt',map_location='cpu',weights_only=True))
                current,profile=gradients(model,layers,[batch(probe_data,i,'cpu') for i in ids]) if step else (fixed,json.loads((shared/f'G0_{split}_profile.json').read_text()))
                path=attempt/f'metrics_{step}_{split}.json'
                write_json(path,dict(step=step,split=split,profile=profile,rows=coverage_rows(layers,current,fixed,step,split)))
                records.append(str(path.relative_to(output)))
                if split=='construction':
                    refs=current;reference_step=step
                    reference_file=(shared/'G0_construction.pt' if step==0 else attempt/f'construction_gradient_{step}.pt')
                    if step:atomic_save(reference_file,current)
                else:del current,fixed
        if resume:
            extra=torch.load(resume,map_location='cpu',weights_only=True)['extra']
            reference_file=output/extra['reference_file']
            if file_hash(reference_file)!=extra['reference_sha256']:raise ValueError('Checkpoint reference gradient changed')
            refs=torch.load(reference_file,map_location='cpu',weights_only=True)
            reference_step=extra['reference_step']
        else:
            measure(0)
            save_checkpoint(attempt/'checkpoint_0.pt',layers,opt,scheduler,phase,0,order,identity,
                extra=dict(history=history,records=records,reference_file=str(reference_file.relative_to(output)),
                           reference_step=reference_step,reference_sha256=file_hash(reference_file)))
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for completed in range(start,2000):
            model.train();side=phase.prepare()
            if side!=expected_side(arm,completed+1):raise ValueError('Official schedule mismatch')
            old={n:(m.weight_a.detach().cpu().clone(),m.weight_b.detach().cpu().clone()) for n,m in layers.items()}
            frozen={p: (p.detach().clone(),opt.state.get(p),{k:v._version for k,v in opt.state.get(p,{}).items() if torch.is_tensor(v)}) for p in params if not p.requires_grad}
            rng=set_step_rng(42,completed+1)
            atomic_save(attempt/f'rng_{completed+1}.pt',rng)
            loss=model(**batch(train,order['samples'][completed]['index'])).loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            activity={n:{k:dict(active=p.requires_grad,grad_norm=None if p.grad is None else float(p.grad.norm())) for k,p in [('A',m.weight_a),('B',m.weight_b)]} for n,m in layers.items()}
            for p in params:
                if p.grad is not None and not torch.isfinite(p.grad).all():raise ValueError('Nonfinite factor gradient')
            used_lr=opt.param_groups[0]['lr']
            if abs(used_lr-learning_rate(completed))>1e-14:raise ValueError('LR drift')
            opt.step()
            if any(not torch.isfinite(v).all() for state in opt.state.values() for v in state.values() if torch.is_tensor(v)):
                raise ValueError('Nonfinite optimizer state')
            for p,(value,state,versions) in frozen.items():
                if p.grad is not None or not torch.equal(p,value) or opt.state.get(p) is not state or any(state[k]._version!=v for k,v in versions.items()):raise ValueError('Frozen factor/momentum changed')
            scheduler.step();opt.zero_grad(set_to_none=True)
            if any(not torch.isfinite(p).all() for p in params):raise ValueError('Nonfinite parameters')
            step=completed+1
            row=dict(step=step,index=order['samples'][completed]['index'],loss=float(loss.detach()),lr=used_lr,
                side=side,rng_sha256=rng_fingerprint(rng),activity=activity,
                factor_changes={n:dict(A=float((m.weight_a.detach().cpu()-old[n][0]).norm()),B=float((m.weight_b.detach().cpu()-old[n][1]).norm())) for n,m in layers.items()},
                optimizer_steps={n:{k:int(opt.state.get(p,{}).get('step',0)) for k,p in [('A',m.weight_a),('B',m.weight_b)]} for n,m in layers.items()})
            # Matrix diagnostics only on scheduled endpoint updates; all steps retain factor activity.
            if step in STAGES:
                path=attempt/f'updates_{step}.json';write_json(path,update_rows(layers,old,refs,fixed_construction,completed,reference_step))
                records.append(str(path.relative_to(output)))
                measure(step)
            history.append(row);write_json(attempt/f'step_{step}.json',row)
            if step in CHECKPOINT_STEPS:
                save_checkpoint(attempt/f'checkpoint_{step}.pt',layers,opt,scheduler,phase,step,order,identity,extra=dict(history=history,records=records,reference_file=str(reference_file.relative_to(output)),reference_step=reference_step,reference_sha256=file_hash(reference_file)))
        verify_disk_weights(base,Path(config['model']));verify_model(config['model'],config['base_hashes'])
        final_state=adapter_state(layers)
        if (directory/'final_adapter.pt').exists():
            previous=torch.load(directory/'final_adapter.pt',map_location='cpu',weights_only=True)
            if set(previous)!=set(final_state) or any(previous[n]['spec']!=v['spec'] or not torch.equal(previous[n]['A'],v['A']) or not torch.equal(previous[n]['B'],v['B']) for n,v in final_state.items()):
                raise ValueError('Previously published final adapter differs from restored endpoint')
        else:atomic_save(directory/'final_adapter.pt',final_state)
        write_json(directory/'report.json',dict(status='PASS',complete=True,identity=identity,steps=2000,
            history=history,records=records,final_adapter_sha256=file_hash(directory/'final_adapter.pt'),
            latest_checkpoint=str(resume if start==2000 else attempt/'checkpoint_2000.pt'),base_verified=True,
            attempt_seconds=time.perf_counter()-started,peak_cuda_allocated=torch.cuda.max_memory_allocated(),
            rank_activity={str(r):{side:{'active_steps':sum(int(row['activity'][n][side]['active']) for row in history for n,m in layers.items() if m.rank==r),
                'frozen_steps':sum(int(not row['activity'][n][side]['active']) for row in history for n,m in layers.items() if m.rank==r)} for side in ('A','B')} for r in (1,2,3,4)}))
    except BaseException:
        write_json(attempt/'failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','train'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--resume',type=Path)
    a=parser.parse_args()
    if a.command=='prepare':
        if a.config is None or a.resume or a.arm:parser.error('prepare requires config, forbids resume/arm')
        prepare(json.loads(a.config.read_text()),a.output)
    else:
        if a.arm is None or a.config:parser.error('train requires arm; config comes only from shared state')
        train_arm(a.output,a.arm,a.resume)

if __name__=='__main__':main()
