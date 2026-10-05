"""Read-only Dense LoRA diagnostics. No optimizer, training, correction or expansion."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def gaps(rows):
    result=[]
    energy=sum(r['gradient_fro_norm']**2 for r in rows)
    for row in rows:
        r=dict(row)
        dg=r['oracle_tangent_coverage']-r['tangent_coverage']
        cg=1-r['oracle_tangent_coverage']
        r.update(direction_gap=dg, capacity_gap=cg,
                 gap_class='balanced' if abs(dg-cg)<1e-8 else
                 'direction_dominant' if dg>cg else 'capacity_dominant')
        result.append(r)
    summary=dict(global_gradient_energy=energy, counts=dict(Counter(r['gap_class'] for r in result)))
    for key in ('direction_gap','capacity_gap','tangent_coverage','oracle_tangent_coverage'):
        summary['global_energy_weighted_'+key]=(sum(r[key]*r['gradient_fro_norm']**2 for r in result)/energy
                                               if energy else None)
    return result,summary


def save_rows(folder, rows):
    (folder/'modules.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))


def offline(args):
    if args.mode=='gaps':
        for file in args.inputs:
            folder=args.output_dir/Path(file).parent.name
            folder.mkdir(exist_ok=False)
            rows,summary=gaps(records(file))
            save_rows(folder,rows);write(folder/'summary.json',summary)
    elif args.mode=='compare':
        tables=[{r['module']:r for r in records(p)} for p in args.inputs]
        if any(set(t)!=set(tables[0]) for t in tables):
            raise ValueError('Cannot compare different module sets')
        rows=[]
        keys=('gid_full','gradient_fro_norm','oracle_tangent_coverage','tangent_coverage',
              'direction_efficiency','gid_residual','residual_rank_k90','residual_fro_norm','allocated_rank')
        for name in tables[0]:
            row=dict(module=name)
            for key in keys:
                values=[t[name].get(key) for t in tables]
                if all(v is not None for v in values):
                    row[key]=values
                    row[key+'_range']=max(values)-min(values)
            rows.append(row)
        save_rows(args.output_dir,rows)
        write(args.output_dir/'summary.json',dict(inputs=args.inputs,num_modules=len(rows),
            max_ranges={k:max(r[k+'_range'] for r in rows) for k in keys if k+'_range' in rows[0]},
            rank_distributions=[dict(Counter(r.get('allocated_rank',r.get('rank')) for r in t.values())) for t in tables],
            modules_with_changed_rank=sum(r.get('allocated_rank_range',0)>0 for r in rows)))


def model_probe(args):
    import torch
    from transformers import AutoTokenizer, default_data_collator, set_seed
    from adaptive_lora.model_loading import load_training_model
    from adaptive_lora.training_data import ChatTrainingDataset
    from adaptive_lora.checkpoint import load_adapter
    from adaptive_lora.official import dense_factors, LinearWithLoRA
    from adaptive_lora.gradient_probe import probe_weight
    from adaptive_lora.residual_gid import svd_with_gid
    from adaptive_lora.residual_gradient import residual_gradient
    from adaptive_lora.subspace_matching import match_subspaces, orth
    from adaptive_lora.tangent_metrics import tangent_metrics, frobenius_metrics, residual_energy_rank
    from adaptive_lora.dense_init import allocate_dense_ranks
    config=read(args.run_dir/'run_config.json')
    write(args.output_dir/'source_run_config.json',config)
    set_seed(args.seed)
    manifest_path=args.run_dir/'calibration_indices.json'
    manifest=read(manifest_path)
    expected=config.get('calibration_indices_sha256')
    if expected and hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Source calibration manifest hash mismatch')
    if manifest['split']!='train': raise ValueError('Only training calibration is allowed')
    tokenizer=AutoTokenizer.from_pretrained(config['model_name_or_path'],trust_remote_code=True,local_files_only=True)
    if tokenizer.chat_template is None:
        tokenizer.chat_template=(Path(config['model_name_or_path'])/'chat_template.jinja').read_text()
    tokenizer.pad_token=tokenizer.eos_token
    dataset=ChatTrainingDataset(config['train_data_path'],tokenizer,config['model_max_length'])
    if len(dataset)!=manifest['dataset_length']: raise ValueError('Training dataset length changed')
    if args.indices is None:
        groups=manifest['batch_indices'][:args.k]
    else:
        bs=config['batch_size']
        groups=[args.indices[i:i+bs] for i in range(0,len(args.indices),bs)]
    ids=[i for group in groups for i in group]
    if len(groups)!=args.k or len(ids)!=len(set(ids)) or any(i<0 or i>=len(dataset) for i in ids):
        raise ValueError('Provide K distinct fixed training batches; no cycling')
    write(args.output_dir/'calibration_indices.json',dict(manifest,batch_indices=groups,num_calibration_batches=args.k))
    batches=[default_data_collator([dataset[i] for i in group]) for group in groups]
    model=load_training_model(config['model_name_or_path'],getattr(torch,config['dtype']),config['model_family']).to(args.device)
    model.config.use_cache=False
    if config.get('gradient_checkpointing'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if args.mode=='initial':
        names=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)
               and n.rsplit('.',1)[-1] in config['target_modules']]
    else:
        names=load_adapter(model,args.adapter)
        if any(type(model.get_submodule(n)) is not LinearWithLoRA for n in names):
            raise ValueError('This CLI requires ordinary Dense adapters, without block or expanded wrappers')
    if set(names)!=set(config['targets']): raise ValueError('Target modules differ from source run')
    if args.modules:
        if not set(args.modules)<=set(names): raise ValueError('Unknown requested module')
        names=[n for n in names if n in args.modules]
    if args.oracle_modules and not set(args.oracle_modules)<=set(names):
        raise ValueError('Oracle test requests modules outside selected targets')
    model.eval()
    failures=[]
    for repeat in range(args.repeats):
        set_seed(args.seed)  # identical model/checkpoint/data/RNG for repeatability
        folder=args.output_dir/f'repeat_{repeat+1}' if args.repeats>1 else args.output_dir
        folder.mkdir(exist_ok=True)
        rows=[]
        for index,name in enumerate(names):
            layer=model.get_submodule(name)
            g,losses=probe_weight(model,layer,batches,return_losses=True)
            g=g.to(args.device)
            u,s,vh,gid=svd_with_gid(g)
            row=dict(module=name,gid_full=gid,shape=list(g.shape),effective_rank=gid,
                     gradient_fro_norm=float(g.double().norm()),probe_batch_losses=losses,
                     mean_probe_loss=sum(losses)/len(losses),num_calibration_batches=args.k,
                     base_weight_unchanged=True)
            if args.mode!='initial':
                a,b=dense_factors(layer);a,b=a.to(g.device),b.to(g.device)
                qa,qb,ca,cb=match_subspaces(a,b,u,s,vh)
                r=residual_gradient(g,qa,qb)
                if r.norm() <= 1e-6*g.norm(): r.zero_()
                _,sr,_,gidr=svd_with_gid(r)
                rank=a.shape[0]
                metrics=tangent_metrics(g,r,s,rank,ca,cb)
                row.update(rank=rank,coverage_A=ca,coverage_B=cb,gid_residual=gidr,
                           residual_rank_k90=residual_energy_rank(sr),**metrics)
                row.update(frobenius_metrics(g,r,row['residual_energy_ratio']))
                if args.random_trials:
                    generator=torch.Generator(device=g.device).manual_seed(args.seed+index)
                    values=[]
                    for _ in range(args.random_trials):
                        ra=orth(torch.randn(g.shape[1],rank,device=g.device,generator=generator))
                        rb=orth(torch.randn(g.shape[0],rank,device=g.device,generator=generator))
                        rr=residual_gradient(g,ra,rb)
                        values.append(float((g-rr).double().square().sum()/(g.double().square().sum()+1e-30)))
                    mean=sum(values)/len(values)
                    std=(sum((x-mean)**2 for x in values)/max(1,len(values)-1))**.5
                    row.update(random_coverages=values,random_mean=mean,random_std=std,
                        actual_minus_random_mean=row['tangent_coverage']-mean,
                        random_empirical_upper_tail=(1+sum(x>=row['tangent_coverage'] for x in values))/(len(values)+1))
                if args.oracle_modules and name in args.oracle_modules:
                    # Complementary singular directions, with full-width bases if 2r>min(shape).
                    m=len(s); indices=list(range(rank,min(2*rank,m)))+list(range(max(0,2*rank-m)))
                    rr=residual_gradient(g,vh[indices].T,u[:,:rank])
                    actual=float((g-rr).double().square().sum()/(g.double().square().sum()+1e-30))
                    error=abs(actual-row['oracle_tangent_coverage'])
                    passed=error<=args.tolerance
                    row.update(constructed_oracle_coverage=actual,oracle_construction_error=error,oracle_test_passed=passed)
                    if not passed: failures.append(name)
                if args.save_gradients:
                    cache=folder/'gradient_cache';cache.mkdir(exist_ok=True)
                    valid=row['valid_singular_values'];d=min(args.drift_rank,valid)
                    torch.save(dict(g=g.cpu(),qa=qa.cpu(),qb=qb.cpu(),u=u[:,:d].cpu(),v=vh[:d].T.cpu()),cache/(name+'.pt'))
                del a,b,qa,qb,r,sr
            rows.append(row)
            print(f'{repeat+1}: {name}',flush=True)
            del g,u,s,vh
        if args.mode=='initial':
            ranks,total=allocate_dense_ranks(rows,args.budget,config['dense_rank_min'],config['dense_rank_max'])
            for row in rows: row['allocated_rank']=ranks[row['module']]
            summary=dict(total_lora_params=total,parameter_budget=args.budget,rank_pattern=ranks,
                         rank_distribution=dict(Counter(ranks.values())))
        else:
            rows,summary=gaps(rows)
        summary.update(num_modules=len(rows),num_calibration_batches=args.k,
                       mean_probe_loss=rows[0]['mean_probe_loss'],probe_batch_losses=rows[0]['probe_batch_losses'],
                       oracle_failures=failures,model_training=model.training)
        save_rows(folder,rows);write(folder/'summary.json',summary)
    if failures: raise RuntimeError(f'Oracle construction failed, see saved results: {failures}')


def drift(args):
    import torch
    from adaptive_lora.residual_gradient import residual_gradient
    folders=[Path(p) for p in args.inputs]
    manifests=[read(f/'calibration_indices.json') for f in folders]
    if any(m!=manifests[0] for m in manifests): raise ValueError('Drift requires identical calibration manifests')
    configs=[read(f/'diagnostic_config.json') for f in folders]
    if any(c['drift_rank']!=configs[0]['drift_rank'] for c in configs):
        raise ValueError('Drift subspace widths differ across caches')
    files=[set(p.name for p in (f/'gradient_cache').glob('*.pt')) for f in folders]
    if not files[0] or any(f!=files[0] for f in files): raise ValueError('Missing/mismatched cached gradients')
    n=len(folders);cross=[[0.]*n for _ in range(n)];dot=[[0.]*n for _ in range(n)];energies=[0.]*n
    rows=[]
    for name in sorted(files[0]):
        items=[torch.load(f/'gradient_cache'/name,map_location='cpu',weights_only=True) for f in folders]
        e=[float(x['g'].double().square().sum()) for x in items]
        row=dict(module=name[:-3],cosine=[],cross_time_tangent_coverage=[],principal_angles_U=[],principal_angles_V=[],overlap_U=[],overlap_V=[])
        for j in range(n): energies[j]+=e[j]
        for i,x in enumerate(items):
            cos=[];cov=[];au=[];av=[];ou=[];ov=[]
            for j,y in enumerate(items):
                inner=float((x['g'].double()*y['g'].double()).sum());dot[i][j]+=inner
                rr=residual_gradient(y['g'],x['qa'],x['qb'])
                covered=float((y['g']-rr).double().square().sum());cross[i][j]+=covered
                cos.append(inner/math.sqrt(e[i]*e[j]) if e[i]*e[j]>0 else None)
                cov.append(covered/e[j] if e[j]>0 else None)
                for key,angles,overlaps in [('u',au,ou),('v',av,ov)]:
                    s=torch.linalg.svdvals(x[key].T@y[key]).clamp(0,1)
                    angles.append(torch.acos(s).tolist())
                    overlaps.append(float(s.square().mean()) if len(s) else None)
            row['cosine'].append(cos);row['cross_time_tangent_coverage'].append(cov)
            row['principal_angles_U'].append(au);row['principal_angles_V'].append(av)
            row['overlap_U'].append(ou);row['overlap_V'].append(ov)
        rows.append(row)
    save_rows(args.output_dir,rows)
    write(args.output_dir/'summary.json',dict(labels=args.inputs,rows='LoRA checkpoint',columns='gradient checkpoint',
        principal_angle_units='radians',overlap='mean squared principal cosine; cached top-d numerical singular directions',
        global_gradient_cosine=[[dot[i][j]/math.sqrt(energies[i]*energies[j]) if energies[i]*energies[j]>0 else None for j in range(n)] for i in range(n)],
        global_cross_time_tangent_coverage=[[cross[i][j]/energies[j] if energies[j]>0 else None for j in range(n)] for i in range(n)]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['initial','probe','gaps','compare','drift'])
    p.add_argument('--run_dir',type=Path)
    p.add_argument('--adapter',type=Path)
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--inputs',nargs='+')
    p.add_argument('--k',type=int,default=8)
    p.add_argument('--indices',nargs='+',type=int)
    p.add_argument('--budget',type=int,default=480256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--random_trials',type=int,default=0)
    p.add_argument('--oracle_modules',nargs='+')
    p.add_argument('--modules',nargs='+')
    p.add_argument('--tolerance',type=float,default=2e-5)
    p.add_argument('--save_gradients',action='store_true')
    p.add_argument('--drift_rank',type=int,default=8)
    args=p.parse_args()
    if args.k<1 or args.repeats<1 or args.random_trials<0 or args.drift_rank<1: p.error('Invalid counts')
    if args.mode in ('initial','probe'):
        if args.run_dir is None: p.error('--run_dir is required')
        if args.mode=='probe' and (args.adapter is None or not args.adapter.is_file()):
            p.error('Checkpoint missing: saved diagnostic JSON cannot reconstruct adapter weights')
    elif not args.inputs: p.error('--inputs is required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    write(args.output_dir/'diagnostic_config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    if args.mode in ('gaps','compare'): offline(args)
    elif args.mode=='drift': drift(args)
    else: model_probe(args)


if __name__=='__main__': main()
