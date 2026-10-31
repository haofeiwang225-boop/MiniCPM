"""Independent base-to-2000 fixed-rank study. No historical adapter loading."""
import ast, argparse, copy, hashlib, json, random, sys, tempfile, traceback, csv, itertools, types
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator, set_seed
import global84_temporal_rank_demand as t
import global84_experiment as w
import global84_rank_math as m
from adaptive_lora.residual_gid import svd_with_gid
from rank_gain_probe_support import load_original_lora_class
STAGES=(0,100,400,800,1200,1600,2000)
ANALYSIS_SEED=20261031
PROBE_SAMPLING_SEED=20261031
SCOPE='pretrained base -> GID initialization -> continuous 2000-step fresh AdamW trajectory'
HISTORY=w.ROOT/'finetune/adaptive_lora/outputs/metamath_pilot_control_2000_seed42/run_config.json'

def collect_equal_gradients(model,layers,batches):
    """Singleton reuse: exactly one backward/sample, equal sample means, no layer loop."""
    w.require(bool(batches),'Empty equal-example probe')
    total=None; profiles=[]
    for batch in batches:
        gs,profile=w.core.collect_weight_gradients(model,layers,[batch])
        if total is None:total={n:torch.zeros_like(g) for n,g in gs.items()}
        for n,g in gs.items():total[n].add_(g,alpha=1/len(batches))
        profiles.append(profile)
    return total,dict(num_target_matrices=len(layers),forward_calls=len(batches),backward_calls=len(batches),
        weighting='equal per-example masked mean loss',sample_weights=[1/len(batches)]*len(batches),
        effective_tokens=sum(p['effective_tokens'] for p in profiles),singleton_profiles=profiles)


def bound_function(fn,**dependencies):
    # Private function globals: NEVER monkeypatch old experiment modules.
    return types.FunctionType(fn.__code__,dict(fn.__globals__,**dependencies),fn.__name__,fn.__defaults__,fn.__closure__)


def equal_stage(*args):
    equal_counts=lambda plan,role:[8]*4
    analysis=bound_function(w.analyze_module,block_tokens=equal_counts)
    local_core=types.SimpleNamespace(**vars(w.core));local_core.collect_weight_gradients=collect_equal_gradients
    local_w=types.SimpleNamespace(**vars(w));local_w.core=local_core;local_w.block_tokens=equal_counts;local_w.analyze_module=analysis
    w.require(len(args)==8,'Expected complete stage arguments')
    fixed=list(args);fixed[5]=ANALYSIS_SEED
    return bound_function(t.stage,w=local_w)(*fixed)


def check_history(hist):
    expected=dict(max_steps=2000,batch_size=1,learning_rate=5e-5,weight_decay=.01,lora_dropout=.1,dense_lora_param_budget=480256,
        model_family='minicpm5',lora_structure='dense',lora_rank=2,lora_alpha=4.,dense_rank_min=1,dense_rank_max=4,
        weight_a_init_method='kaiming',dynamic_length=True,model_max_length=2048,gradient_checkpointing=True,
        adaptive_lora=False,direction_correction='none')
    for k,v in expected.items():w.require(k in hist and hist[k]==v,f'Historical protocol mismatch: {k}; expected {v!r}, got {hist.get(k)!r}')


def validate_initial(layers,ranks,total):
    names={f'model.layers.{i}.self_attn.{kind}' for i in range(42) for kind in ('q_proj','v_proj')}
    w.require(set(layers)==set(ranks)==names,'Incomplete initialized targets')
    actual=0
    for n,l in layers.items():
        r=ranks[n];w.require(type(r)==int and 1<=r<=4 and l.lora_rank==r,f'Invalid rank: {n}')
        w.require(l.weight_a.shape==(r,l.in_features) and l.weight_b.shape==(l.out_features,r),f'Invalid factors: {n}')
        w.require(torch.isfinite(l.weight_a).all() and torch.count_nonzero(l.weight_b)==0,f'Invalid zero-B initialization: {n}')
        actual+=l.weight_a.numel()+l.weight_b.numel()
    w.require(actual==total and actual<=480256,'Allocator/actual budget mismatch')
    return actual


def training_smoke(model,layers,data):
    initial=t.snapshot(layers);results=[]
    for probe in (False,True):
        with torch.no_grad():
            for n,s in initial.items():
                for k,v in s.items():getattr(layers[n],k).copy_(v)
        set_seed(42);params=[getattr(l,k) for l in layers.values() for k in ('weight_a','weight_b')]
        optimizer=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1-step/2000)
        losses=[];activity={n:{k:dict(max_gradient_norm=0.,nonzero_gradient_steps=0) for k in s} for n,s in initial.items()}
        for step in range(2):
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=model(**w.shared.batch(data['train'][step],'cuda')).loss
            w.require(torch.isfinite(loss),'Smoke loss nonfinite');loss.backward();t.activity_update(layers,activity)
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);losses.append(float(loss.detach()))
            if probe and step==0:
                _,profile=collect_equal_gradients(model,layers,[w.shared.batch(data['construction'][i],'cpu') for i in range(2)])
        for p in params:
            state=optimizer.state[p];w.require(int(state['step'])==2,'Adam step mismatch')
            w.require(torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all(),'Invalid Adam moments')
        results.append(dict(factors=t.snapshot(layers),losses=losses,states={i:{k:v.detach().cpu().clone() for k,v in optimizer.state[p].items()} for i,p in enumerate(params)},activity=activity))
    a,b=results
    w.require(a['losses']==b['losses'],'Probe changed subsequent loss')
    for n,s in a['factors'].items():
        for k,v in s.items():w.require(torch.equal(v,b['factors'][n][k]),'Probe changed subsequent parameters')
    for i,state in a['states'].items():
        for k,v in state.items():w.require(torch.equal(v,b['states'][i][k]),'Probe changed subsequent Adam state')
    w.require(all(x['max_gradient_norm']>0 for s in b['activity'].values() for x in s.values()),'Inactive smoke factor')
    return dict(losses=a['losses'],probe=profile,continuation_bitwise_equal=True,optimizer_steps=2,activity=b['activity'],
        dropout=.1,use_reentrant=False,input_require_grads_hook=False,peak_allocated_bytes=torch.cuda.max_memory_allocated())

def allocator(records):
    # Execute the unchanged pure allocation function, avoiding unrelated official
    # package eager imports (DeepSpeed). Source hash is part of the manifest.
    path=w.ROOT/'finetune/adaptive_lora/dense_init.py'
    tree=ast.parse(path.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='allocate_dense_ranks')
    import math
    ns={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns['allocate_dense_ranks'](records,480256,1,4)

def sample_probe_pool(source,banned,per_family=64):
    """Full-source group sampling; canonical representative independent of line order."""
    source=Path(source);before=w.shared.sha256(source);groups={};lines=0
    with source.open() as f:
        for lines,line in enumerate(f,1):
            r=json.loads(line);family=r.get('type','').split('_')[0];original=r.get('original_question')
            if family not in ('GSM','MATH') or not isinstance(original,str):continue
            h=hashlib.sha256(original.encode()).hexdigest()
            if h in banned:continue
            w.require(isinstance(r.get('query'),str) and isinstance(r.get('response'),str),'Invalid candidate text')
            sample=hashlib.sha256(json.dumps([r['query'],r['response']],ensure_ascii=False).encode()).hexdigest()
            old=groups.get(h)
            if old:w.require(old['family']==family,'Conflicting family for original question')
            if old is None or sample<old['sample_sha256']:
                groups[h]=dict(original_question_sha256=h,family=family,source_line_1based=lines,sample_sha256=sample)
    rng=random.Random(PROBE_SAMPLING_SEED);selected=[];sizes={}
    for family in ('GSM','MATH'):
        pool=sorted(h for h,r in groups.items() if r['family']==family);sizes[family]=len(pool)
        w.require(len(pool)>=per_family,f'Insufficient {family} independent candidate groups')
        selected.extend(groups[h] for h in rng.sample(pool,per_family))
    wanted={r['source_line_1based'] for r in selected};rows={}
    with source.open() as f:
        for i,line in enumerate(f,1):
            if i in wanted:
                r=json.loads(line);rows[i]={'messages':[{'role':'user','content':r['query']},{'role':'assistant','content':r['response']}]}
    w.require(w.shared.sha256(source)==before,'Source changed during full-pool sampling')
    return [rows[r['source_line_1based']] for r in selected],selected,dict(
        seed=PROBE_SAMPLING_SEED,source_sha256=before,scanned_lines=lines,candidate_group_counts=sizes,
        method='full scan, group hash sorting, uniform sample without replacement within family',
        representative='minimum query/response SHA256 per original-question group; no token-length filtering')

def prepare(args):
    out=args.output_dir
    hist=json.loads(HISTORY.read_text())
    check_history(hist)
    tm=w.shared.build_source_metadata(args.train_data,args.source_data);dm=w.shared.build_source_metadata(args.dev_data,args.source_data)
    w.require(len(tm['samples'])==2000 and len(dm['samples'])==400,'Expected historical train2000/dev400')
    banned={x['original_question_sha256'] for meta in (tm,dm) for x in meta['samples']}
    w.require(not ({x['original_question_sha256'] for x in tm['samples']} & {x['original_question_sha256'] for x in dm['samples']}),'Train/dev overlap')
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=True)
    if tok.chat_template is None:tok.chat_template=(args.model/'chat_template.jinja').read_text()
    tok.pad_token=tok.eos_token
    chosen,metadata,sampling=sample_probe_pool(args.source_data,banned)
    indices=list(range(128));random.Random(PROBE_SAMPLING_SEED).shuffle(indices)
    data={'train':w.shared.ChatTrainingDataset(args.train_data,tok,2048,dynamic_length=True)}
    meta={'train':tm['samples'],'dev':dm['samples']};tokens={}
    for j,role in enumerate(w.ROLES):
        ids=indices[32*j:32*(j+1)];p=out/f'{role}.json';w.write(p,[chosen[i] for i in ids])
        data[role]=w.shared.ChatTrainingDataset(p,tok,2048,dynamic_length=True)
        meta[role]=[metadata[i] for i in ids]
        tokens[role]=[w.effective_tokens(x) for x in data[role]]
    sampling['role_effective_tokens']={r:dict(values=tokens[r],min=min(tokens[r]),max=max(tokens[r]),mean=sum(tokens[r])/len(tokens[r])) for r in w.ROLES}
    sampling['role_family_counts']={r:{f:sum(x['family']==f for x in meta[r]) for f in ('GSM','MATH')} for r in w.ROLES}
    files=[args.train_data,args.dev_data,args.source_data,HISTORY]+list(args.model.glob('*.json'))+list(args.model.glob('*.safetensors'))+list(args.model.glob('*.jinja'))
    code=dict(w.code_hashes());code.update({str(p):w.shared.sha256(p) for p in [Path(__file__),Path(t.__file__),w.ROOT/'finetune/adaptive_lora/dense_init.py',w.ROOT/'finetune/adaptive_lora/residual_gid.py']})
    plan=dict(seed_roles=dict(training=dict(allowed=[42,43,44],responsibility='training order and dropout only; actual seed is recorded in report.json'),analysis=dict(value=ANALYSIS_SEED,responsibility='all stage Krylov starts, restarts and candidate random frames'),probe_sampling=dict(value=PROBE_SAMPLING_SEED,responsibility='full-pool group sampling and role ordering'),initialization=dict(value=42,responsibility='one shared GID initial adapter')),probe_sampling=sampling,trajectory_scope=SCOPE,metadata=meta,tokens=tokens,input_sha256={str(p.resolve()):w.shared.sha256(p) for p in files},code_sha256=code,
        protocol=dict(loss='historical per-example masked mean NLL (NOT Global84 token-weighted training)',dropout=.1,gradient_checkpointing=True,use_reentrant=False,probe_weighting='equal samples within and across blocks; tokens retained only for audit',lr=5e-5,scheduler='linear 1-step/2000',initial_probe='first eight historical training examples; separate from temporal probe128',stages=list(STAGES)),historical_config=hist)
    w.write(out/'plan.json',plan);return plan,data

def base(args):
    model,info=w.shared.load_probe_model(args.model,torch.bfloat16);model.to('cuda');model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model,dict(model.named_parameters()),info

def attach(model,ranks):
    cls,cfg=load_original_lora_class(w.ROOT);layers={}
    targets=[(n,l) for n,l in model.named_modules() if isinstance(l,torch.nn.Linear) and n.endswith(('.q_proj','.v_proj'))]
    w.require(len(targets)==84,'Expected 84 modules')
    for n,old in targets:
        conf=cfg(in_features=old.in_features,out_features=old.out_features,lora_rank=ranks[n] if ranks else 2,lora_scaler=4,lora_dropout=.1,weight_a_init_method='kaiming',run_lora_in_fp32=True)
        l=cls(conf);l.init_lora_weights();l.to(old.weight.device);l.weight=old.weight;l.bias=old.bias
        parent,_,child=n.rpartition('.');setattr(model.get_submodule(parent),child,l);layers[n]=l
    for p in model.parameters():p.requires_grad_(False)
    for l in layers.values():l.weight_a.requires_grad_(True);l.weight_b.requires_grad_(True)
    return layers

def init(model,data,out):
    layers=attach(model,None)
    avg={n:torch.zeros_like(l.weight,device='cpu',dtype=torch.float32) for n,l in layers.items()}
    for i in range(8):
        gs,_=w.core.collect_weight_gradients(model,layers,[w.shared.batch(data['train'][i],'cpu')])
        for n in avg:avg[n].add_(gs[n],alpha=1/8)
    records=[]
    for n,g in avg.items():
        _,_,_,gid=svd_with_gid(g);records.append(dict(module=n,shape=list(g.shape),effective_rank=gid))
    ranks,total=allocator(records)
    # Replace only during initialization, before optimizer creation.
    cls,cfg=load_original_lora_class(w.ROOT)
    for n,l in list(layers.items()):
        new=cls(cfg(in_features=l.in_features,out_features=l.out_features,lora_rank=ranks[n],lora_scaler=4,lora_dropout=.1,weight_a_init_method='kaiming',run_lora_in_fp32=True));new.init_lora_weights();new.to(l.weight.device);new.weight=l.weight;new.bias=l.bias
        parent,_,child=n.rpartition('.');setattr(model.get_submodule(parent),child,new);layers[n]=new
    for r in records:
        l=layers[r['module']];r.update(rank=l.lora_rank,A_shape=list(l.weight_a.shape),B_shape=list(l.weight_b.shape),scaling=l.lora_scaler,A_numerical_rank=m.basis(l.weight_a.T).shape[1],B_numerical_rank=m.basis(l.weight_b).shape[1])
    validate_initial(layers,ranks,total)
    payload=dict(ranks=ranks,factors=t.snapshot(layers),records=records,actual_parameters=total)
    torch.save(payload,out/'initial_adapter.pt');w.write(out/'initial_gid.json',dict(records=records,actual_parameters=total,nominal_budget=480256,sha256=w.shared.sha256(out/'initial_adapter.pt')))
    loaded=torch.load(out/'initial_adapter.pt',map_location='cpu',weights_only=True)
    w.require(all(torch.equal(v,loaded['factors'][n][k]) for n,s in payload['factors'].items() for k,v in s.items()),'Initial adapter serialization mismatch')
    w.require(w.shared.sha256(out/'initial_adapter.pt')==json.loads((out/'initial_gid.json').read_text())['sha256'],'Initial adapter SHA mismatch')
    return payload

def allocation_audit(records,inventory,initial):
    result={}
    for kind,family in [('C','residual_svd'),('DE','cross_alignment')]:
        options={r['module']:r['analysis'][kind+'_prefix_options'] for r in records}
        best,_=m.exact_stratified_allocation(options,inventory)
        value=lambda a:sum(options[n][k]['value'] for n,k in a.items())
        alternatives=[]
        for n,k in best.items():
            changed=copy.deepcopy(options);changed[n][k]['value']=-1e300
            candidate,_=m.exact_stratified_allocation(changed,inventory)
            if candidate[n]!=k:alternatives.append((value(candidate),candidate))
        second=max(alternatives,key=lambda x:x[0]) if alternatives else None
        def evaluate(a):
            return dict(selection_proxy=value(a),audit_full_gradient_U=sum(sum(r['analysis']['candidates'][family]['audit_U'][:a[r['module']]]) for r in records))
        result[kind]=dict(best=evaluate(best),step0=evaluate(initial[kind]),
            runner_up=None if second is None else second[0],gap=None if second is None else value(best)-second[0],
            note='Audit full-gradient U is independent diagnostic only, not a training gain')
    return result

def run(args):
    args.output_dir.mkdir(parents=True,exist_ok=False);report=dict(status='RUNNING',trajectory_scope=SCOPE,mode=args.mode,seed=args.seed,analysis_seed=ANALYSIS_SEED,probe_sampling_seed=PROBE_SAMPLING_SEED)
    try:
        plan,data=prepare(args)
        if args.mode=='dry-run':report['status']='PASS';return
        report['gpu']=w.gpu_admission(30);set_seed(42);model,weights,report['loading']=base(args)
        if args.mode=='smoke':
            layers=attach(model,None);torch.cuda.reset_peak_memory_stats()
            report['training_smoke']=training_smoke(model,layers,data)
        elif args.mode=='initialize':init(model,data,args.output_dir)
        else:
            w.require(args.initial_adapter is not None,'Supply shared --initial-adapter')
            origin=json.loads((args.initial_adapter.parent/'plan.json').read_text())
            w.require(origin==plan,'Initialization data/code/config mismatch')
            initial_hash=w.shared.sha256(args.initial_adapter)
            w.require(json.loads((args.initial_adapter.parent/'initial_gid.json').read_text())['sha256']==initial_hash,'Initial adapter hash mismatch')
            report['initial_adapter_sha256']=initial_hash
            payload=torch.load(args.initial_adapter,map_location='cpu',weights_only=True)
            layers=attach(model,payload['ranks'])
            with torch.no_grad():
                for n,s in payload['factors'].items():
                    for k,v in s.items():getattr(layers[n],k).copy_(v)
            validate_initial(layers,payload['ranks'],payload['actual_parameters'])
            set_seed(args.seed);params=[getattr(l,k) for l in layers.values() for k in ('weight_a','weight_b')]
            optimizer=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:max(0,1-step/2000))
            loader=iter(DataLoader(data['train'],batch_size=1,shuffle=True,generator=torch.Generator().manual_seed(args.seed),collate_fn=default_data_collator))
            inventory=[dict(module=n,rank=l.lora_rank,d_in=l.in_features,d_out=l.out_features,cost_per_added_rank=l.in_features+l.out_features) for n,l in layers.items()]
            temporal_plan={'common':dict(tokens=plan['tokens'],inventory=dict(modules=inventory))};previous={};metrics=[];allocations={};stability=[]
            initial=t.snapshot(layers);activity={n:{k:dict(max_gradient_norm=0.,nonzero_gradient_steps=0) for k in s} for n,s in initial.items()}
            for step in range(2001):
                if step:
                    model.train();batch={k:v.to('cuda') for k,v in next(loader).items()};loss=model(**batch).loss;loss.backward();t.activity_update(layers,activity)
                    # Exact one-step decay-only counterfactual at current parameters;
                    # actual minus this counterfactual includes Adam's gradient term.
                    pre=[p.detach().clone() for p in params];lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
                    beyond_decay=sum(float((p.detach().double()-q.clone().mul_(1-lr*.01).double()).square().sum()) for p,q in zip(params,pre))**.5
                    with (args.output_dir/'train.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=float(loss.detach()),lr_used=lr,lr_next=scheduler.get_last_lr()[0],update_beyond_decay_norm=beyond_decay))+'\n')
                if step in STAGES:
                    rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone();state_versions={id(p):{k:v._version for k,v in state.items() if torch.is_tensor(v)} for p,state in optimizer.state.items()}
                    rr,aa,previous,ss=equal_stage(model,layers,temporal_plan,data,step,args.seed,args.output_dir,previous)
                    w.require(torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state()),'Stage changed RNG')
                    w.require(state_versions=={id(p):{k:v._version for k,v in state.items() if torch.is_tensor(v)} for p,state in optimizer.state.items()},'Stage changed optimizer state')
                    w.require(all(l.lora_rank==payload['ranks'][n] for n,l in layers.items()),'Rank changed')
                    for row in rr:
                        layer=layers[row['module']];row['A_numerical_rank']=m.basis(layer.weight_a.T).shape[1];row['B_numerical_rank']=m.basis(layer.weight_b).shape[1]
                    metrics+=rr;stability+=ss;allocations[str(step)]=aa
                    w.write(args.output_dir/f'allocation_audit_{step}.json',allocation_audit(rr,inventory,allocations['0']))
                    t.csv_write(args.output_dir/'temporal_metrics.csv',metrics);t.csv_write(args.output_dir/'subspace_stability.csv',stability);w.write(args.output_dir/'allocation_by_stage.json',allocations)
                    w.write(args.output_dir/f'state_{step}.json',dict(step=step,lr=scheduler.get_last_lr(),scheduler_epoch=scheduler.last_epoch,adam_state_count=len(optimizer.state),sha256=w.shared.sha256(args.output_dir/f'adapter_step_{step}.pt')))
            turnovers=[]
            for old,new in zip(STAGES,STAGES[1:]):
                for kind in ('C','DE'):
                    prev=[r for r in metrics if r['step']==old];now=[r for r in metrics if r['step']==new]
                    turnovers.append(dict(old=old,new=new,kind=kind,spearman=t.spearman([r['analysis'][kind+'_prefix_options'][1]['value'] for r in prev],[r['analysis'][kind+'_prefix_options'][1]['value'] for r in now]),**t.turnover(allocations[str(old)][kind],allocations[str(new)][kind])))
            t.csv_write(args.output_dir/'rank_turnover.csv',turnovers)
            w.write(args.output_dir/'results.json',dict(trajectory_scope=SCOPE,conclusion='现有证据不足',allocations=allocations,turnover=turnovers))
            w.write(args.output_dir/'training_activity.json',t.activity_finish(layers,initial,activity));w.require(w.shared.sha256(args.initial_adapter)==initial_hash,'Initial adapter changed')
        report['base_verified']=w.shared.verify_disk_weights(weights,args.model);w.check_hashes(plan['input_sha256']);report['status']='PASS'
    except BaseException as e:report.update(status='FAIL',error=str(e),traceback=traceback.format_exc());raise
    finally:w.write(args.output_dir/'report.json',report)

def summarize(argv):
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs=3,type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(argv)
    metrics={};alloc={};reference=None
    for folder in a.runs:
        report=json.loads((folder/'report.json').read_text());plan=json.loads((folder/'plan.json').read_text())
        w.require(report['status']=='PASS' and report['mode']=='train','Completed training required')
        seed=report['seed'];w.require(seed not in metrics and seed in (42,43,44),'Wrong seeds')
        if reference is None:reference=plan
        w.require(plan==reference,'Different data/code/protocol')
        with (folder/'temporal_metrics.csv').open() as f:rows=list(csv.DictReader(f))
        metrics[seed]={}
        for r in rows:
            for key in ('M_full','Q_residual','block_consistency','analysis'):r[key]=json.loads(r[key])
            key=(int(r['step']),r['module']);w.require(key not in metrics[seed],'Duplicate row');metrics[seed][key]=r
        w.require(len(metrics[seed])==588 and {k[0] for k in metrics[seed]}==set(STAGES),'Incomplete stages')
        alloc[seed]=json.loads((folder/'allocation_by_stage.json').read_text())
    a.output_dir.mkdir(parents=True,exist_ok=False)
    pairs=[dict(step=step,kind=kind,seeds=[x,y],**t.turnover(alloc[x][str(step)][kind],alloc[y][str(step)][kind])) for step in STAGES for kind in ('C','DE') for x,y in itertools.combinations(sorted(alloc),2)]
    t.csv_write(a.output_dir/'cross_seed_stability.csv',pairs)
    w.write(a.output_dir/'results.json',dict(trajectory_scope=SCOPE,metrics=t.aggregate_metrics(metrics),paired_seed_stability=pairs,conclusion='现有证据不足',note='Three paired trajectories; seven time points are not independent seeds; proxies do not prove NLL benefit'))

def main():
    if len(sys.argv)>1 and sys.argv[1]=='summarize':return summarize(sys.argv[2:])
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['dry-run','smoke','initialize','train']);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--initial-adapter',type=Path);p.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    p.add_argument('--model',type=Path,default=w.core.DEFAULT_MODEL);p.add_argument('--train-data',type=Path,default=w.shared.DATA/'train.json');p.add_argument('--dev-data',type=Path,default=w.shared.DATA/'dev.json');p.add_argument('--source-data',type=Path,default=Path('/home/hongchang/whf/datasets/MetaMathQA/MetaMathQA-395K.jsonl'));run(p.parse_args())
if __name__=='__main__':main()
