"""Fixed-rank temporal diagnostics. No adapter correction or rank mutation."""
import argparse, csv, itertools, json, random, tempfile, time, traceback, sys, shutil
from pathlib import Path
import numpy as np
import torch
import global84_experiment as w
import global84_rank_math as m
STEPS=(0,100,200,300,400)

def snapshot(layers):
    return {n:{k:getattr(l,k).detach().cpu().clone() for k in ('weight_a','weight_b')} for n,l in layers.items()}

def overlap(x,y,k):
    if x.shape[1]<k or y.shape[1]<k: return None
    return float((x[:,:k].T@y[:,:k]).square().sum()/k)

def turnover(a,b):
    return dict(rank_turnover=sum(abs(a[n]-b[n]) for n in a)/(2*sum(a.values())),
        persistent=[n for n in a if a[n] and b[n]],entered=[n for n in a if not a[n] and b[n]],exited=[n for n in a if a[n] and not b[n]])

def spearman(x,y):
    def ranks(z):
        return np.array([sum(v<t for v in z)+(sum(v==t for v in z)-1)/2 for t in z])
    a,b=ranks(x),ranks(y)
    return None if a.std()==0 or b.std()==0 else float(np.corrcoef(a,b)[0,1])

def metrics(g,a,b,V):
    qa,qb=m.basis(a.T),m.basis(b)
    r=w.shared.residual_gradient(g.double(),qa,qb); t=g-r
    energy=float(g.double().square().sum()); re=float(r.square().sum())
    ct=float(t.square().sum())/(energy+1e-30); er=re/(energy+1e-30)
    if energy: w.require(abs(ct+er-1)<1e-6,'Tangent decomposition failed')
    rv=r@V.double(); sigma=rv.norm(dim=0)
    valid=sigma>max(float(sigma.max())*1e-7,1e-30)
    # Numerically absent singular directions have no identifiable subspace.
    u=rv[:,valid]/sigma[valid]; v=V.double()[:,valid]
    return dict(gradient_norm=energy**.5,CT=ct,ER=er,residual_norm=re**.5,
        zero_gradient=energy==0,singular_values=sigma.tolist(),
        M=(sigma.square().cumsum(0)/(energy+1e-30)).tolist(),
        spectrum_method='converged matrix-free Ritz approximation, not exact full spectrum'),u,v

def consistency(blocks,a,b):
    qa,qb=m.basis(a.T),m.basis(b)
    rs=[w.shared.residual_gradient(g,qa,qb).flatten() for g in blocks]
    return [None if float(x.norm()*y.norm())==0 else float(torch.dot(x,y)/(x.norm()*y.norm())) for x,y in itertools.combinations(rs,2)]

def csv_write(path,rows):
    if not rows: return
    with path.open('w') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader()
        for row in rows: out.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def stage(model,layers,plan,data,step,seed,out,previous):
    before=snapshot(layers); rows=plan['common']['inventory']['modules']
    w.require(shutil.disk_usage(out).free>14*2**30,'Need at least 14 GiB temporary disk space')
    torch.save(before,out/f'adapter_step_{step}.pt')
    records=[]; spaces={}; stability=[]; opts={'C':{},'DE':{}}
    # Only this invocation's temporary directory is removed, never historical data.
    with tempfile.TemporaryDirectory(prefix=f'probe_{step}_',dir=out) as tmp:
        folder=Path(tmp); artifacts={}
        for role in w.ROLES:
            for block in range(4):
                grads,profile=w.core.collect_weight_gradients(model,layers,[w.shared.batch(data[role][i],'cpu') for i in range(block*8,block*8+8)])
                w.require(profile['num_target_matrices']==84 and profile['backward_calls']==8,'Incomplete simultaneous probe')
                for i,row in enumerate(rows):
                    p=folder/f'gradients/{role}/{block}/{i:02d}.pt';p.parent.mkdir(parents=True,exist_ok=True)
                    torch.save(grads[row['module']],p);artifacts[str(p.relative_to(folder))]=w.shared.sha256(p)
                del grads
                print(f'step {step}: {role} block {block+1}/4',flush=True)
                w.write(out/f'probe_{step}_{role}_{block}.json',profile)
        payload=dict(layers={n:dict(state=s) for n,s in before.items()})
        for i,row in enumerate(rows):
            n=row['module']; shape=(row['d_out'],row['d_in'])
            frames,record=w.analyze_module(folder,artifacts,plan,payload,i,row,seed,'cpu')
            blocks={role:w.load_gradient_blocks(folder,artifacts,role,i,shape,'cpu') for role in w.ROLES}
            g=m.weighted_mean(blocks['construction'],w.block_tokens(plan,'construction'))
            a,b=before[n]['weight_a'],before[n]['weight_b']
            values,u,v=metrics(g,a,b,frames['residual_svd'])
            spaces[n]=(u,v)
            records.append(dict(step=step,module=n,rank=row['rank'],**values,
                block_consistency={r:consistency(bs,a,b) for r,bs in blocks.items()},analysis=record))
            for kind,key in [('C','C_prefix_options'),('DE','DE_prefix_options')]:opts[kind][n]=record[key]
            if n in previous:
                for k in (1,2,4): stability.append(dict(step=step,module=n,k=k,left=overlap(previous[n][0],u,k),right=overlap(previous[n][1],v,k)))
    w.require(all(w.shared.tensor_exact(getattr(layers[n],k).cpu(),p) for n,s in before.items() for k,p in s.items()),'Diagnostic changed adapter')
    allocations={kind:m.exact_stratified_allocation(o,rows)[0] for kind,o in opts.items()}
    return records,allocations,spaces,stability

def run(args):
    out=args.output_dir; out.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',mode=args.mode,seed=args.seed,scientific_conclusion='现有证据不足')
    try:
        plan,data=w.prepare(args)
        report['code_sha256']=dict(w.code_hashes(),**{__file__:w.shared.sha256(Path(__file__))})
        report['input_sha256']=plan['common']['input_sha256']
        report['plan_fingerprint']=plan['fingerprint']
        if args.mode=='dry-run': report.update(status='PASS',cpu_only=True);return
        report['gpu']=w.gpu_admission(args.min_free_gib)
        model,layers,base,report['loading']=w.load_full(plan)
        params=[getattr(l,k) for l in layers.values() for k in ('weight_a','weight_b')]
        for p in params:p.requires_grad_(True)
        w.require(sum(p.numel() for p in params)==480000,'Wrong fixed budget')
        optimizer=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01)
        torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        order=list(range(400));random.Random(m.stream(args.seed,'train_order')).shuffle(order)
        counts=[w.effective_tokens(x) for x in data['train']]; mean=sum(counts)/400
        allrows=[]; allocations={}; previous={}; stability=[]; changes=[]
        for step in range(401):
            if step:
                model.eval();optimizer.zero_grad(set_to_none=True);i=order[step-1]
                loss=model(**w.shared.batch(data['train'][i],'cuda'),use_cache=False).loss
                objective=loss*counts[i]/mean;w.require(torch.isfinite(objective),'Invalid loss');objective.backward()
                w.require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),'Invalid gradients')
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,index=i,nll=float(loss.detach()),effective_tokens=counts[i],objective=float(objective.detach()),lr=5e-5))+'\n')
            if step in STEPS:
                rr,alloc,previous,ss=stage(model,layers,plan,data,step,args.seed,out,previous)
                if allocations:
                    for kind in alloc:
                        prev=allocations[str(step-100)][kind]
                        old=[r['analysis'][kind+'_prefix_options' if kind=='C' else 'DE_prefix_options'][1]['value'] for r in allrows[-84:]]
                        new=[r['analysis'][kind+'_prefix_options' if kind=='C' else 'DE_prefix_options'][1]['value'] for r in rr]
                        changes.append(dict(step=step,kind=kind,spearman=spearman(old,new),**turnover(prev,alloc[kind])))
                allocations[str(step)]=alloc;allrows+=rr;stability+=ss
                csv_write(out/'temporal_metrics.csv',allrows);csv_write(out/'rank_turnover.csv',changes);csv_write(out/'subspace_stability.csv',stability)
                w.write(out/'allocation_by_stage.json',allocations)
        report['base_verified']=w.shared.verify_disk_weights(base,args.model)
        report.update(status='PASS',fresh_AdamW=True,historical_momentum_restored=False,
            snapshot_sha256={str(k):w.shared.sha256(out/f'adapter_step_{k}.pt') for k in STEPS})
        w.write(out/'results.json',dict(interpretation='现有证据不足',reason='Capacity proxies only; inspect residual norms, M, consistency, independent audit and paired seeds together',allocations=allocations))
    except BaseException as e:report.update(status='FAIL',error=str(e),traceback=traceback.format_exc());raise
    finally:
        if 'input_sha256' in report:w.check_hashes(report['input_sha256'])
        w.write(out/'report.json',report)

def summarize(argv):
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs=3,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv); data={}; fingerprint=None; codes=None
    for folder in args.runs:
        report=json.loads((folder/'report.json').read_text())
        w.require(report['status']=='PASS' and report['mode']=='train','Completed trajectory required')
        seed=report['seed'];w.require(seed in (42,43,44) and seed not in data,'Expected unique seeds 42/43/44')
        if fingerprint is None:fingerprint=report['plan_fingerprint'];codes=report['code_sha256']
        w.require(fingerprint==report['plan_fingerprint'] and codes==report['code_sha256'],'Different protocol/data/code')
        for step,h in report['snapshot_sha256'].items():w.require(w.shared.sha256(folder/f'adapter_step_{step}.pt')==h,'Snapshot changed')
        data[seed]=json.loads((folder/'allocation_by_stage.json').read_text())
    rows=[]
    for step in STEPS:
        for kind in ('C','DE'):
            for a,b in itertools.combinations(sorted(data),2):rows.append(dict(step=step,kind=kind,seed_a=a,seed_b=b,**turnover(data[a][str(step)][kind],data[b][str(step)][kind])))
    args.output_dir.mkdir(parents=True,exist_ok=False)
    csv_write(args.output_dir/'cross_seed_stability.csv',rows)
    w.write(args.output_dir/'results.json',dict(status='PASS',cross_seed=rows,conclusion='现有证据不足',
        interpretation_rules={
            '秩需求确实随时间变化':'Requires reproducible allocation and normalized marginal-potential changes, consistent independent audit; not CT alone',
            '只有梯度强度变化，秩分配没有变化':'Stable allocation and normalized potential despite norm shifts',
            '变化主要来自样本噪声':'Requires block instability and lack of reproducibility; cannot infer from one seed',
            '现有证据不足':'Default: proxies cannot establish causal validation-loss benefit'},
        warning='Three paired seed trajectories, NOT 15 independent experiments; manual joint review required'))


def main():
    if len(sys.argv)>1 and sys.argv[1]=='summarize':return summarize(sys.argv[2:])
    p=w.parser();p.description=__doc__
    args=p.parse_args();w.require(args.mode in ('dry-run','train'),'Only dry-run/train supported');run(args)
if __name__=='__main__':main()
