"""Fixed-rank temporal diagnostics. No adapter correction or rank mutation."""
import argparse, csv, itertools, json, random, tempfile, time, traceback, sys, shutil
from pathlib import Path
import numpy as np
import torch
import global84_experiment as w
import global84_rank_math as m
STEPS=(0,100,200,300,400)
SCOPE="step1600_pre + 400 fresh-AdamW updates; not the historical step0-2000 optimizer trajectory"

def snapshot(layers):
    return {n:{k:getattr(l,k).detach().cpu().clone() for k in ('weight_a','weight_b')} for n,l in layers.items()}

def overlap(x,y,k):
    if x.shape[1]<k or y.shape[1]<k: return None
    x,y=m.basis(x[:,:k]),m.basis(y[:,:k])
    if x.shape[1]<k or y.shape[1]<k: return None
    return float(((x.T@y).square().sum()/k).clamp(0,1))

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
    mf=sigma.square().cumsum(0)/(energy+1e-30)
    qr=sigma.square().cumsum(0)/(re+1e-30)
    if energy: w.require(torch.allclose(mf,er*qr,atol=1e-10,rtol=1e-6),'M_full != ER * Q_residual')
    return dict(M_full=mf.tolist(),Q_residual=qr.tolist(),gradient_norm=energy**.5,CT=ct,ER=er,residual_norm=re**.5,
        zero_gradient=energy==0,singular_values=sigma.tolist(),
        M=(sigma.square().cumsum(0)/(energy+1e-30)).tolist(),
        spectrum_method='converged matrix-free Ritz approximation, not exact full spectrum'),u,v

def consistency(blocks,a,b):
    qa,qb=m.basis(a.T),m.basis(b)
    rs=[w.shared.residual_gradient(g,qa,qb).flatten() for g in blocks]
    return [None if float(x.norm()*y.norm())==0 else float(torch.dot(x,y)/(x.norm()*y.norm())) for x,y in itertools.combinations(rs,2)]

def normalized_scores(g,selection,counts,a,b,frames):
    """Diagnostic only: block U divided by construction and selection Fro norms."""
    qa,qb=m.basis(a.T),m.basis(b); result={}
    for kind,family in [('C','residual_svd'),('DE','cross_alignment')]:
        ref=w.shared.residual_gradient(g,qa,qb) if kind=='C' else g
        blocks=[w.shared.residual_gradient(x,qa,qb) for x in selection] if kind=='C' else selection
        v=frames[family].double(); values=[]
        for x in blocks:
            denominator=float(ref.norm()*x.norm())
            values.append(None if denominator==0 else (((ref@v)*(x@v)).sum(0)/denominator).tolist())
        mean=None if any(x is None for x in values) else np.average(values,axis=0,weights=counts).tolist()
        result[kind]=dict(per_block=values,per_direction=mean,prefix=None if mean is None else np.cumsum(mean).tolist())
    return result


def activity_update(layers,stats):
    for n,l in layers.items():
        for k in ('weight_a','weight_b'):
            p=getattr(l,k);w.require(p.grad is not None and torch.isfinite(p.grad).all(),'Invalid factor gradient')
            v=float(p.grad.detach().double().norm());r=stats[n][k]
            r['max_gradient_norm']=max(r['max_gradient_norm'],v);r['nonzero_gradient_steps']+=int(v>0)


def activity_finish(layers,initial,stats):
    for n,l in layers.items():
        for k in ('weight_a','weight_b'):
            p=getattr(l,k).detach().cpu().double();r=stats[n][k]
            r.update(initial_norm=float(initial[n][k].double().norm()),final_norm=float(p.norm()),change_norm=float((p-initial[n][k].double()).norm()))
            r['effective_update_observed']=r['nonzero_gradient_steps']>0 and r['change_norm']>0
    return stats


def aggregate_metrics(by_seed):
    w.require(set(by_seed)=={42,43,44},'Three paired seeds required')
    keys=set(by_seed[42]);w.require(all(set(x)==keys for x in by_seed.values()),'Metric grids differ')
    def stats(values):
        valid=[float(x) for x in values if x is not None]
        w.require(all(np.isfinite(x) for x in valid),'Nonfinite summary value')
        return dict(n=len(valid),mean=None if not valid else float(np.mean(valid)),std=None if len(valid)<2 else float(np.std(valid,ddof=1)),range=None if not valid else [min(valid),max(valid)])
    result=[]
    for step,module in sorted(keys):
        rr=[by_seed[s][step,module] for s in (42,43,44)]
        row=dict(step=step,module=module)
        for field in ('CT','ER'): row[field]=stats([float(r[field]) for r in rr])
        for field in ('M_full','Q_residual'):row[field]=[stats([r[field][i] for r in rr]) for i in range(4)]
        row['block_consistency']={role:[stats([r['block_consistency'][role][i] for r in rr]) for i in range(6)] for role in w.ROLES}
        row['audit_U']={family:[stats([r['analysis']['candidates'][family]['audit_U'][i] for r in rr]) for i in range(4)] for family in m.FAMILIES}
        result.append(row)
    return result

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
            cross=frames['cross_alignment'].double()
            eig=record['solver']['cross_alignment']['eigenvalues']
            # Zero eigenvalues do not identify meaningful alignment directions.
            valid=[abs(x)>max(max(abs(z) for z in eig)*1e-7,1e-30) for x in eig]
            cross=cross[:,valid]
            spaces[n]=(u,v,cross)
            normalized=normalized_scores(g,blocks['selection'],w.block_tokens(plan,'selection'),a,b,frames)
            records.append(dict(step=step,module=n,rank=row['rank'],normalized_scores=normalized,**values,
                block_consistency={r:consistency(bs,a,b) for r,bs in blocks.items()},analysis=record))
            for kind,key in [('C','C_prefix_options'),('DE','DE_prefix_options')]:opts[kind][n]=record[key]
            if n in previous:
                for k in (1,2,4): stability.append(dict(step=step,module=n,k=k,
                    residual_svd_left=overlap(previous[n][0],u,k),residual_svd_right=overlap(previous[n][1],v,k),
                    cross_alignment_right=overlap(previous[n][2],cross,k)))
    w.require(all(w.shared.tensor_exact(getattr(layers[n],k).cpu(),p) for n,s in before.items() for k,p in s.items()),'Diagnostic changed adapter')
    allocations={kind:m.exact_stratified_allocation(o,rows)[0] for kind,o in opts.items()}
    return records,allocations,spaces,stability

def run(args):
    out=args.output_dir; out.mkdir(parents=True,exist_ok=False)
    report=dict(trajectory_scope=SCOPE,status='RUNNING',mode=args.mode,seed=args.seed,scientific_conclusion='现有证据不足')
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
        initial=snapshot(layers)
        activity={n:{k:dict(max_gradient_norm=0.,nonzero_gradient_steps=0) for k in ('weight_a','weight_b')} for n in layers}
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
                activity_update(layers,activity)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,index=i,nll=float(loss.detach()),effective_tokens=counts[i],objective=float(objective.detach()),lr=5e-5))+'\n')
            if step in STEPS:
                rr,alloc,previous,ss=stage(model,layers,plan,data,step,args.seed,out,previous)
                if allocations:
                    for kind in alloc:
                        prev=allocations[str(step-100)][kind]
                        old=[r['analysis'][kind+'_prefix_options' if kind=='C' else 'DE_prefix_options'][1]['value'] for r in allrows[-84:]]
                        new=[r['analysis'][kind+'_prefix_options' if kind=='C' else 'DE_prefix_options'][1]['value'] for r in rr]
                        norm_old=[r['normalized_scores'][kind]['prefix'] for r in allrows[-84:]]
                        norm_new=[r['normalized_scores'][kind]['prefix'] for r in rr]
                        normalized_corr=None if any(x is None for x in norm_old+norm_new) else [spearman([x[k] for x in norm_old],[x[k] for x in norm_new]) for k in range(4)]
                        raw_normalized=None if any(x is None for x in norm_new) else [spearman([r['analysis'][kind+'_prefix_options'][k+1]['value'] for r in rr],[x[k] for x in norm_new]) for k in range(4)]
                        changes.append(dict(step=step,kind=kind,spearman_normalized=normalized_corr,spearman_raw_vs_normalized=raw_normalized,spearman=spearman(old,new),**turnover(prev,alloc[kind])))
                allocations[str(step)]=alloc;allrows+=rr;stability+=ss
                csv_write(out/'temporal_metrics.csv',allrows);csv_write(out/'rank_turnover.csv',changes);csv_write(out/'subspace_stability.csv',stability)
                w.write(out/'allocation_by_stage.json',allocations)
        w.write(out/'training_activity.json',activity_finish(layers,initial,activity))
        report['base_verified']=w.shared.verify_disk_weights(base,args.model)
        report.update(status='PASS',fresh_AdamW=True,historical_momentum_restored=False,
            snapshot_sha256={str(k):w.shared.sha256(out/f'adapter_step_{k}.pt') for k in STEPS})
        w.write(out/'results.json',dict(trajectory_scope=SCOPE,interpretation='现有证据不足',reason='Capacity proxies only; inspect residual norms, M, consistency, independent audit and paired seeds together',allocations=allocations))
    except BaseException as e:report.update(status='FAIL',error=str(e),traceback=traceback.format_exc());raise
    finally:
        if 'input_sha256' in report:w.check_hashes(report['input_sha256'])
        w.write(out/'report.json',report)

def summarize(argv):
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs=3,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv); data={}; metric_data={}; fingerprint=None; codes=None
    for folder in args.runs:
        report=json.loads((folder/'report.json').read_text())
        w.require(report['status']=='PASS' and report['mode']=='train','Completed trajectory required')
        seed=report['seed'];w.require(seed in (42,43,44) and seed not in data,'Expected unique seeds 42/43/44')
        if fingerprint is None:fingerprint=report['plan_fingerprint'];codes=report['code_sha256']
        w.require(fingerprint==report['plan_fingerprint'] and codes==report['code_sha256'],'Different protocol/data/code')
        for step,h in report['snapshot_sha256'].items():w.require(w.shared.sha256(folder/f'adapter_step_{step}.pt')==h,'Snapshot changed')
        with (folder/'temporal_metrics.csv').open() as f:
            parsed=list(csv.DictReader(f))
        metric_data[seed]={}
        for row in parsed:
            for key in ('M_full','Q_residual','block_consistency','analysis'):row[key]=json.loads(row[key])
            key=(int(row['step']),row['module']);w.require(key not in metric_data[seed],'Duplicate metric row')
            metric_data[seed][key]=row
        w.require(len(metric_data[seed])==420 and {k[0] for k in metric_data[seed]}==set(STEPS),'Incomplete metric grid')
        data[seed]=json.loads((folder/'allocation_by_stage.json').read_text())
    rows=[]
    for step in STEPS:
        for kind in ('C','DE'):
            for a,b in itertools.combinations(sorted(data),2):rows.append(dict(step=step,kind=kind,seed_a=a,seed_b=b,**turnover(data[a][str(step)][kind],data[b][str(step)][kind])))
    args.output_dir.mkdir(parents=True,exist_ok=False)
    csv_write(args.output_dir/'cross_seed_stability.csv',rows)
    w.write(args.output_dir/'results.json',dict(trajectory_scope=SCOPE,metrics_by_stage_module=aggregate_metrics(metric_data),status='PASS',cross_seed=rows,conclusion='现有证据不足',
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
