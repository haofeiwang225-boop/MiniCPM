#!/usr/bin/env python3
"""CPU-only paired, crossed seed x original-question cluster bootstrap.

Two modules are fixed experimental strata, never independent replicates.
The same sampled seed IDs and question clusters are used for all modules/arms.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback
import numpy as np

MODULES=['model.layers.16.self_attn.q_proj','model.layers.18.self_attn.q_proj']
SEEDS=[42,43,44,45,46]
ARMS=['A','B','C']
COMPARISONS={'C-B':(2,1),'C-A':(2,0),'B-A':(1,0)}
REPEATS=10000
MARGIN=.0005


def require(ok,message):
    if not bool(ok): raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def identity(row):
    s=row['source']
    return (row['index'],s['index'],s['sample_sha256'],s['original_question_sha256'],s['family'],row['effective_tokens'])


def load_runs(paths):
    require(len(paths)==10,'Require all 2 modules x 5 seeds, no incomplete analysis')
    sums=np.empty((2,5,3,400),dtype=np.float64); tokens=None; meta=None
    seen=set(); common=None; sources=[]
    for path in paths:
        path=Path(path); report=json.loads((path/'report.json').read_text()); plan=json.loads((path/'plan.json').read_text())
        require(report['status']=='PASS' and report['mode']=='train','Only completed train runs accepted')
        require(report['input_files_unchanged'],'Input integrity not verified')
        coord=plan['coordinate']
        require(coord['common_fingerprint']==digest(plan['common']) and plan['fingerprint']==digest(coord),'Corrupt plan fingerprint')
        key=(coord['module'],coord['seed'])
        require(key not in seen and key[0] in MODULES and key[1] in SEEDS,'Duplicate/unknown coordinate')
        require(report['fingerprint']==plan['fingerprint'],'Run fingerprint mismatch')
        protocol=plan['common']['protocol']
        require(protocol['version']=='rank_gain_decisive_v1' and protocol['steps']==200,'Wrong endpoint protocol')
        require(protocol['min_C_minus_B_improvement']==MARGIN and protocol['C_minus_A_noninferiority_margin']==MARGIN,'Changed decision margins')
        require(protocol['bootstrap_repeats']==REPEATS and protocol['bootstrap_seed']==42,'Changed bootstrap protocol')
        if common is None: common=plan['common']
        require(plan['common']==common,'Different data, normalization, code, versions or settings across runs')
        seen.add(key); m,s=MODULES.index(key[0]),SEEDS.index(key[1]); baseline=None
        for a,arm in enumerate(ARMS):
            folder=path/arm; artifact=json.loads((folder/'artifacts.json').read_text())
            require(set(artifact)=={'result.json','train.jsonl','target_factors.pt'},'Missing artifacts')
            require(all(sha(folder/k)==v for k,v in artifact.items()),'Corrupt result artifact')
            r=json.loads((folder/'result.json').read_text())
            require(r['status']=='PASS' and r['fingerprint']==plan['fingerprint'] and r['arm']==arm,'Invalid arm')
            require(r['training']['completed_steps']==200 and len(r['training']['steps'])==200,'Missing step 200')
            steps=r['training']['steps']; expected=[plan['common']['indices']['train'][i] for i in coord['training_order']]
            require([x['step'] for x in steps]==list(range(1,201)) and [x['source_index'] for x in steps]==expected,'Training order mismatch')
            require(all(r['invariants'].values()),'Invariant failure')
            before=r['before']['samples']; rows=r['after']['samples']
            require(len(before)==len(rows)==400,'Expected fixed Dev400')
            ids=[identity(x) for x in rows]
            expected_meta=plan['common']['metadata']['dev']
            require([x['source'] for x in rows]==expected_meta and [x['index'] for x in rows]==list(range(400)), 'Dev differs from preregistered manifest')
            require([x['effective_tokens'] for x in rows]==plan['common']['effective_tokens']['dev'],'Dev tokenization differs from plan')
            require([identity(x) for x in before]==ids,'Before/after data changed')
            if meta is None:
                meta=ids; tokens=np.array([x['effective_tokens'] for x in rows],dtype=float)
            require(ids==meta,'Dev sample/source/hash/type/token mismatch')
            require(all(x[5]>0 for x in ids),'Zero tokens')
            initial=[(x['logits_sha256'],x['logits_shape'],x['logits_dtype']) for x in before]
            if baseline is None: baseline=initial
            require(initial==baseline,'Insertion logits were not identical')
            values=np.array([x['nll_sum'] for x in rows],dtype=float)
            require(np.isfinite(values).all() and (values>=0).all(),'Invalid NLL sums')
            require(np.allclose(values/tokens,[x['nll'] for x in rows],rtol=1e-12,atol=1e-12),'Inconsistent per-question NLL')
            require(math.isclose(float(values.sum()/tokens.sum()),r['after']['token_weighted_nll'],rel_tol=1e-12,abs_tol=1e-12),'NLL aggregate inconsistent')
            sums[m,s,a]=values
        sources.append(dict(path=str(path.resolve()),report_sha256=sha(path/'report.json'),plan_sha256=sha(path/'plan.json')))
    require(len(seen)==10,'Missing coordinate')
    return sums,tokens,meta,dict(common=common,inputs=sources)


def crossed_bootstrap(sums,tokens,groups,repeats=REPEATS,seed=42):
    """Resample problem groups and seeds independently, retain all module strata.

    Input [module,seed,arm,question] contains token NLL *sums*, not means.
    Within each draw all arms share identical denominators and sampled IDs.
    """
    require(sums.ndim==4 and sums.shape[-1]==len(tokens)==len(groups),'Bootstrap shape mismatch')
    require(np.isfinite(sums).all() and np.isfinite(tokens).all() and (tokens>0).all(),'Nonfinite bootstrap input')
    unique=list(dict.fromkeys(groups)); mapping={g:i for i,g in enumerate(unique)}
    membership=np.array([mapping[g] for g in groups]); rng=np.random.default_rng(seed)
    draws=np.empty((repeats,sums.shape[0],sums.shape[2]))
    for start in range(0,repeats,100):
        size=min(100,repeats-start)
        cluster_counts=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=size)
        weights=cluster_counts[:,membership]
        seed_ids=rng.integers(sums.shape[1],size=(size,sums.shape[1]))
        totals=np.einsum('msaq,bq->bmsa',sums,weights,optimize=True)
        denom=weights@tokens
        for j in range(size):
            draws[start+j]=totals[j][:,seed_ids[j],:].mean(axis=1)/denom[j]
    return draws


def ci(values):
    return [float(v) for v in np.quantile(values,[.025,.975])]


def decision(cb_mean,cb_interval,ca_interval,wins,module_means):
    gates=dict(effect_size=cb_mean<=-MARGIN,paired_C_B_interval=cb_interval[1]<0,
               at_least_8_of_10=wins>=8,both_modules_support_C=all(x<0 for x in module_means),
               C_A_noninferiority=ca_interval[1]<=MARGIN)
    if all(gates.values()): label='观察到收益'
    elif cb_interval[0]>0 or ca_interval[0]>MARGIN: label='观察到退化'
    else: label='证据不足'
    return dict(label=label,gates=gates,not_a_mathematical_proof=True,
                C_A_noninferiority_margin=MARGIN,criterion='All five preregistered gates required for benefit')


def analyze(sums,tokens,meta,repeats=REPEATS):
    groups=[x[3] for x in meta]; families=[x[4] for x in meta]
    require(set(families)=={'GSM','MATH'},'Unknown family')
    for g in set(groups): require(len({f for f,h in zip(families,groups) if h==g})==1,'Group spans families')
    result={}
    for label,indices in [('overall',np.arange(len(tokens)))]+[(f,np.array([i for i,t in enumerate(families) if t==f])) for f in ('GSM','MATH')]:
        data=sums[:,:,:,indices]; nt=tokens[indices]; grp=[groups[i] for i in indices]
        point=data.sum(axis=-1)/nt.sum()  # [module,seed,arm]
        draws=crossed_bootstrap(data,nt,grp,repeats=repeats)
        comparisons={}
        for name,(hi,lo) in COMPARISONS.items():
            delta=point[:,:,hi]-point[:,:,lo]
            boot=draws[:,:,hi]-draws[:,:,lo]
            per_item=(data[:,:,hi,:]-data[:,:,lo,:])/nt
            item_mean=per_item.mean(axis=(0,1))
            comparisons[name]=dict(mean_token_nll_difference=float(delta.mean()),ci95=ci(boot.mean(axis=1)),
                module_means={m:float(delta[i].mean()) for i,m in enumerate(MODULES)},
                module_ci95={m:ci(boot[:,i]) for i,m in enumerate(MODULES)},
                per_seed={str(seed):{m:float(delta[i,j]) for i,m in enumerate(MODULES)} for j,seed in enumerate(SEEDS)},
                better_coordinates=int((delta<0).sum()),worse_coordinates=int((delta>0).sum()),
                per_question=dict(improved=int((item_mean<0).sum()),worsened=int((item_mean>0).sum()),equal=int((item_mean==0).sum()),
                    mean=float(item_mean.mean()),quantiles={str(q):float(np.quantile(item_mean,q)) for q in (0,.05,.25,.5,.75,.95,1)},
                    rows=[dict(dev_index=int(indices[i]),sample_sha256=meta[indices[i]][2],original_question_sha256=grp[i],
                               mean_nll_difference=float(item_mean[i]),per_coordinate_nll_difference=per_item[:,:,i].tolist()) for i in range(len(indices))]),
                per_coordinate_question_counts={m:{str(seed):dict(improved=int((per_item[i,j]<0).sum()),worsened=int((per_item[i,j]>0).sum()),equal=int((per_item[i,j]==0).sum())) for j,seed in enumerate(SEEDS)} for i,m in enumerate(MODULES)})
        result[label]=dict(num_questions=len(indices),num_question_groups=len(set(grp)),effective_tokens=int(nt.sum()),
            mean_token_nll={arm:float(point[:,:,a].mean()) for a,arm in enumerate(ARMS)},
            per_seed_token_nll={m:{str(seed):{arm:float(point[i,j,a]) for a,arm in enumerate(ARMS)} for j,seed in enumerate(SEEDS)} for i,m in enumerate(MODULES)},
            mean_per_question_nll={arm:float((data[:,:,a,:]/nt).mean()) for a,arm in enumerate(ARMS)},comparisons=comparisons)
    cb=result['overall']['comparisons']['C-B']; ca=result['overall']['comparisons']['C-A']
    result['decision']=decision(cb['mean_token_nll_difference'],cb['ci95'],ca['ci95'],cb['better_coordinates'],list(cb['module_means'].values()))
    result['statistics']=dict(method='paired crossed bootstrap over original-question groups and seed IDs; fixed two-module strata',
        repeats=repeats,seed=42,interval='percentile 95%, no question/seed pseudo-replication',
        caveats=['Five seeds give limited seed-uncertainty precision.','GSM/MATH and secondary comparisons are descriptive, no simultaneous CI claim.',
                 'Existing Dev has prior use; not an independent blind generalization test.','PASS denotes engineering validity, not scientific benefit.'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING'}
    try:
        sums,tokens,meta,provenance=load_runs(args.runs)
        report.update(analyze(sums,tokens,meta),provenance=provenance,status='COMPLETE')
        print(report['decision']['label'])
        for name,r in report['overall']['comparisons'].items(): print(name,r['mean_token_nll_difference'],r['ci95'])
    except BaseException as exc:
        report.update(status='FAIL',error=str(exc),traceback=traceback.format_exc()); raise
    finally: (args.output_dir/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
