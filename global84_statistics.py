#!/usr/bin/env python3
"""CPU-only paired global84 analysis; engineering completion is not efficacy."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
from minicpm_rank_gain_statistics import crossed_bootstrap

SEEDS=[42,43,44]
ARMS=['A','B','C','D','E']
COMPARISONS={'E-D':(4,3),'E-C':(4,2),'D-B':(3,1),'E-B':(4,1),'E-A':(4,0)}


def require(ok,msg):
    if not bool(ok): raise ValueError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def load(paths):
    sums=np.empty((1,3,5,400)); tokens=None; metadata=None; common=None; seen=set(); allocations={}; orders={}; sources=[]
    for folder in paths:
        folder=Path(folder); report=json.loads((folder/'report.json').read_text()); plan=json.loads((folder/'plan.json').read_text())
        require(report['mode']=='train' and report['status']=='PASS' and report['input_files_unchanged'],'Only completed, intact training outputs accepted')
        require(plan['fingerprint']==digest(plan['common'])==report['plan_fingerprint'],'Plan fingerprint mismatch')
        require(plan['common']['protocol']['version']=='global84_stratified_v1','Wrong protocol')
        if common is None: common=plan['common']
        require(common==plan['common'],'Data/code/versions/protocol differ')
        seed=report['seed']; require(seed in SEEDS,'Unexpected seed')
        allocation=json.loads((folder/'allocation.json').read_text())
        require(digest(allocation['allocations']['D'])==digest(allocation['allocations']['E']),'D/E rank allocation mismatch')
        if seed in allocations: require(allocations[seed]==allocation,'Different allocation for same seed')
        allocations[seed]=allocation
        if seed in orders: require(orders[seed]==report['source_order'],'Different order in paired arms')
        orders[seed]=report['source_order']; require(len(orders[seed])==400 and len(set(orders[seed]))==400,'Not a 400-example permutation')
        for arm in report['completed_arms']:
            require(arm in ARMS and (seed,arm) not in seen,'Duplicate/unknown seed arm')
            seen.add((seed,arm)); path=folder/arm
            hashes=json.loads((path/'artifacts.json').read_text())
            require(set(hashes)=={'result.json','train.jsonl','adapter.pt'},'Missing artifacts')
            require(all(sha(path/k)==v for k,v in hashes.items()),'Artifact corruption')
            r=json.loads((path/'result.json').read_text())
            require(r['status']=='PASS' and r['seed']==seed and r['arm']==arm and r['plan_fingerprint']==plan['fingerprint'],'Wrong arm identity')
            require(r['training']['completed_steps']==400 and len(r['training']['steps'])==400,'Wrong endpoint')
            require([s['step'] for s in r['training']['steps']]==list(range(1,401)),'Step sequence incomplete')
            require([common['indices']['train'][s['pool_index']] for s in r['training']['steps']]==orders[seed],'Paired training order differs')
            require(r['added_parameters']==(0 if arm=='A' else 25600),'Parameter budget mismatch')
            require(r['training']['trainable_parameters']==480000+r['added_parameters'],'Actual trainable parameter count mismatch')
            require(r['allocation']==allocation['allocations'][arm],'Wrong applied allocation')
            require(all(r['invariants'].values()),'Failed safety invariant')
            before,after=r['before']['samples'],r['after']['samples']
            require(len(before)==len(after)==400,'Expected Dev400')
            for rows in (before,after):
                require([x['index'] for x in rows]==list(range(400)),'Dev order changed')
                require([x['source'] for x in rows]==common['metadata']['dev'],'Dev source/hash mismatch')
                require([x['effective_tokens'] for x in rows]==common['tokens']['dev'],'Dev token mismatch')
            if metadata is None:
                metadata=common['metadata']['dev']; tokens=np.array(common['tokens']['dev'],dtype=float)
            values=np.array([x['nll_sum'] for x in after])
            require(np.isfinite(values).all() and (tokens>0).all(),'Invalid NLL')
            require(np.allclose(values/tokens,[x['nll'] for x in after],rtol=1e-12,atol=1e-12),'Per-question NLL inconsistent')
            require(np.isclose(values.sum()/tokens.sum(),r['after']['token_weighted_nll'],rtol=1e-12,atol=1e-12),'Token NLL inconsistent')
            sums[0,SEEDS.index(seed),ARMS.index(arm)]=values
        sources.append(dict(path=str(folder.resolve()),report_sha256=sha(folder/'report.json'),plan_sha256=sha(folder/'plan.json')))
    require(seen=={(s,a) for s in SEEDS for a in ARMS},'Need complete 3 seeds x 5 arms; no omission of failed arms')
    return sums,tokens,metadata,dict(common=common,sources=sources)


def analyze(sums,tokens,metadata,repeats=10000):
    require(sums.shape==(1,3,5,len(tokens)),'Unexpected result grid')
    require(np.isfinite(sums).all() and (tokens>0).all(),'Nonfinite results')
    result={}
    for family in ('overall','GSM','MATH'):
        idx=np.array([i for i,m in enumerate(metadata) if family=='overall' or m['family']==family])
        require(len(idx)>0,'Missing family')
        x=sums[:,:,:,idx]; n=tokens[idx]; groups=[metadata[i]['original_question_sha256'] for i in idx]
        means=x.sum(axis=-1)/n.sum()
        draws=crossed_bootstrap(x,n,groups,repeats=repeats,seed=42)[:,0,:]
        comparisons={}
        for name,(hi,lo) in COMPARISONS.items():
            dif=means[0,:,hi]-means[0,:,lo]; bootstrap=draws[:,hi]-draws[:,lo]
            individual=(x[0,:,hi,:]-x[0,:,lo,:])/n
            average=individual.mean(axis=0)
            comparisons[name]=dict(mean_token_nll_difference=float(dif.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                per_seed={str(s):float(dif[i]) for i,s in enumerate(SEEDS)},better_seeds=int((dif<0).sum()),
                question_counts=dict(improved=int((average<0).sum()),worsened=int((average>0).sum()),equal=int((average==0).sum())),
                per_question=[dict(source=metadata[int(j)],mean_nll_difference=float(average[i]),seed_differences=individual[:,i].tolist()) for i,j in enumerate(idx)])
        result[family]=dict(num_samples=len(idx),effective_tokens=int(n.sum()),
            mean_token_nll={a:float(means[0,:,i].mean()) for i,a in enumerate(ARMS)},
            mean_question_nll={a:float((x[0,:,i,:]/n).mean()) for i,a in enumerate(ARMS)},
            per_seed_nll={str(s):{a:float(means[0,j,i]) for i,a in enumerate(ARMS)} for j,s in enumerate(SEEDS)},comparisons=comparisons)
    primary=[result['overall']['comparisons'][c] for c in ('E-D','E-C')]
    result['scientific_interpretation']=('主要比较均观察到改善，仍需独立验证' if all(x['ci95'][1]<0 for x in primary)
        else '主要比较存在退化证据' if any(x['ci95'][0]>0 for x in primary) else '主要比较的收益证据不足')
    result['statistical_protocol']=dict(repeats=repeats,seed=42,method='paired crossed bootstrap: original-question clusters and 3 seed IDs',
        primary=['E-D','E-C'],secondary=['D-B','E-B','E-A'],engineering_PASS_is_not_efficacy=True,
        caveats=['Only 3 seeds; uncertainty estimate is imprecise.','Intervals are descriptive 95%, not simultaneous multiple-comparison guarantees.',
                 'No minimum-effect success margin was supplied for this new protocol; no previous margin is imported.',
                 'Existing Dev is not blind. Proxy improvement does not prove post-training benefit.'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--runs',type=Path,nargs='+',required=True); p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=False); report=dict(status='RUNNING')
    try:
        sums,tokens,meta,provenance=load(args.runs)
        report.update(analyze(sums,tokens,meta),provenance=provenance,status='COMPLETE')
        print(report['scientific_interpretation'])
        for key,row in report['overall']['comparisons'].items(): print(key,row['mean_token_nll_difference'],row['ci95'])
    except BaseException as error:
        report.update(status='FAIL',error=str(error),traceback=traceback.format_exc()); raise
    finally: (args.output_dir/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
