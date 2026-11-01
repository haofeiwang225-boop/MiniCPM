"""Paired seed/question-cluster CPU analysis; final continuation-step300 endpoint."""
import argparse,json
from pathlib import Path
import numpy as np
from minicpm_rank_gain_statistics import crossed_bootstrap
from global84_step100_rank_gain import ARMS,w

def analyze(paths):
    seen={};meta=None;tokens=None;sources=[];protocol=None
    for p in paths:
        r=json.loads((p/'report.json').read_text());w.require(r['status']=='PASS' and r['mode']=='train','Completed full training required')
        seed=r['seed'];w.require(seed in (42,43,44) and seed not in seen,'Unique three seeds required');seen[seed]=[]
        common=(r['inputs'],r['code_sha256'])
        if protocol is None:protocol=common
        w.require(protocol==common,'Input/code mismatch')
        for arm in ARMS:
            result=json.loads((p/arm/'result.json').read_text());w.require(result['status']=='PASS','Failed arm')
            rows=result['evaluations']['300']['samples'];w.require(len(rows)==400,'Dev400 required')
            current=[x['source'] for x in rows];n=np.array([x['effective_tokens'] for x in rows])
            if meta is None:meta=current;tokens=n
            w.require(meta==current and np.array_equal(tokens,n),'Paired evaluation mismatch')
            seen[seed].append([x['nll_sum'] for x in rows])
        sources.append(str(p.resolve()))
    w.require(set(seen)=={42,43,44},'Need all three seeds')
    x=np.array([[seen[s] for s in (42,43,44)]]);out={}
    for family in ('overall','GSM','MATH'):
        ids=[i for i,m in enumerate(meta) if family=='overall' or m['family']==family];v=x[:,:,:,ids];n=tokens[ids]
        draws=crossed_bootstrap(v,n,[meta[i]['original_question_sha256'] for i in ids],repeats=10000,seed=42)[:,0,:]
        means=v.sum(-1)[0]/n.sum();comparisons={}
        for hi,lo in [(3,1),(4,1),(3,2),(4,2),(1,0),(2,0),(3,0),(4,0)]:
            diff=means[:,hi]-means[:,lo];sample=(v[0,:,hi]-v[0,:,lo])/n
            comparisons[f'{ARMS[hi]}-{ARMS[lo]}']=dict(per_seed=diff.tolist(),mean=float(diff.mean()),std=float(diff.std(ddof=1)),ci95=np.quantile(draws[:,hi]-draws[:,lo],[.025,.975]).tolist(),
                primary_ci97_5=np.quantile(draws[:,hi]-draws[:,lo],[.0125,.9875]).tolist() if lo==1 and hi in (3,4) else None,per_question_seed_differences=sample.tolist())
        out[family]=dict(per_seed_nll=means.tolist(),comparisons=comparisons)
    return dict(results=out,sources=sources,arms=ARMS,seed_order=[42,43,44],method='10000 paired crossed bootstrap: original-question groups and seed IDs',
        caveat='Only three training seeds; intervals may be poorly calibrated. Dev previously used, not blind. Primary comparisons C/random and DE/random have Bonferroni97.5% intervals; others exploratory.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs=3,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    w.write(a.output_dir/'results.json',analyze(a.runs))
if __name__=='__main__':main()
