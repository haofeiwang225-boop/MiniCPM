"""Fail-closed three-arm summaries from committed checkpoint lineage only."""
import argparse
import csv
import json
from pathlib import Path
from .protocol import ARMS,STAGES
from .state import write_json
from .benchmark import paired


def csv_write(path,rows):
    with path.open('x',newline='') as f:
        if rows:
            fields=sorted(set().union(*(r.keys() for r in rows)))
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def summarize(root,output=None):
    reports={a:json.loads((root/a/'report.json').read_text()) for a in ARMS}
    identities=[{k:v for k,v in r['identity'].items() if k!='arm'} for r in reports.values()]
    if not all(x==identities[0] for x in identities):raise ValueError('Unpaired experiments')
    train=[];metrics=[];updates=[];nll=[]
    for arm,r in reports.items():
        if r['status']!='PASS' or not r['complete'] or r['steps']!=2000:raise ValueError('Incomplete arm')
        if [x['step'] for x in r['history']]!=list(range(1,2001)):raise ValueError('Incomplete train history')
        for row in r['history']:train.append(dict(arm=arm,**{k:row[k] for k in ('step','index','loss','lr','side','rng_sha256')}))
        observed=set()
        for relative in r['records']:
            data=json.loads((root/relative).read_text())
            if isinstance(data,list):updates.extend(dict(arm=arm,**x) for x in data)
            else:
                step,split=data['step'],data['split']
                if step not in STAGES:continue
                key=(step,split)
                # Resume may remeasure exactly at checkpoint. Use last accepted lineage entry.
                if key in observed:
                    metrics=[x for x in metrics if not (x['arm']==arm and x['step']==step and x['probe_split']==split)]
                observed.add(key)
                losses=data['profile']['batch_losses']
                tokens=sum(x['effective_tokens'] for x in losses)
                nll=[x for x in nll if not (x['arm']==arm and x['step']==step and x['split']==split)]
                nll.append(dict(arm=arm,step=step,split=split,source='held-out training-side probe; NOT official test or Dev400',
                    question_nll=sum(x['nll'] for x in losses)/len(losses),token_nll=sum(x['nll_sum'] for x in losses)/tokens,effective_tokens=tokens))
                rows=data['rows']
                if len(rows)!=84*2*2 or len({(x['module'],x['weighting'],x['reference']) for x in rows})!=336:raise ValueError('Missing metric modules')
                metrics.extend(dict(arm=arm,**x) for x in rows)
        if observed!={(s,k) for s in STAGES for k in ('construction','audit')}:raise ValueError('Missing diagnostic stages')
    for step in range(2000):
        entries=[reports[a]['history'][step] for a in ARMS]
        if len({(x['index'],x['rng_sha256'],x['lr']) for x in entries})!=1:raise ValueError('Training pairing mismatch')
    output=output or root/'summary';output.mkdir(parents=True,exist_ok=False)
    csv_write(output/'training_comparison.csv',train);csv_write(output/'gradient_coverage.csv',metrics)
    csv_write(output/'update_direction.csv',updates)
    csv_write(output/'nll_comparison.csv',nll)
    from adaptive_lora.gid_validation import global_coverage
    grouped={}
    for row in metrics:
        keys=[('all','all'),('projection',row['module'].rsplit('.',1)[-1]),('rank',str(row['rank']))]
        for group,value in keys:
            key=(row['arm'],row['step'],row['probe_split'],row['reference'],row['weighting'],group,value)
            grouped.setdefault(key,[]).append(row)
    csv_write(output/'coverage_summary.csv',[dict(zip(('arm','step','probe_split','reference','weighting','group','value'),key),**global_coverage(rows)) for key,rows in grouped.items()])
    results={};benchmarks=[]
    for dataset in ('gsm8k','math500'):
        paths={a:root/'benchmarks'/a/dataset for a in ('base',*ARMS)}
        if not all((p/'report.json').exists() for p in paths.values()):
            results[dataset]={'status':'NOT_EVALUATED'};continue
        for path in paths.values():
            report=json.loads((path/'report.json').read_text())
            if report.get('status')!='PASS' or not report.get('complete'):raise ValueError('Incomplete benchmark')
            benchmarks.append(dict(model=path.parent.name,dataset=dataset,correct=report['correct'],total=report['total'],accuracy=report['accuracy'],ci_lower=report['ci95'][0],ci_upper=report['ci95'][1]))
        common={json.loads((p/'identity.json').read_text())['common_sha256'] for p in paths.values()}
        if len(common)!=1:raise ValueError('Benchmark protocol mismatch')
        rows={a:[json.loads(line) for line in (p/'samples.jsonl').read_text().splitlines()] for a,p in paths.items()}
        results[dataset]={f'{b}-{a}':paired(rows[a],rows[b]) for a,b in [('vanilla','altlora_false'),('vanilla','altlora_true'),('altlora_false','altlora_true')]}
    csv_write(output/'benchmark_comparison.csv',benchmarks)
    write_json(output/'paired_statistics.json',results)
    (output/'summary.md').write_text('三组从零实验。只有seed42，不代表跨seed优势。\n更新余弦参照Construction检查点梯度，滞后字段必须保留。\n正式测试结果见paired_statistics.json；NOT_EVALUATED不是零准确率。\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();summarize(a.root,a.output)
