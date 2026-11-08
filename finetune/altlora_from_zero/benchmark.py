"""Pinned local official test files; existing math500 prompting/parser/grader.
No download and no claim that repository grader is an official scoring release.
"""
import argparse
import json
import math
from pathlib import Path
import random
from .protocol import file_hash, canonical_hash, ARMS


def load_official_samples(path, dataset):
    if dataset not in ('gsm8k','math500'):raise ValueError('Unsupported official dataset')
    rows=[]
    with Path(path).open() as stream:
        for index,line in enumerate(stream):
            if not line.strip():raise ValueError('Blank test row: '+str(index))
            row=json.loads(line)
            if dataset=='gsm8k':
                problem=row['question'];answer=row['answer']
                if answer.count('####')!=1:raise ValueError('GSM8K reference must have one #### final-answer delimiter')
                gold=answer.split('####')[1].strip()
            else:problem=row['problem'];gold=row['answer']
            if not isinstance(problem,str) or not problem.strip() or not isinstance(gold,str) or not gold.strip():raise ValueError('Invalid reference')
            rows.append(dict(index=index,problem=problem,answer=gold,sample_sha256=canonical_hash(row)))
    if len(rows)!=(1319 if dataset=='gsm8k' else 500):raise ValueError('Official test count mismatch')
    if len({r['problem'] for r in rows})!=len(rows):raise ValueError('Duplicate official question')
    return rows


def wilson(correct,total):
    if total<=0:raise ValueError('Empty accuracy sample')
    z=1.959963984540054;p=correct/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [center-half,center+half]


def paired(left,right,seed=42,repetitions=10000):
    if len(left)!=len(right) or not left:raise ValueError('Unpaired benchmark')
    if any(a['sample_sha256']!=b['sample_sha256'] for a,b in zip(left,right)):raise ValueError('Different benchmark samples/order')
    delta=[int(b['correct'])-int(a['correct']) for a,b in zip(left,right)]
    rng=random.Random(seed);n=len(delta)
    boot=sorted(sum(delta[rng.randrange(n)] for _ in range(n))/n for _ in range(repetitions))
    wins=delta.count(1);losses=delta.count(-1);discordant=wins+losses
    # Exact two-sided McNemar/binomial conditional test, no extra dependency.
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.
    return dict(right_minus_left=sum(delta)/n,paired_bootstrap_95=[boot[int(.025*repetitions)],boot[int(.975*repetitions)]],
        right_only_correct=wins,left_only_correct=losses,mcnemar_exact_p=p,seed=seed,
        limitation='question uncertainty at one training seed; not cross-training-seed evidence')


def evaluate(root,model_key,dataset,endpoint,served_model,server_manifest):
    from benchmarks.math500.evaluate import build_request,endpoint_json,score_response
    from .state import write_json
    # Test use is blocked until all three formal arms have completed.
    shared_identity=json.loads((root/'shared/identity.json').read_text())
    for arm in ARMS:
        report=json.loads((root/arm/'report.json').read_text())
        if report.get('status')!='PASS' or not report.get('complete') or report.get('steps')!=2000:raise ValueError('All formal training must finish before tests')
        if report['identity']['pair_id']!=shared_identity['pair_id'] or report['identity']['arm']!=arm:raise ValueError('Formal arms are not paired')
    config=json.loads((root/'shared/config.json').read_text())
    spec=config[dataset];path=Path(spec['path'])
    if file_hash(path)!=spec['sha256'] or not spec.get('revision') or not spec.get('source_url'):raise ValueError('Dataset version/hash not pinned')
    # User-supplied deployment manifest binds served endpoint to exported files;
    # a network API cannot independently prove that the server actually loaded them.
    server=json.loads(server_manifest.read_text())
    expected=(config['base_hashes'] if model_key=='base' else
              json.loads((root/'exports'/model_key/'export_manifest.json').read_text())['files'])
    if server['model_key']!=model_key or server['served_model']!=served_model or server['files']!=expected:raise ValueError('Server deployment identity mismatch')
    protocol=dict(enable_thinking=True,generation={'temperature':1.,'top_p':.95,'n':1},served_model_name=served_model)
    samples=load_official_samples(path,dataset)
    output=root/'benchmarks'/model_key/dataset;output.mkdir(parents=True,exist_ok=False)
    implementation={str(Path(p)):file_hash(Path(p)) for p in ('benchmarks/math500/evaluate.py','benchmarks/math500/prompt.py','benchmarks/math500/answer_parser.py','benchmarks/math500/grader.py')}
    for name,digest in implementation.items():
        if config['code_hashes'].get(name)!=digest:raise ValueError('Scoring code changed since experiment preparation: '+name)
    common=dict(dataset=spec,protocol={k:v for k,v in protocol.items() if k!='served_model_name'},scoring_code=implementation,
        server_sampling=server['sampling'],seed=42)
    if server['sampling'].get('temperature')!=1. or server['sampling'].get('top_p')!=.95 or server['sampling'].get('n')!=1:raise ValueError('Actual server sampling differs')
    write_json(output/'identity.json',dict(common=common,common_sha256=canonical_hash(common),model_key=model_key,server=server,
        grader='repository math500 grader, NOT claimed official scorer',generation='stochastic; no retry; no added max_tokens cap'))
    rows=[]
    try:
        with (output/'samples.jsonl').open('x') as log:
            for sample in samples:
                request=build_request(protocol,sample['problem']);request['seed']=42
                response=endpoint_json(endpoint.rstrip('/')+'/chat/completions',request)
                if len(response['choices'])!=1:raise ValueError('Expected n=1 response')
                choice=response['choices'][0];raw=choice['message']['content']
                final,finished,extraction,grading=score_response(raw,sample['problem'],sample['answer'])
                row=dict(**sample,raw_generation=raw,final=final,thinking_finished=finished,extraction=extraction,
                    grading=grading,correct=bool(grading['correct']),finish_reason=choice.get('finish_reason'),
                    usage=response.get('usage'),response=response)
                log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush();rows.append(row)
        correct=sum(r['correct'] for r in rows)
        write_json(output/'report.json',dict(status='PASS',complete=True,total=len(rows),correct=correct,accuracy=correct/len(rows),ci95=wilson(correct,len(rows))))
    except BaseException:
        import traceback
        write_json(output/'failure.json',dict(status='FAIL',completed=len(rows),traceback=traceback.format_exc()));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model-key',choices=['base',*ARMS],required=True);p.add_argument('--dataset',choices=['gsm8k','math500'],required=True)
    p.add_argument('--endpoint',required=True);p.add_argument('--served-model',required=True);p.add_argument('--server-manifest',type=Path,required=True)
    a=p.parse_args();evaluate(a.root,a.model_key,a.dataset,a.endpoint,a.served_model,a.server_manifest)

if __name__=='__main__':main()
