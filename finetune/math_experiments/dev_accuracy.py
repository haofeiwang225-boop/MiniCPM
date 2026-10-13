"""MetaMath dev generation using the frozen author-email MATH-500 protocol."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
# File-based import avoids an unrelated installed `benchmarks` package.
import importlib.util
_math_root=Path(__file__).resolve().parents[2]/'benchmarks/math500'
sys.path.insert(0,str(_math_root))
_spec=importlib.util.spec_from_file_location('frozen_math_endpoint',_math_root/'evaluate.py')
_endpoint=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_endpoint)
build_request, endpoint_json, score_response = _endpoint.build_request, _endpoint.endpoint_json, _endpoint.score_response
parse_answer, grade_answer = _endpoint.parse_answer, _endpoint.grade_answer
from adaptive_lora.data_provenance import file_sha256, git_version


def load_dev(path, metadata_path):
    rows=json.loads(Path(path).read_text()); meta=json.loads(Path(metadata_path).read_text())
    if meta['data_sha256']!=file_sha256(path) or len(meta['samples'])!=len(rows):
        raise ValueError('Dev metadata hash/count mismatch')
    samples=[]
    for index,(row,info) in enumerate(zip(rows,meta['samples'])):
        digest=hashlib.sha256(json.dumps(row,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        if info['index']!=index or info['sample_sha256']!=digest or info['family'] not in ('GSM','MATH'):
            raise ValueError(f'Dev metadata mismatch at {index}')
        messages=row['messages']
        if len(messages)!=2 or [m['role'] for m in messages]!=['user','assistant']:
            raise ValueError('Expected existing single-turn dev format')
        problem,reference=[m['content'] for m in messages]
        # MetaMath references are complete worked responses, not MATH-500 answer fields.
        markers=list(re.finditer(r'The answer is:\s*',reference))
        if not markers: raise ValueError(f'No explicit reference declaration at {index}')
        tail=reference[markers[-1].end():].strip()
        if not tail or '\n' in tail: raise ValueError(f'Ambiguous reference suffix at {index}')
        gold=parse_answer('The answer is: '+tail,problem=problem)['answer']
        if not gold: raise ValueError(f'Empty reference answer at {index}')
        gsm=re.findall(r'^####\s*(.+)$',reference,re.M)
        reference_check='explicit_final_declaration'
        if gsm and gsm[-1].strip()!=gold.strip():
            check=grade_answer(gsm[-1],gold,problem=problem)
            if not check['correct']: raise ValueError(f'Conflicting reference declarations at {index}: {check}')
            reference_check='declaration_and_hash_marker_equivalent'
        samples.append(dict(question_id=index,problem=problem,answer=gold,family=info['family'],
                            sample_sha256=digest,reference_extraction=reference_check,
                            original_question_sha256=info['original_question_sha256']))
    return samples


def wilson(correct,total):
    if not total:return None
    z=1.959963984540054; p=correct/total; d=1+z*z/total
    mid=(p+z*z/(2*total))/d
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [mid-radius,mid+radius]


def summarize(rows,expected):
    result=dict(expected_total=expected,total=len(rows),complete=len(rows)==expected)
    for family in ('overall','GSM','MATH'):
        subset=rows if family=='overall' else [r for r in rows if r['family']==family]
        correct=sum(r['correct'] for r in subset)
        result[family]=dict(total=len(subset),correct=correct,accuracy=correct/len(subset) if subset else None,
                            wilson95=wilson(correct,len(subset)))
    result['parse_reasons']=dict(Counter(r['parse_reason'] for r in rows))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]/'adaptive_lora/data/metamath_pilot_v1'
    p.add_argument('--data',type=Path,default=root/'dev.json')
    p.add_argument('--metadata',type=Path,default=root/'dev.source_metadata.json')
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--base_url',default='http://127.0.0.1:8000/v1')
    p.add_argument('--server_record',type=Path,required=True)
    p.add_argument('--max_samples',type=int,default=10)
    p.add_argument('--audit_only',action='store_true')
    args=p.parse_args()
    if args.max_samples<=0:p.error('max_samples must be positive')
    samples=load_dev(args.data,args.metadata)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'reference_audit.json').write_text(json.dumps(samples,indent=2))
    if args.audit_only:
        print(dict(total=len(samples),families=dict(Counter(s['family'] for s in samples))))
        return
    record=json.loads(args.server_record.read_text())
    config=dict(model=record['model'],served_model_name=record['served_model_name'],enable_thinking=True,
                generation=dict(temperature=1.,top_p=.95,n=1))
    metadata=endpoint_json(args.base_url+'/models')
    if not any(m['id']==config['served_model_name'] and m.get('max_model_len')==131072 for m in metadata['data']):
        raise ValueError('Endpoint model/context differs from author protocol')
    source_root=Path(__file__).resolve().parents[2]/'benchmarks/math500'
    run=dict(**config,**git_version(),dev_sha256=file_sha256(args.data),server_record=record,
             server_record_sha256=file_sha256(args.server_record),endpoint_metadata=metadata,
             frozen_files={n:file_sha256(source_root/n) for n in ('evaluate.py','prompt.py','answer_parser.py','grader.py')},
             request_seed=None,server_seed=record['seed'],sampling_note='one stochastic draw; not deterministic accuracy')
    (args.output_dir/'run_config.json').write_text(json.dumps(run,indent=2))
    samples=samples[:args.max_samples]; results=[]
    with (args.output_dir/'results.jsonl').open('x') as stream, (args.output_dir/'requests.jsonl').open('x') as requests:
        for sample in samples:
            payload=build_request(config,sample['problem'])
            requests.write(json.dumps(dict(question_id=sample['question_id'],request=payload))+'\n'); requests.flush()
            try:
                response=endpoint_json(args.base_url+'/chat/completions',payload)
                if len(response.get('choices',[]))!=1:raise ValueError('Expected exactly one choice')
                choice=response['choices'][0]; raw=choice['message'].get('content') or ''
                final,finished,extraction,grade=score_response(raw,sample['problem'],sample['answer'])
                row=dict(**sample,model_output=raw,final_response=final,thinking_finished=finished,
                         parsed_answer=extraction['answer'],parse_reason=extraction['reason'],grading=grade,
                         correct=bool(grade['correct']),generated_tokens=response.get('usage',{}).get('completion_tokens'),
                         finish_reason=choice.get('finish_reason'),endpoint_response=response)
            except Exception as error:
                row=dict(**sample,model_output=None,final_response='',thinking_finished=False,
                         parsed_answer=None,parse_reason='endpoint_or_scoring_error',correct=False,
                         generated_tokens=None,finish_reason='error',error=repr(error))
            row.update(request=payload,ended_with_eos=None)
            stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush();results.append(row)
            (args.output_dir/'summary.json').write_text(json.dumps(summarize(results,len(samples)),indent=2))
            print(f"{sample['question_id']}: correct={row['correct']} reason={row['parse_reason']} tokens={row['generated_tokens']}",flush=True)


if __name__=='__main__':main()
