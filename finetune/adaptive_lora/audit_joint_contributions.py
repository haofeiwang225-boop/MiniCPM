"""Historical step1600 subset ablation: CPU preflight, then explicit forward-only run.

No candidate generation or data-dependent search. NLL is never consulted when
constructing policies. Independent layer updates have a concatenated Frobenius
norm sqrt(sum_l ||s_l delta(B_l A_l)||_F^2), not a network-function distance.
"""
import argparse
import json
import math
from pathlib import Path
import random

import torch

from .audit_cross_batch_acceptance import (verified, extract_historical_candidates,
    bind_historical_candidates, preserved_state, temporary_candidates, candidate_check)
from .data_provenance import file_sha256, git_version, build_source_metadata
from .direction_correction import relative_A_projection_change, relative_B_projection_change
from evaluate_adapter_loss import evaluate_dataset


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(row):
    import hashlib
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def checked_result(result, records):
    require(result.get('pre_factors_restored') is True and not result.get('invariant_violations')
            and not result.get('error'), 'Historical A/B audit was not successfully restored')
    expected = {r['module']: r['side'] for r in records}
    rows = result['candidates']
    require(len(rows) == len(expected) and len({r['module'] for r in rows}) == len(rows),
            'A/B candidate coverage or duplicates')
    selected, negative, slopes = set(), set(), {}
    for r in rows:
        name = r['module']
        require(name in expected and r['side'] == expected[name], 'A/B historical side mismatch')
        values = []
        for group in ('proposal', 'selector'):
            entry = r[group]
            require(len(entry['per_sample']) == len(result['groups'][group]) and entry['per_sample'],
                    'Directional derivative sample count mismatch')
            require(all(math.isfinite(v) for v in entry['per_sample']) and math.isfinite(entry['mean']),
                    'Nonfinite saved slope')
            require(math.isclose(math.fsum(entry['per_sample'])/len(entry['per_sample']), entry['mean'],
                                 abs_tol=1e-15, rel_tol=1e-12), 'Saved slope mean mismatch')
            values.append(entry['mean'])
        require(r['check']['feasible'] is True, 'Historical candidate failed constraints')
        accepted = values[0] < 0 and values[1] < 0
        require(type(r['selected']) is bool and r['selected'] == accepted, 'Saved selection/slope mismatch')
        if accepted: selected.add(name)
        if values[0] < 0: negative.add(name)
        slopes[name] = r['proposal']
    return selected, negative, slopes


def make_policies(a, b, records, seed, random_count=30, singletons=False):
    fa, na, sa = checked_result(a, records)
    fb, nb, sb = checked_result(b, records)
    require(a['groups']['proposal'] == b['groups']['proposal'] and sa == sb and na == nb,
            'A/B must share identical Proposal samples and slopes')
    names = {r['module'] for r in records}
    require(type(seed) is int and type(random_count) is int and 0 < random_count <= len(names),
            'Invalid fixed random seed/count')
    rng = random.Random(seed)  # Never consume the training/global RNG.
    pool = sorted(na)
    if len(pool) >= random_count:
        chosen = rng.sample(pool, random_count)
        fallback = False
    else:
        chosen = pool + rng.sample(sorted(names-na), random_count-len(pool))
        fallback = True
    policies = dict(PRE=set(), ALL=names, FA=fa, FB=fb, COMMON=fa & fb,
                    FA_ONLY=fa-fb, FB_ONLY=fb-fa, UNION=fa | fb, RANDOM30=set(chosen))
    if singletons:
        policies.update({f'SINGLE::{n}': {n} for n in sorted(fa | fb)})
    counts = dict(FA=len(fa), FB=len(fb), COMMON=len(fa & fb), FA_ONLY=len(fa-fb),
                  FB_ONLY=len(fb-fa), proposal_negative=len(pool))
    return {k: sorted(v) for k,v in policies.items()}, dict(
        counts=counts, random_seed=seed, random_count=random_count, random_pool=pool,
        random_fallback_to_remaining=fallback, random_selected=sorted(chosen))


def validate_metadata(data_path, metadata):
    raw = read_json(data_path)
    require(metadata['data_sha256'] == file_sha256(data_path), 'Metadata data SHA mismatch')
    rows = metadata['samples']
    require(len(rows) == len(raw) == metadata['num_samples'], 'Metadata sample count mismatch')
    for i, (row, sample) in enumerate(zip(raw, rows)):
        group = sample.get('original_question_sha256', '')
        require(sample['index'] == i and sample['sample_sha256'] == fingerprint(row),
                f'Metadata index/sample SHA mismatch: {i}')
        require(isinstance(group, str) and len(group) == 64 and all(c in '0123456789abcdef' for c in group),
                f'Missing original question group: {i}')
    return raw


def validate_isolation(train_meta, eval_meta, calibration, evaluation, used, separate_data):
    def indices(ids, size, label):
        require(ids and len(set(ids)) == len(ids) and all(type(i) is int and 0 <= i < size for i in ids),
                f'Invalid {label} indices')
    indices(calibration, len(train_meta['samples']), 'calibration')
    indices(evaluation, len(eval_meta['samples']), 'evaluation')
    indices(sorted(used), len(train_meta['samples']), 'historical calibration/audit')
    def groups(meta, ids):
        return {meta['samples'][i]['original_question_sha256'] for i in ids}
    calibration_groups = groups(train_meta, calibration)
    historical_groups = groups(train_meta, used)
    eval_groups = groups(eval_meta, evaluation)
    # Explicit new calibration may reuse Proposal/Selector samples, but never eval.
    forbidden = groups(train_meta, range(len(train_meta['samples']))) if separate_data else historical_groups | calibration_groups
    require(not (eval_groups & forbidden), 'Evaluation original-question groups overlap training/calibration/history')
    return dict(calibration_original_groups=len(calibration_groups), evaluation_original_groups=len(eval_groups),
                historical_original_groups=len(historical_groups), original_group_overlap=0,
                evaluation_scope='held_out_dataset' if separate_data else 'reserved_training_samples_not_dev',
                evaluation_compared_against='entire_training_dataset' if separate_data else 'all_A_B_groups_and_calibration')


def checked_inputs(args):
    paths = {k: str(getattr(args, k).resolve()) for k in
             ('results_a','results_b','pre_adapter','post_adapter','historical_audit','evaluation_data','evaluation_metadata')}
    a, b = read_json(paths['results_a']), read_json(paths['results_b'])
    cfg = a['config']
    require(cfg == b['config'] and a['dtype'] == b['dtype'], 'A/B model/config/precision mismatch')
    require(cfg['model_family'] == 'minicpm5' and cfg['lora_structure'] == 'dense', 'MiniCPM5 Dense only')
    paths['train_data'] = str(Path(cfg['train_data_path']).resolve())
    paths['calibration_manifest'] = str(Path(cfg['calibration_indices_path']).resolve())
    hashes = {p: file_sha256(p) for p in paths.values()}
    for r in (a,b):
        require(r['pre_sha256'] == hashes[paths['pre_adapter']] and r['post_sha256'] == hashes[paths['post_adapter']],
                'A/B PRE/POST hashes mismatch')
        require(r['input_sha256'].get(paths['train_data']) == hashes[paths['train_data']], 'A/B train hash mismatch')
        require(r['input_sha256'].get(paths['historical_audit']) == hashes[paths['historical_audit']], 'A/B history hash mismatch')
    verified(paths['train_data'], cfg['train_data_sha256'])
    verified(paths['calibration_manifest'], cfg['calibration_indices_sha256'])
    calibration = read_json(paths['calibration_manifest'])
    require(calibration['split'] == 'train' and calibration['batch_size'] == 1 and
            calibration['batch_indices'] == [[i] for i in a['groups']['proposal']], 'Original Proposal calibration mismatch')
    history = read_json(paths['historical_audit'])
    verified(paths['pre_adapter'], history['checkpoint_pre_sha256'])
    verified(paths['post_adapter'], history['checkpoint_post_sha256'])
    pre = torch.load(paths['pre_adapter'], map_location='cpu', weights_only=True)
    post = torch.load(paths['post_adapter'], map_location='cpu', weights_only=True)
    require(Path(pre['base_model']).resolve() == Path(cfg['model_name_or_path']).resolve() and
            set(pre['layers']) == set(cfg['targets']), 'PRE model/targets mismatch')
    records, changes, counts = extract_historical_candidates(pre, post, cfg['direction_function_tolerance'], history)
    policies, selection = make_policies(a,b,records,args.random_seed,30,args.include_singletons)
    require(selection['counts'] == dict(FA=24,FB=30,COMMON=8,FA_ONLY=16,FB_ONLY=22,proposal_negative=54),
            f'A/B experiment profile changed: {selection["counts"]}; no masks were substituted')
    # Build a new sidecar in memory, never write over training data or its metadata.
    source = args.train_source.resolve()
    train_meta = build_source_metadata(paths['train_data'], source)
    hashes[str(source)] = train_meta['source_sha256']
    eval_meta = read_json(paths['evaluation_metadata'])
    eval_source = Path(eval_meta['source_path']).resolve()
    if str(eval_source) not in hashes: hashes[str(eval_source)] = file_sha256(eval_source)
    require(hashes[str(eval_source)] == eval_meta['source_sha256'], 'Evaluation source hash mismatch')
    raw = validate_metadata(paths['train_data'], train_meta)
    validate_metadata(paths['evaluation_data'], eval_meta)
    used = set()
    for r in (a,b):
        for group, ids in r['groups'].items():
            require(ids and len(set(ids)) == len(ids) and all(type(i) is int and 0 <= i < len(raw) for i in ids),
                    'A/B sample indices invalid')
            require([fingerprint(raw[i]) for i in ids] == r['sample_sha256'][group], 'A/B sample SHA mismatch')
            used.update(ids)
    isolation = validate_isolation(train_meta, eval_meta, args.calibration_indices, args.evaluation_indices,
                                   used, paths['evaluation_data'] != paths['train_data'])
    plan = dict(version=1, mode='historical_joint_ablation', **git_version(), paths=paths, input_sha256=hashes,
        config=cfg, dtype=a['dtype'], policies=policies, selection=selection, historical_changes=changes,
        historical_sides=counts, calibration_indices=args.calibration_indices, evaluation_indices=args.evaluation_indices,
        isolation=isolation, selection_uses_evaluation=False,
        training_metadata=train_meta, evaluation_metadata=eval_meta,
        include_singletons=args.include_singletons,
        norm_definition='sqrt(sum_layer ||scaling * (Bpost Apost - Bpre Apre)||_F^2)',
        limitation='Fixed historical directions; block interaction is not per-layer causal attribution.')
    return plan


def policy_norms(pre, records, policies):
    by_name = {c['module']: c for c in records}
    norms = {}
    for name,c in by_name.items():
        spec = pre['layers'][name]
        a,b = spec['state']['weight_a'], spec['state']['weight_b']
        fn = relative_A_projection_change if c['side']=='A' else relative_B_projection_change
        relative, change, denominator = fn(a,b,c['target'])
        norms[name] = dict(side=c['side'], scaled_update_fro_norm=abs(spec['scaling'])*change,
                           relative_update_change=relative, denominator=denominator)
    return {policy: dict(num_changed_modules=len(names), per_module={n:norms[n] for n in names},
                        lora_update_fro_norm=math.sqrt(math.fsum(norms[n]['scaled_update_fro_norm']**2 for n in names)))
            for policy,names in policies.items()}


def interactions(results):
    """Negative interaction means joint loss change is better than additive blocks."""
    output = {}
    for group in ('calibration','evaluation'):
        loss = {p:r[group]['token_weighted_nll'] for p,r in results.items()}
        baseline = loss['PRE']
        output[group] = dict(
            FA_only_given_common=loss['FA']-loss['COMMON'],
            FB_only_given_common=loss['FB']-loss['COMMON'],
            FA_block_interaction=loss['FA']-loss['COMMON']-loss['FA_ONLY']+baseline,
            FB_block_interaction=loss['FB']-loss['COMMON']-loss['FB_ONLY']+baseline,
            union_block_interaction=loss['UNION']-loss['COMMON']-loss['FA_ONLY']-loss['FB_ONLY']+2*baseline,
            FB_minus_random30=loss['FB']-loss['RANDOM30'])
        singles = {k[8:]:v for k,v in loss.items() if k.startswith('SINGLE::')}
        if singles:
            for policy in ('FA','FB','COMMON','FA_ONLY','FB_ONLY','UNION'):
                names=results[policy]['modules']
                output[group][policy+'_minus_sum_singletons'] = loss[policy]-baseline-math.fsum(singles[n]-baseline for n in names)
    return output


def evaluate_policies(model, candidates, policies, datasets, tolerance):
    require('PRE' in policies and not policies['PRE'], 'Require an empty PRE baseline')
    by_name = {c['module']:c for c in candidates}
    require(len(by_name)==len(candidates), 'Duplicate candidate module')
    for c in candidates:
        require(candidate_check(c,tolerance)['feasible'], 'Loaded candidate rank/trust mismatch')
    originals = {c['parameter']:c['parameter'].detach().clone() for c in candidates}
    metadata = [(n,p,p._version,p.requires_grad,tuple(p.shape)) for n,p in model.named_parameters()]
    ranks = {c['module']:(c['layer'].lora_rank,c['layer'].lora_scaler) for c in candidates}
    results = {}
    try:
        with preserved_state(model):
            for policy,names in policies.items():
                require(len(set(names))==len(names) and set(names)<=set(by_name), 'Invalid policy members')
                with temporary_candidates([by_name[n] for n in names]):
                    values = {}
                    for group,(dataset,ids) in datasets.items():
                        summary,rows = evaluate_dataset(model,[dataset[i] for i in ids],next(model.parameters()).device)
                        for row,index in zip(rows,ids):
                            row['index']=index
                            row['nll']=row['nll_sum']/row['effective_tokens']
                        values[group]=dict(**summary,samples=rows)
                require(all(torch.equal(p,old) for p,old in originals.items()), 'Policy failed to restore PRE')
                results[policy]=dict(modules=names,**values)
            for policy,r in results.items():
                for group in datasets:
                    r[group]['delta_from_PRE']=r[group]['token_weighted_nll']-results['PRE'][group]['token_weighted_nll']
    finally:
        current=dict(model.named_parameters())
        issues=[]
        for n,p,version,flag,shape in metadata:
            if current.get(n) is not p or p.requires_grad!=flag or tuple(p.shape)!=shape:
                issues.append(n+': metadata')
            if p not in originals and p._version!=version: issues.append(n+': untouched parameter changed')
        with torch.no_grad():
            for p,old in originals.items():
                if not torch.equal(p,old):
                    p.copy_(old); issues.append('factor not restored by trial')
        if ranks!={c['module']:(c['layer'].lora_rank,c['layer'].lora_scaler) for c in candidates}:
            issues.append('rank/scaling')
        require(not issues, 'PRE restoration audit: '+str(issues))
    return results


def execute(plan_path, output, device):
    require(not output.exists(), 'Refusing existing output directory')
    plan=read_json(plan_path)
    require(plan['version']==1 and plan['mode']=='historical_joint_ablation', 'Unsupported plan')
    for path,digest in plan['input_sha256'].items(): verified(path,digest)
    paths,cfg=plan['paths'],plan['config']
    pre=torch.load(paths['pre_adapter'],map_location='cpu',weights_only=True)
    post=torch.load(paths['post_adapter'],map_location='cpu',weights_only=True)
    records,_,_=extract_historical_candidates(pre,post,cfg['direction_function_tolerance'],read_json(paths['historical_audit']))
    policies,selection=make_policies(read_json(paths['results_a']),read_json(paths['results_b']),records,
                                    plan['selection']['random_seed'],30,plan['include_singletons'])
    require(policies==plan['policies'] and selection==plan['selection'], 'Frozen policies changed')
    train_meta,eval_meta=plan['training_metadata'],plan['evaluation_metadata']
    validate_metadata(paths['train_data'],train_meta); validate_metadata(paths['evaluation_data'],eval_meta)
    used=set()
    for path in ('results_a','results_b'):
        for ids in read_json(paths[path])['groups'].values(): used.update(ids)
    validate_isolation(train_meta,eval_meta,plan['calibration_indices'],plan['evaluation_indices'],used,
                       paths['train_data']!=paths['evaluation_data'])
    # Imports and all large allocations only occur in the explicit run command.
    from transformers import AutoTokenizer
    from .model_loading import load_training_model
    from .checkpoint import load_adapter
    from .training_data import ChatTrainingDataset
    tokenizer=AutoTokenizer.from_pretrained(cfg['model_name_or_path'],trust_remote_code=True,local_files_only=True)
    if tokenizer.chat_template is None:
        tokenizer.chat_template=(Path(cfg['model_name_or_path'])/'chat_template.jinja').read_text()
    tokenizer.pad_token=tokenizer.eos_token
    datasets={}
    for group,key,indices in [('calibration','train_data',plan['calibration_indices']),
                              ('evaluation','evaluation_data',plan['evaluation_indices'])]:
        datasets[group]=(ChatTrainingDataset(paths[key],tokenizer,cfg['model_max_length'],dynamic_length=cfg['dynamic_length']),indices)
    torch.manual_seed(cfg['seed'])
    model=load_training_model(cfg['model_name_or_path'],getattr(torch,plan['dtype']),'minicpm5').to(device)
    model.config.use_cache=False
    load_adapter(model,paths['pre_adapter'])
    candidates=bind_historical_candidates(model,pre,records)
    norms=policy_norms(pre,records,policies)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(mode='offline_diagnostic_only',plan_sha256=file_sha256(plan_path),**git_version(),
                input_sha256=plan['input_sha256'],device=device,dtype=plan['dtype'],
                selection=selection,isolation=plan['isolation'],optimizer_created=False,scheduler_created=False)
    try:
        results=evaluate_policies(model,candidates,policies,datasets,cfg['direction_function_tolerance'])
        for key in results: results[key].update(norms[key])
        report.update(results=results,interactions=interactions(results),pre_factors_restored=True)
    except BaseException as error:
        report['error']=repr(error)
        raise
    finally:
        for path,digest in plan['input_sha256'].items(): verified(path,digest)
        with (output/'results.json').open('x') as stream: json.dump(report,stream,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    check=sub.add_parser('preflight',help='CPU only: hashes, masks, group isolation; no model forward')
    for name in ('results_a','results_b','pre_adapter','post_adapter','historical_audit',
                 'train_source','evaluation_data','evaluation_metadata','output_dir'):
        check.add_argument('--'+name,type=Path,required=True)
    check.add_argument('--calibration_indices',type=int,nargs='+',required=True)
    check.add_argument('--evaluation_indices',type=int,nargs='+',required=True)
    check.add_argument('--random_seed',type=int,default=42)
    check.add_argument('--include_singletons',action='store_true',help='Add one-at-a-time FA-union-FB policies (extra forward cost)')
    run=sub.add_parser('run')
    run.add_argument('--plan',type=Path,required=True)
    run.add_argument('--output_dir',type=Path,required=True)
    run.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.command=='preflight':
        require(not args.output_dir.exists(),'Refusing existing output directory')
        torch.set_num_threads(1)
        plan=checked_inputs(args)
        args.output_dir.mkdir(parents=True,exist_ok=False)
        with (args.output_dir/'plan.json').open('x') as stream: json.dump(plan,stream,indent=2,allow_nan=False)
        print(json.dumps(dict(counts=plan['selection']['counts'],isolation=plan['isolation'],
                              policies={k:len(v) for k,v in plan['policies'].items()}),indent=2))
    else:
        execute(args.plan,args.output_dir,args.device)


if __name__=='__main__': main()
