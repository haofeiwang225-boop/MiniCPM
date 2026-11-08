"""Pinned input identities; no model loading and no historical adapter reading."""
import json
import unicodedata
from pathlib import Path
from adaptive_lora.data_provenance import build_source_metadata
from .protocol import file_hash, canonical_hash, read_rank_inventory, PLAN_SHA256, TRAIN_SHA256


def normalized(text):return ' '.join(unicodedata.normalize('NFKC',text).split())


def checked(entry):
    path=Path(entry['path']).resolve()
    if file_hash(path)!=entry['sha256']:raise ValueError('Input hash mismatch: '+str(path))
    return path


def model_hashes(model):
    files={p.name:file_hash(p) for p in Path(model).iterdir()
           if p.is_file() and (p.suffix in ('.json','.safetensors','.jinja','.py','.model'))}
    if not any(n.endswith('.safetensors') for n in files) or 'config.json' not in files:
        raise ValueError('Missing local model files')
    return files


def verify_model(model, expected):
    if model_hashes(model)!=expected:raise ValueError('Base/tokenizer file identities changed')


def audit(config):
    required=('rank_plan','historical_config','train','probe','source','gsm8k','math500')
    paths={k:checked(config[k]) for k in required}
    for k in ('gsm8k','math500'):
        if not config[k].get('revision') or not config[k].get('source_url'):raise ValueError('Official test source/revision missing: '+k)
    vendor=Path(__file__).resolve().parents[2]/'third_party/altlora_official'
    provenance=json.loads((vendor/'PROVENANCE.json').read_text())
    for name,digest in provenance['files'].items():
        if file_hash(vendor/name)!=digest:raise ValueError('Official source changed: '+name)
    if config['rank_plan']['sha256']!=PLAN_SHA256:raise ValueError('Untrusted rank plan')
    if config['train']['sha256']!=TRAIN_SHA256:raise ValueError('Untrusted training data')
    ranks=read_rank_inventory(paths['rank_plan'],historical_config=paths['historical_config'])
    verify_model(config['model'],config['base_hashes'])
    train_meta=build_source_metadata(paths['train'],paths['source'])
    probe_meta=build_source_metadata(paths['probe'],paths['source'])
    originals={}
    needed={r['source_line_1based'] for m in (train_meta,probe_meta) for r in m['samples']}
    with paths['source'].open() as f:
        for i,line in enumerate(f,1):
            if i in needed:originals[i]=normalized(json.loads(line)['original_question'])
    train_groups={originals[r['source_line_1based']] for r in train_meta['samples']}
    probe_groups=[originals[r['source_line_1based']] for r in probe_meta['samples']]
    if len(set(probe_groups))!=len(probe_groups) or set(probe_groups)&train_groups:
        raise ValueError('Probe questions duplicate or overlap original training question groups')
    from .benchmark import load_official_samples
    test_sets={k:load_official_samples(paths[k],k) for k in ('gsm8k','math500')}
    test_groups={normalized(r['problem']) for rows in test_sets.values() for r in rows}
    if test_groups&set(probe_groups):raise ValueError('Probe overlaps official test questions')
    # Do not silently train on detected exact original-question contamination.
    if test_groups&train_groups:raise ValueError('Training original questions overlap official tests')
    split=config['probe_splits']
    if set(split)!= {'construction','audit'}:raise ValueError('Require construction/audit measurement splits')
    indices=split['construction']+split['audit']
    if not indices or not all(split.values()) or any(type(i)is not int for i in indices) or sorted(indices)!=list(range(len(probe_groups))):
        raise ValueError('Probe indices must partition every probe sample exactly once')
    return dict(ranks=ranks,train_metadata=train_meta,probe_metadata=probe_meta,
        probe_groups_sha256=canonical_hash(probe_groups),splits=split,
        isolation='exact NFKC-whitespace-normalized original question identity; semantic paraphrase overlap is not proven absent')
