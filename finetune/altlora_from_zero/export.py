"""Separate post-training export for standard HF/vLLM; never PEFT-load custom state."""
import argparse
import json
from pathlib import Path
import torch
from .run import load_model, batch, datasets
from .initialization import attach_pattern
from altlora.adapters import restore_factors
from .protocol import file_hash
from .state import preserved,write_json


def export(root,arm):
    report=json.loads((root/arm/'report.json').read_text())
    if report.get('status')!='PASS' or report.get('steps')!=2000:raise ValueError('Final complete arm required')
    source=root/arm/'final_adapter.pt'
    if file_hash(source)!=report['final_adapter_sha256']:raise ValueError('Final adapter changed')
    config=json.loads((root/'shared/config.json').read_text());model,_=load_model(config)
    pattern=json.loads((root/'shared/rank_map.json').read_text())['pattern']
    layers=attach_pattern(model,pattern);restore_factors(layers,torch.load(source,map_location='cpu',weights_only=True))
    train,_=datasets(config);inputs=batch(train,0)
    with preserved(model),torch.no_grad():
        model.eval();before=model(**inputs).logits.float().cpu()
        # BF16 base + FP32 factors have different rounding when merged. Fail if
        # explicit, fixed tolerance is exceeded; never silently report equivalence.
        for n,m in layers.items():
            merged=m.base_layer
            merged.weight.copy_((merged.weight.float()+m.scaling*(m.weight_b@m.weight_a)).to(merged.weight.dtype))
            parent,_,child=n.rpartition('.');setattr(model.get_submodule(parent),child,merged)
        after=model(**inputs).logits.float().cpu()
    if not torch.allclose(before,after,rtol=.01,atol=.05):raise ValueError('Merged model output exceeds declared BF16 tolerance; no export approved')
    output=root/'exports'/arm;output.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(output,safe_serialization=True)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(config['model'],trust_remote_code=True,local_files_only=True);tok.save_pretrained(output)
    write_json(output/'export_manifest.json',dict(arm=arm,adapter_sha256=file_hash(source),
        files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()},
        merge_max_abs_error=float((after-before).abs().max()),merge_atol=.05,merge_rtol=.01,
        limitation='single training example BF16 output compatibility check; not bitwise equivalent'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=['vanilla','altlora_false','altlora_true'],required=True)
    a=p.parse_args();export(a.root,a.arm)
