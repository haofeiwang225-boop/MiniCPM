"""Verify custom adapter merge in FP32, then export an ordinary HF model for vLLM."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from transformers import AutoTokenizer
from adaptive_lora.model_loading import load_training_model
from adaptive_lora.checkpoint import load_adapter, merge_for_export
from adaptive_lora.data_provenance import file_sha256, git_version


def verify_merge(model, names, batches, atol=2e-4, rtol=2e-4):
    model.eval()
    with torch.no_grad():
        before = [model(**batch).logits.float().cpu() for batch in batches]
        model = merge_for_export(model, names)
        errors = []
        for batch, reference in zip(batches, before):
            actual = model(**batch).logits.float().cpu()
            torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
            errors.append(float((actual-reference).abs().max()))
    return model, dict(verification_dtype='float32', max_abs_logit_errors=errors, atol=atol, rtol=rtol), before


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='/home/hongchang/whf/models/MiniCPM5-2B')
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if not tokenizer.chat_template:
        tokenizer.chat_template=(Path(args.model)/'chat_template.jinja').read_text()
    model=load_training_model(args.model, torch.float32, 'minicpm5').to(args.device)
    names=load_adapter(model,args.adapter)
    model.config.use_cache=False
    # Fixed synthetic probes, never test-set questions or labels.
    texts=['Compute 2 + 3.', 'Solve x + 4 = 9.', 'Simplify 1/2 + 1/3.']
    batches=[tokenizer(text,return_tensors='pt').to(args.device) for text in texts]
    model, report, reference=verify_merge(model,names,batches)
    model.to(torch.bfloat16)
    with torch.no_grad():
        # BF16 export is rounded; report its drift separately from FP32 merge equality.
        drift=[]
        for batch, ref in zip(batches,reference):
            logits=model(**batch).logits.float().cpu()
            logp=ref.log_softmax(-1); logq=logits.log_softmax(-1)
            kl=float((logp.exp()*(logp-logq)).sum(-1).mean())
            if not torch.isfinite(logits).all() or kl>0.01:
                raise RuntimeError(f'BF16 export drift exceeds diagnostic KL tolerance: {kl}')
            drift.append(dict(max_abs_logit_error=float((ref-logits).abs().max()),mean_kl=kl,
                              argmax_agreement=float((ref.argmax(-1)==logits.argmax(-1)).float().mean())))
    model.config.use_cache=True
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size='4GB')
    tokenizer.save_pretrained(args.output_dir)
    source=Path(args.model)/'chat_template.jinja'
    if source.exists(): shutil.copy2(source,args.output_dir/'chat_template.jinja')
    report.update(git_version(), adapter=str(args.adapter.resolve()),adapter_sha256=file_sha256(args.adapter),
                  model=args.model, bf16_drift=drift, verification_inputs=texts, custom_adapter_not_peft=True)
    (args.output_dir/'merge_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
