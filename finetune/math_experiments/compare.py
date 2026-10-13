"""Paired dev accuracy and uncertainty; never tune using MATH-500."""
import argparse
import json
import math
from pathlib import Path
import random
from dev_accuracy import summarize


def paired(left,right,seed=42):
    a={r['question_id']:r for r in left}; b={r['question_id']:r for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or a.keys()!=b.keys():
        raise ValueError('Comparison requires identical unique IDs; no intersection-only scoring')
    for key in a:
        if any(a[key][f]!=b[key][f] for f in ('sample_sha256','answer','family')):
            raise ValueError('Paired data/reference mismatch')
    deltas=[int(b[k]['correct'])-int(a[k]['correct']) for k in a]
    rescued=deltas.count(1); harmed=deltas.count(-1); n=len(a)
    if not n:raise ValueError('Empty comparison')
    rng=random.Random(seed)
    estimates=sorted(sum(rng.choices(deltas,k=n))/n for _ in range(10000))
    discordant=rescued+harmed
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(rescued,harmed)+1))/2**discordant) if discordant else 1.
    return dict(rescued=rescued,harmed=harmed,common_correct=sum(a[k]['correct'] and b[k]['correct'] for k in a),
                common_wrong=sum(not a[k]['correct'] and not b[k]['correct'] for k in a),
                accuracy_delta=sum(deltas)/n,paired_bootstrap95=[estimates[249],estimates[9749]],
                mcnemar_exact_p=p,bootstrap_seed=seed,
                uncertainty_scope='question sampling only; one stochastic generation per model, decoding variability not estimated')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=lambda path:[json.loads(line) for line in path.read_text().splitlines()]
    a,b=rows(args.left),rows(args.right)
    for key in ('dev_sha256','generation','frozen_files'):
        ca=json.loads((args.left.parent/'run_config.json').read_text())
        cb=json.loads((args.right.parent/'run_config.json').read_text())
        if ca[key]!=cb[key]:raise ValueError('Evaluation protocol mismatch: '+key)
    result={f:paired([r for r in a if f=='overall' or r['family']==f],
                     [r for r in b if f=='overall' or r['family']==f]) for f in ('overall','GSM','MATH')}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
