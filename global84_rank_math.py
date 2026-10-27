"""Matrix-free directions and exact q/v-stratified integer allocation.

All scores are training-side predictive proxies, NOT true post-training gains.
Audit gradients are deliberately excluded from every allocation function.
"""
import hashlib
import math
import random
import time
import resource

import torch
import minicpm_single_layer_rank_gain as shared
from global84_rank_direction import require

FAMILIES=('random','residual_svd','projected_full_svd','cross_alignment')


def stream(seed,name):
    return int(hashlib.sha256(f'global84-v1:{seed}:{name}'.encode()).hexdigest()[:8],16)


def basis(x):
    q=shared.orth(x).double()
    return torch.linalg.qr(q,mode='reduced')[0] if q.shape[1] else q


def project(x,q):
    return x-q@(q.T@x)


def random_frame(q,k,seed):
    require(q.shape[0]-q.shape[1]>=k,'Insufficient orthogonal complement')
    rng=torch.Generator(device='cpu').manual_seed(seed)
    x=torch.randn(q.shape[0],k,dtype=torch.float64,generator=rng).to(q.device)
    x=project(project(x,q),q)
    out,r=torch.linalg.qr(x,mode='reduced')
    require(float(torch.linalg.svdvals(r).min())>1e-12,'Random frame numerically deficient')
    return out


def symmetric_ritz(operator,q,k,seed,max_basis=128,tolerance=1e-4):
    """Full-reorthogonalized Krylov/Rayleigh-Ritz for largest ALGEBRAIC values.

    Only vectors and a <=128x128 projected operator are stored; never forms
    d_in x d_in Gram/projector/identity. Breakdown restarts with a fixed local
    RNG in the remaining complement (handles repeated/zero eigenvalues).
    Explicit failure on nonconvergence, never silently claim an exact optimum.
    """
    n=q.shape[0]; count=min(max_basis,n-q.shape[1])
    require(0<k<=count and tolerance>0,'Invalid eigensolver dimensions')
    rng=torch.Generator(device='cpu').manual_seed(seed)
    vectors=[]; products=[]; next_vector=None
    for _ in range(count):
        def orth(v):
            for _ in range(2):
                v=project(v,q)
                if vectors:
                    old=torch.stack(vectors,dim=1); v=v-old@(old.T@v)
            return v
        v=orth(next_vector) if next_vector is not None else None
        if v is None or float(v.norm())<1e-12:
            v=orth(torch.randn(n,dtype=torch.float64,generator=rng).to(q.device))
        require(float(v.norm())>1e-12,'Krylov complement exhausted unexpectedly')
        v=v/v.norm(); av=project(operator(v),q)
        require(torch.isfinite(av).all(),'Nonfinite operator output')
        vectors.append(v); products.append(av); next_vector=av
    Q=torch.stack(vectors,dim=1); AQ=torch.stack(products,dim=1)
    small=Q.T@AQ; small=(small+small.T)/2
    values,z=torch.linalg.eigh(small); idx=torch.arange(count-1,count-k-1,-1,device=q.device)
    values=values[idx]; V=Q@z[:,idx]; AV=AQ@z[:,idx]
    residuals=(AV-V*values).norm(dim=0)
    scale=max(float(AQ.norm()),1e-30)
    relative=residuals/scale
    require(torch.isfinite(V).all() and float(relative.max())<=tolerance,
            f'Matrix-free eigensolver not converged: {float(relative.max()):.6g} > {tolerance}; no silent fallback')
    require(float((V.T@V-torch.eye(k,device=q.device,dtype=torch.float64)).abs().max())<1e-7,'Nonorthogonal Ritz directions')
    if q.shape[1]: require(float((q.T@V).abs().max())<1e-7,'Ritz direction outside complement')
    # Canonical signs are only for reproducible serialization, not optimization.
    for j in range(k):
        if V[V[:,j].abs().argmax(),j]<0: V[:,j].neg_()
    return V,dict(eigenvalues=values.cpu().tolist(),relative_ritz_residuals=relative.cpu().tolist(),
                  absolute_ritz_residuals=residuals.cpu().tolist(),ritz_normalization=scale,
                  tolerance=tolerance,operator_calls=count,krylov_dimension=count,
                  algorithm='full-reorthogonalized Krylov largest-algebraic Rayleigh-Ritz',
                  dense_gram_constructed=False,converged=True)


def candidate_frames(a,b,gc,alignment_gradient,k,seed,max_basis=128,tolerance=1e-4):
    """Fit using construction and alignment only. No allocation-selection input."""
    qa,qb=basis(a.T),basis(b)
    gc,gs=gc.double(),alignment_gradient.double()
    require(gc.shape==gs.shape==(b.shape[0],a.shape[1]),'Gradient/factor shape mismatch')
    require(torch.isfinite(gc).all() and torch.isfinite(gs).all(),'Nonfinite candidate gradients')
    residual=shared.residual_gradient(gc,qa,qb)
    def gram(x):
        x=project(x,qa); return project(gc.T@(gc@x),qa)
    def residual_gram(x):
        x=project(x,qa); return project(residual.T@(residual@x),qa)
    def alignment(x):
        x=project(x,qa)
        return project((gc.T@(gs@x)+gs.T@(gc@x))*.5,qa)
    frames={'random':random_frame(qa,k,stream(seed,'random'))}; diagnostics={}
    for family,op in [('residual_svd',residual_gram),('projected_full_svd',gram),('cross_alignment',alignment)]:
        started=time.perf_counter()
        frames[family],diagnostics[family]=symmetric_ritz(op,qa,k,stream(seed,family),max_basis,tolerance)
        V=frames[family]
        diagnostics[family].update(seconds=time.perf_counter()-started,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            memory_note='Linux process lifetime peak RSS; not incremental solver allocation',
            orthogonality_error=float((V.T@V-torch.eye(k,device=V.device,dtype=V.dtype)).abs().max()),
            old_A_orthogonality_error=float((qa.T@V).abs().max()) if qa.shape[1] else 0.)
    for name,V in frames.items():
        require(V.shape==(a.shape[1],k),'Wrong candidate count')
        require(float((V.T@V-torch.eye(k,device=V.device,dtype=V.dtype)).abs().max())<1e-6,'Directions not mutually orthogonal')
        require(float((qa.T@V).abs().max())<1e-6,'Directions overlap old A')
    return frames,diagnostics


def weighted_mean(gradients,token_counts):
    require(len(gradients)==len(token_counts)>0 and all(n>0 for n in token_counts),'Invalid gradient block weights')
    return sum(g.double()*(n/sum(token_counts)) for g,n in zip(gradients,token_counts))


def proxy_prefixes(reference,selection_blocks,token_counts,V,cost,penalty=1.):
    """Prefix U and heuristic uncertainty across independent 8-sample blocks.

    Covariance among directions is retained by summing each block's gains
    before computing prefix uncertainty. Directions were fitted on separate
    construction/alignment samples. Four-block SE is heuristic, not a confidence bound.
    """
    require(cost>0 and penalty>=0 and len(selection_blocks)>=2,'Invalid scoring inputs')
    weights=torch.tensor(token_counts,dtype=torch.float64,device=V.device); weights/=weights.sum()
    ref=reference.double()@V
    contributions=torch.stack([(ref*(g.double()@V)).sum(dim=0) for g in selection_blocks])
    require(torch.isfinite(contributions).all(),'Nonfinite gain proxy')
    options=[dict(k=0,cost=0,mean=0.,standard_error=0.,value=0.,per_parameter=0.,marginal_value=0.)]
    for k in range(1,V.shape[1]+1):
        values=contributions[:,:k].sum(dim=1); mean=(weights*values).sum()
        se=(weights.square()*(values-mean).square()).sum()/(1-weights.square().sum())
        se=se.clamp_min(0).sqrt(); value=float(mean-penalty*se)
        options.append(dict(k=k,cost=k*cost,mean=float(mean),standard_error=float(se),value=value,
                            per_parameter=value/(k*cost),marginal_value=value-options[-1]['value']))
    return options,contributions.cpu().tolist()


def exact_stratified_allocation(options,inventory):
    """Exact prefix DP for FOUR ranks within each type, not free q/v knapsack."""
    require(set(options)=={r['module'] for r in inventory},'All modules must compete')
    allocation={r['module']:0 for r in inventory}; reports={}
    for kind,cost in [('q_proj',4096),('v_proj',2304)]:
        rows=[r for r in inventory if r['module'].endswith('.'+kind)]
        require(len(rows)==42 and all(r['cost_per_added_rank']==cost for r in rows),'Unexpected real type counts/costs')
        # rank_count -> (value, tuple of prefix lengths). Lexicographic tie break.
        dp={0:(0.,())}
        for row in rows:
            candidates=options[row['module']]
            require([x['k'] for x in candidates]==list(range(5)),'Need prefix options 0..4 for every module')
            nxt={}
            for used,(value,path) in dp.items():
                for choice in candidates:
                    k=choice['k']; require(choice['cost']==k*cost and math.isfinite(choice['value']),'Invalid prefix cost/value')
                    if used+k>4: continue
                    item=(value+choice['value'],path+(k,)); old=nxt.get(used+k)
                    if old is None or item[0]>old[0] or (item[0]==old[0] and item[1]<old[1]): nxt[used+k]=item
            dp=nxt
        require(4 in dp,'Cannot satisfy exact four-rank type budget')
        value,path=dp[4]
        for row,k in zip(rows,path): allocation[row['module']]=k
        reports[kind]=dict(added_ranks=4,added_parameters=4*cost,total_proxy=value,
                          negative_marginals_forced=sum(sum(c['marginal_value']<0 for c in options[r['module']][1:k+1]) for r,k in zip(rows,path)))
    require(sum(allocation[r['module']]*r['cost_per_added_rank'] for r in inventory)==25600,'Budget mismatch')
    return allocation,dict(strata=reports,added_parameters=25600,scope='q/v-stratified exact prefix allocation, NOT unrestricted global q/v optimum')


def random_allocation(inventory,seed):
    out={r['module']:0 for r in inventory}; rng=random.Random(stream(seed,'random_allocation'))
    for kind in ('q_proj','v_proj'):
        names=[r['module'] for r in inventory if r['module'].endswith('.'+kind)]
        require(len(names)==42,'Expected 42 modules per type')
        for _ in range(4): out[rng.choice(names)]+=1  # with replacement: multi-rank allowed
    return out
