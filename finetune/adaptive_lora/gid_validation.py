"""Independent from-zero mathematical audit primitives. No experiment on import.
No historical adapter input is accepted here. Existing algorithms are not mutated.
"""
import math
import random
import torch
from .subspace_matching import orth
from .tangent_metrics import residual_energy_rank
from .residual_gid import gid_from_singular_values


def spectrum_metrics(g):
    g=g.detach().cpu().double()
    if g.ndim!=2 or not torch.isfinite(g).all():raise ValueError('Expected finite full-weight gradient matrix')
    s=torch.linalg.svdvals(g);total=s.sum();p=s/total if total>0 else torch.zeros_like(s)
    # Exact requested entropy convention, distinct from legacy log(p+eps).
    positive=p[p>0]
    gid=float(torch.exp(-(positive*positive.log()).sum())) if total>0 else 0.
    return dict(singular_values=s.tolist(),singular_distribution=p.tolist(),gid=gid,
        legacy_gid=gid_from_singular_values(s),gid_definition='exp(-sum(p log p)), p=s/sum(s)',
        legacy_difference=gid-gid_from_singular_values(s),k90=residual_energy_rank(s,.90),
        k95=residual_energy_rank(s,.95),gradient_energy=float(g.square().sum()),zero_gradient=bool(total==0))


def coverage_diagnostics(g,a,b):
    g,a,b=(x.detach().cpu().double() for x in (g,a,b))
    if any(x.ndim!=2 or not torch.isfinite(x).all() for x in (g,a,b)):raise ValueError('Nonfinite or nonmatrix input')
    if a.shape[1]!=g.shape[1] or b.shape!=(g.shape[0],a.shape[0]):raise ValueError('Factor/gradient shape mismatch')
    # Keep established numerical-rank threshold; QR after conversion removes FP32 orthogonality error.
    qa,qb=(torch.linalg.qr(orth(x).double(),mode='reduced')[0] for x in (a.T,b))
    ga=(g@qa)@qa.T;bg=qb@(qb.T@g);gt=bg+ga-qb@(qb.T@g@qa)@qa.T;r=g-gt
    energy=float(g.square().sum());rank=a.shape[0];ra,rb=qa.shape[1],qb.shape[1]
    row=dict(rank=rank,numerical_rank_A=ra,numerical_rank_B=rb,gradient_energy=energy,
        residual_energy=float(r.square().sum()),basis_policy='legacy FP32 QR/R-SVD max(shape)*eps32*smax, then FP64 QR')
    if energy==0:return dict(row,status='UNDEFINED_ZERO_GRADIENT',C_A=None,C_B=None,C_T=None,rho=None,E=None)
    s=torch.linalg.svdvals(g)
    def bound(k):return float(s[:min(k,len(s))].square().sum())/energy
    ct=float(gt.square().sum())/energy;rho=row['residual_energy']/energy
    active=bound(ra+rb);nominal=bound(2*rank)
    if abs(ct+rho-1)>1e-8 or ct>active+1e-8:raise ArithmeticError('Projection/spectral-bound inconsistency; no clamping applied')
    return dict(row,status='OK',C_A=float(ga.square().sum())/energy,C_B=float(bg.square().sum())/energy,
        C_T=ct,rho=rho,side_oracle_top_r=bound(rank),initial_single_side_oracle=bound(rank),
        nominal_full_oracle_top_2r=nominal,active_oracle=active,E=ct/active if active else None,
        DirectionGap=active-ct,CapacityGap=1-active,decomposition_error=abs(ct+rho-1),residual_k90=residual_energy_rank(torch.linalg.svdvals(r)))


def budget(rank_map,shapes):
    if set(rank_map)!=set(shapes):raise ValueError('Target set mismatch')
    if any(type(r)is not int or not 1<=r<=min(4,min(shapes[n])) for n,r in rank_map.items()):raise ValueError('Invalid rank')
    return sum(rank_map[n]*sum(shapes[n]) for n in shapes)


def strict_uniform_control(shapes,target):
    valid=[{n:r for n in shapes} for r in range(1,5) if all(r<=min(s) for s in shapes.values())]
    for mapping in valid:
        if budget(mapping,shapes)==target:return mapping
    raise ValueError('Exact uniform integer rank is infeasible at this parameter budget; do not relabel a nonuniform map')


def randomized_histogram_controls(rank_map,shapes,seed=42,count=100):
    """Explicit conditional null: shuffle ranks within identical shapes.
    Preserves GID histogram, not an unconditional random-budget baseline.
    """
    target=budget(rank_map,shapes);rng=random.Random(seed);groups={}
    for n in sorted(shapes):groups.setdefault(tuple(shapes[n]),[]).append(n)
    maps=[];seen=set()
    for _ in range(100000):
        result={}
        for names in groups.values():
            ranks=[rank_map[n] for n in names];rng.shuffle(ranks);result.update(zip(names,ranks))
        key=tuple(sorted(result.items()))
        if key in seen:continue
        if budget(result,shapes)!=target:raise ArithmeticError('Budget changed')
        seen.add(key);maps.append(result)
        if len(maps)==count:return maps
    raise ValueError('Insufficient distinct histogram-preserving random maps')


def global_coverage(rows):
    energy=sum(r['gradient_energy'] for r in rows)
    if energy==0:return dict(status='UNDEFINED_ZERO_GRADIENT',C_T=None,E=None)
    active=sum(r['gradient_energy']*r['active_oracle'] for r in rows if r['status']=='OK')
    covered=sum(r['gradient_energy']*r['C_T'] for r in rows if r['status']=='OK')
    return dict(status='OK',C_T=covered/energy,active_oracle=active/energy,E=covered/active if active else None,
        rho=sum(r['residual_energy'] for r in rows)/energy,scope='target matrices only; gradient-energy weighted')


class FixedReference:
    def __init__(self,gradients):self.__initial={n:g.detach().cpu().clone() for n,g in gradients.items()}
    def get(self):return {n:g.clone() for n,g in self.__initial.items()}


def actual_update(a,b,new_a,new_b,scaling,g_current,g_initial):
    a,b,new_a,new_b,g_current,g_initial=(x.detach().cpu().double() for x in (a,b,new_a,new_b,g_current,g_initial))
    da,db=new_a-a,new_b-b;delta=scaling*(new_b@new_a-b@a)
    first=scaling*(b@da+db@a);cross=scaling*(db@da)
    if not torch.allclose(delta,first+cross,atol=1e-10,rtol=1e-8):raise ArithmeticError('Finite update identity failed')
    def cosine(g):
        denom=float(delta.norm()*g.norm())
        return float((delta*(-g)).sum())/denom if denom else None
    norm2=float(delta.square().sum());energy=float(g_current.square().sum())
    alpha=max(0.,-float((delta*g_current).sum())/norm2) if norm2 else 0.
    return dict(delta=delta,cos_dynamic=cosine(g_current),cos_fixed_step0=cosine(g_initial),
        optimal_nonnegative_scale=alpha,update_approximation_error=float((alpha*delta+g_current).norm())/math.sqrt(energy) if energy else None,
        cross_ratio=float(cross.norm()/first.norm()) if first.norm()>0 else None,
        cross_ratio_status='DEFINED' if first.norm()>0 else 'UNDEFINED_ZERO_FIRST_ORDER',
        predicted_loss_change=float((g_current*delta).sum()))


def random_budget_controls(shapes, target, seed=42, count=100):
    """Uniform draws over feasible integer maps, independent of all GID/spectra.

    Integer completion counts give exact conditional sampling at the requested
    budget. Unique maps are required; impossible requests fail explicitly.
    This is distinct from the histogram-permutation ablation above.
    """
    if type(count) is not int or count < 1 or not shapes:
        raise ValueError('Positive count and nonempty shapes required')
    names=sorted(shapes)
    costs=[sum(shapes[n]) for n in names]
    divisor=math.gcd(*costs)
    if type(target) is not int or target % divisor:
        raise ValueError('Budget is not representable by module costs')
    costs=[c//divisor for c in costs];target_units=target//divisor
    caps=[min(4,min(shapes[n])) for n in names]
    suffix=[{} for _ in range(len(names)+1)];suffix[-1]={0:1}
    for i in range(len(names)-1,-1,-1):
        for remaining,ways in suffix[i+1].items():
            for rank in range(1,caps[i]+1):
                total=remaining+rank*costs[i]
                if total<=target_units:suffix[i][total]=suffix[i].get(total,0)+ways
    if suffix[0].get(target_units,0)<count:
        raise ValueError('Fewer feasible rank maps than requested unique controls')
    rng=random.Random(seed);seen=set();maps=[]
    for _ in range(max(10000,count*1000)):
        remaining=target_units;ranks=[]
        for i in range(len(names)):
            ticket=rng.randrange(suffix[i][remaining])
            for rank in range(1,caps[i]+1):
                ways=suffix[i+1].get(remaining-rank*costs[i],0)
                if ticket<ways:
                    ranks.append(rank);remaining-=rank*costs[i];break
                ticket-=ways
        key=tuple(ranks)
        if key in seen:continue
        seen.add(key);mapping=dict(zip(names,ranks))
        if remaining or budget(mapping,shapes)!=target:raise ArithmeticError('Completion count budget violation')
        maps.append(mapping)
        if len(maps)==count:return maps
    raise RuntimeError('Unique-map sampling limit reached; no partial control set returned')


def allocation_oracle(rank_map, gradients):
    """Energy-weighted nominal top-2r bound on an independent gradient split.
    No allocation or tuning takes place here.
    """
    if set(rank_map)!=set(gradients):raise ValueError('Gradient targets differ')
    total=0.;covered=0.;modules={}
    for name,g in gradients.items():
        info=spectrum_metrics(g);s=torch.tensor(info['singular_values'],dtype=torch.float64)
        energy=info['gradient_energy'];bound_energy=float(s[:2*rank_map[name]].square().sum())
        total+=energy;covered+=bound_energy
        modules[name]=dict(oracle=bound_energy/energy if energy else None,gradient_energy=energy)
    return dict(global_oracle=covered/total if total else None,modules=modules,
                interpretation='nominal spectral upper bound, not measured loss improvement')
