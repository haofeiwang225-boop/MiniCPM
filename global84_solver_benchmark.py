#!/usr/bin/env python3
"""CPU constructed-spectrum benchmark, no model/GPU/data and no allocation artifact."""
import argparse
import json
from pathlib import Path
import time
import traceback
import torch
import global84_rank_math as methods
from global84_rank_direction import require


def benchmark_shape(d_out,d_in=2048):
    """Dense real-size gradients with known low-rank rotated spectral solutions."""
    rng=torch.Generator(device='cpu').manual_seed(20261027)
    u=torch.linalg.qr(torch.randn(d_out,12,dtype=torch.float64,generator=rng),mode='reduced')[0]
    v=torch.linalg.qr(torch.randn(d_in,12,dtype=torch.float64,generator=rng),mode='reduced')[0]
    singular=torch.arange(12,0,-1,dtype=torch.float64)
    aligned=singular*.8
    gc=(u*singular)@v.T; ga=(u*aligned)@v.T
    # Input side removes index 0; output side removes index 1.
    a=v[:,:1].T; b=u[:,1:2]
    started=time.perf_counter()
    frames,solver=methods.candidate_frames(a,b,gc,ga,4,42)
    expected={
        'residual_svd':singular[2:6].square(),
        'projected_full_svd':singular[1:5].square(),
        'cross_alignment':(singular*aligned)[1:5]}
    reference_directions={'residual_svd':v[:,2:6],
        'projected_full_svd':v[:,1:5],'cross_alignment':v[:,1:5]}
    for family,values in expected.items():
        measured=torch.tensor(solver[family]['eigenvalues'],dtype=torch.float64)
        error=float((measured-values).abs().max())
        # Compare spans via small Gram; do not construct d_in x d_in projector.
        q=frames[family]; ref=reference_directions[family]
        span_error=abs(4-float((ref.T@q).square().sum()))
        require(torch.allclose(measured,values,rtol=1e-8,atol=1e-8),f'{family} eigenvalue reference mismatch')
        require(span_error<1e-7,f'{family} reference eigenspace mismatch')
        solver[family].update(reference_eigenvalues=values.tolist(),max_reference_eigenvalue_error=error,
            reference_span_error=span_error)
    return dict(shape=[d_out,d_in],status='PASS',solver=solver,seconds=time.perf_counter()-started,
        known_input_gradient_bytes=(gc.numel()+ga.numel())*8,
        reference='Analytic rank-12 orthogonally rotated matrices; no large Gram or full SVD reference',
        caveat='Known low-rank convergence is not a guarantee for real noisy checkpoint gradients')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',cpu_only=True,allocation_eligible=False,shapes=[])
    try:
        for dout in (2048,256):
            row=benchmark_shape(dout); report['shapes'].append(row)
            print(json.dumps(row,indent=2),flush=True)
        report['status']='PASS'
    except BaseException as e:
        report.update(status='FAIL',error=str(e),traceback=traceback.format_exc()); raise
    finally:
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
