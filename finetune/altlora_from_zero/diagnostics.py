"""Fresh dual gradients and separate immutable G0/current coverage. No selection."""
import torch
from global84_rank_direction import collect_weight_gradients_both
from adaptive_lora.gid_validation import coverage_diagnostics, actual_update
from .state import preserved


def gradients(model,layers,batches):
    with preserved(model):return collect_weight_gradients_both(model,layers,batches)


def coverage_rows(layers, current, fixed, step, split):
    rows=[]
    for reference, maps in [('dynamic',current),('fixed_step0',fixed)]:
        for weighting in ('question','token'):
            for n,m in layers.items():
                rows.append(dict(module=n,step=step,probe_split=split,reference=reference,
                    weighting=weighting,**coverage_diagnostics(maps[weighting][n],m.weight_a,m.weight_b)))
    return rows


def update_rows(layers, before, references, fixed, start, gradient_step):
    rows=[]
    for n,m in layers.items():
        for weighting in ('question','token'):
            result=actual_update(*before[n],m.weight_a,m.weight_b,m.scaling,
                                 references[weighting][n],fixed[weighting][n])
            delta=result.pop('delta')
            rows.append(dict(module=n,weighting=weighting,update_start=start,update_end=start+1,
                gradient_step=gradient_step,gradient_is_stale=gradient_step!=start,
                gradient_definition='eval fixed construction set; NOT training-batch/dropout gradient',
                interval_kind='single_optimizer_step_vs_checkpoint_probe',update_norm=float(delta.norm()),**result))
    return rows
