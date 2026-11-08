"""Atomic checkpoints, paired per-update RNG, and measurement transactions."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from altlora.checkpoint import rng_state, restore_rng
from altlora.adapters import adapter_state, restore_factors
from .protocol import validate_resume_identity


def atomic_save(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.incomplete-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush(); os.fsync(stream.fileno())
        # Atomic no-overwrite publication: a concurrent writer cannot be replaced.
        os.link(temp, path)
    finally:
        os.unlink(temp)


def write_json(path, payload):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    fd, temp = tempfile.mkstemp(prefix='.incomplete-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        os.unlink(temp)


def rng_fingerprint(state):
    h=hashlib.sha256()
    h.update(repr(state['python']).encode());h.update(repr(state['numpy']).encode())
    for t in [state['torch'], *state['cuda']]: h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()


def set_step_rng(seed, step):
    """Deliberate paired step reseeding, not historical uninterrupted RNG replay.
    This function is invoked AFTER all setup/measurement, immediately pre-forward.
    Identical architecture/forward order is also required for identical masks.
    """
    key=int.from_bytes(hashlib.sha256(f'paired-dropout-v1:{seed}:{step}'.encode()).digest()[:4],'big')
    random.seed(key);np.random.seed(key);torch.manual_seed(key)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(key)
    return rng_state()


@contextlib.contextmanager
def preserved(model=None):
    rng=rng_state()
    modes=[] if model is None else [(m,m.training) for m in model.modules()]
    params=[] if model is None else [(p,p.requires_grad,p.grad) for p in model.parameters()]
    try:yield
    finally:
        for p,flag,grad in params:p.requires_grad_(flag);p.grad=grad
        for m,mode in modes:m.training=mode
        restore_rng(rng)


def save_checkpoint(path,layers,opt,scheduler,phase,step,order,identity,extra=None):
    if any(p.grad is not None for group in opt.param_groups for p in group['params']):
        raise ValueError('Checkpoint requires cleared gradients')
    atomic_save(path,dict(format='from-zero-v2',adapter=adapter_state(layers),
        optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),phase=phase.state_dict(),
        global_step=step,cursor=step,order=order,identity=identity,rng=rng_state(),extra=extra))


def load_checkpoint(path,layers,opt,scheduler,phase,order,identity):
    data=torch.load(path,map_location='cpu',weights_only=True)
    validate_resume_identity(data['identity'],identity)
    step=data['global_step']
    if (data['format']!='from-zero-v2' or data['order']!=order or data['cursor']!=step
        or data['phase']['step_count']!=step or not 0<=step<=2000
        or data['scheduler']['last_epoch']!=step):raise ValueError('Checkpoint cursor/schedule mismatch')
    restore_factors(layers,data['adapter']);opt.load_state_dict(data['optimizer'])
    scheduler.load_state_dict(data['scheduler']);phase.load_state_dict(data['phase'])
    restore_rng(data['rng'])
    return step
