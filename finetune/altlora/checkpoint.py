"""Complete adapter/optimizer/scheduler/phase/order/RNG checkpoints at step boundaries."""
import random
import numpy as np
import torch
from .adapters import adapter_state,restore_factors

def rng_state():
 n=np.random.get_state()
 return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(s):
 random.setstate(s['python']);n=s['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]));torch.set_rng_state(s['torch'])
 if s['cuda']:
  if len(s['cuda'])!=torch.cuda.device_count():raise ValueError('CUDA topology changed')
  torch.cuda.set_rng_state_all(s['cuda'])
def save(path,layers,optimizer,scheduler,phase,step,order,config):
 if path.exists():raise FileExistsError(path)
 if any(p.grad is not None for g in optimizer.param_groups for p in g['params']):raise ValueError('Save only after zero_grad at step boundary')
 torch.save(dict(format='minicpm-official-altlora-v1',adapter=adapter_state(layers),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),phase=phase.state_dict(),step=step,order=order,config=config,rng=rng_state()),path)
def load(path,layers,optimizer,scheduler,phase,order,config):
 s=torch.load(path,map_location='cpu',weights_only=True)
 if s['format']!='minicpm-official-altlora-v1' or s['config']!=config or s['order']!=order:raise ValueError('Resume source/protocol/order mismatch')
 if s['step']!=s['phase']['step_count']:raise ValueError('Checkpoint phase/step mismatch')
 restore_factors(layers,s['adapter']);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler']);phase.load_state_dict(s['phase']);restore_rng(s['rng']);return s['step']
