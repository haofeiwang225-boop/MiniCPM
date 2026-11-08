"""Naming/rank/LR bridge; updates delegated to unmodified official altlora.step."""
import contextlib,io,copy
from collections import defaultdict
import torch
from .source import official_class,sha256,VENDOR

class _Pairs:
 def __init__(self,pairs):self.pairs=pairs
 def named_parameters(self):
  for n,a,b in self.pairs:
   yield n+'.lora_A.default.weight',a
   yield n+'.lora_B.default.weight',b
 def parameters(self):return (p for _,p in self.named_parameters())

class OfficialAltLoRA(torch.optim.Optimizer):
 def __init__(self,pairs,lr=5e-5,betas=(.9,.999),eps=1e-4,weight_decay=1e-5,reg=1e-5,correct_bias=True):
  pairs=list(pairs)
  if not pairs or reg<=0 or weight_decay<0:raise ValueError('Invalid pairs/reg/decay')
  params=[p for _,a,b in pairs for p in (a,b)]
  if len({id(p) for p in params})!=len(params):raise ValueError('Duplicate factor')
  groups=defaultdict(list)
  for n,a,b in pairs:
   if a.ndim!=2 or b.ndim!=2 or a.shape[0]!=b.shape[1] or a.dtype!=torch.float32 or b.dtype!=torch.float32 or a.device!=b.device:raise ValueError('Official optimizer requires matched FP32 factors')
   groups[a.shape[0]].append((n,a,b))
  defaults=dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,correct_bias=correct_bias)
  super().__init__(params,defaults)
  self.spec=dict(reg=reg,pairs=[(n,list(a.shape),list(b.shape)) for n,a,b in pairs],source_sha256=sha256(VENDOR/'optimizer_new.py'))
  self.delegates=[official_class()(_Pairs(ps),rank=r,reg=reg,**defaults) for r,ps in sorted(groups.items())]
  for d in self.delegates:d.state=self.state
  self.last_diagnostics=''
 @torch.no_grad()
 def step(self,closure=None):
  if closure is not None:raise ValueError('Closure path is not supported')
  if len(self.param_groups)!=1:raise RuntimeError('Only one externally scheduled parameter group supported')
  g=self.param_groups[0]
  for p in g['params']:
   if p.grad is not None and (not p.requires_grad or not torch.isfinite(p.grad).all()):raise RuntimeError('Frozen/nonfinite gradient')
  log=io.StringIO()
  with contextlib.redirect_stdout(log):
   for d in self.delegates:
    # Official code reads defaults rather than param_groups. Bind the actual LR.
    for k in ('lr','betas','eps','weight_decay','correct_bias'):d.defaults[k]=g[k]
    d.step()
  self.last_diagnostics=log.getvalue()
  if 'no inverse' in self.last_diagnostics:raise RuntimeError('Official inverse fallback occurred; run invalid, inspect diagnostics')
  for p in g['params']:
   if not torch.isfinite(p).all():raise RuntimeError('Nonfinite official factor update')
   for v in self.state.get(p,{}).values():
    if torch.is_tensor(v) and not torch.isfinite(v).all():raise RuntimeError('Nonfinite official momentum')
 def state_dict(self):
  state=super().state_dict();state['official_binding']=copy.deepcopy(self.spec);return state
 def load_state_dict(self,state):
  state=copy.deepcopy(state)
  if state.pop('official_binding',None)!=self.spec:raise ValueError('Optimizer rank/order/source/reg mismatch')
  super().load_state_dict(state)
  for d in self.delegates:d.state=self.state

class OfficialSchedule:
 """Exact LogTrainer alt=True cadence for batch1/no accumulation.
 Main script alt=False -> both throughout. Enabled path: both1..9,A10..19,B20..29.
 """
 def __init__(self,pairs,mode='official-main'):
  if mode not in ('official-main','official-trainer'):raise ValueError('Unknown official alternation mode')
  self.pairs=list(pairs);self.mode=mode;self.step_count=0;self.side='both';self.update_A=True
 def prepare(self):
  self.step_count+=1
  if self.mode=='official-trainer' and self.step_count%10==0:
   self.side='A' if self.update_A else 'B';self.update_A=not self.update_A
  self.apply_flags();return self.side
 def apply_flags(self):
  for _,a,b in self.pairs:
   for p,side in [(a,'A'),(b,'B')]:
    p.requires_grad_(self.side in ('both',side))
    if not p.requires_grad:p.grad=None
 def state_dict(self):return dict(mode=self.mode,step_count=self.step_count,side=self.side,update_A=self.update_A)
 def load_state_dict(self,s):
  if s['mode']!=self.mode:raise ValueError('Alternation mode mismatch')
  n=s['step_count'];expected='both' if self.mode=='official-main' or n<10 else ('A' if (n//10)%2 else 'B')
  next_a=True if self.mode=='official-main' else (n//10)%2==0
  if n<0 or s['side']!=expected or s['update_A']!=next_a:raise ValueError('Invalid alternation state')
  self.step_count=n;self.side=s['side'];self.update_A=s['update_A'];self.apply_flags()
