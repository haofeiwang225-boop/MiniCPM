"""Plain dense forward shared by vanilla and AltLoRA; FP32 factors, frozen base.
Arithmetic follows historical LinearWithLoRA: cast delta, scale, then add.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class DenseAdapter(nn.Module):
 def __init__(self,base,rank=2,alpha=4.,dropout=.1):
  super().__init__()
  if not isinstance(base,nn.Linear) or not 0<rank<=min(base.in_features,base.out_features):raise ValueError('Invalid dense base/rank')
  if not math.isfinite(alpha) or alpha<=0 or not 0<=dropout<1:raise ValueError('Invalid alpha/dropout')
  self.base_layer=base;self.rank=rank;self.scaling=float(alpha/rank);self.dropout=nn.Dropout(dropout)
  self.weight_a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
  self.weight_b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
  nn.init.kaiming_uniform_(self.weight_a,a=math.sqrt(5))
  for p in base.parameters():p.requires_grad_(False)
 @property
 def weight(self):return self.base_layer.weight
 @property
 def bias(self):return self.base_layer.bias
 def forward(self,x):
  result=self.base_layer(x)
  delta=F.linear(F.linear(self.dropout(x.to(self.weight_a.dtype)),self.weight_a),self.weight_b).to(result.dtype)
  return result + self.scaling * delta
 def spec(self):return dict(rank=self.rank,scaling=self.scaling,dropout=self.dropout.p,A_shape=list(self.weight_a.shape),B_shape=list(self.weight_b.shape))

def attach(model,rank=2,alpha=4.,dropout=.1,adapter=None,require_84=False):
 targets={n:m for n,m in model.named_modules() if isinstance(m,nn.Linear) and n.rsplit('.',1)[-1] in ('q_proj','v_proj')}
 if require_84 and set(targets)!={f'model.layers.{i}.self_attn.{kind}_proj' for i in range(42) for kind in ('q','v')}:raise ValueError('Expected all84 MiniCPM q/v projections')
 if not targets:raise ValueError('No q/v targets')
 if adapter is not None and set(adapter)!=set(targets):raise ValueError('Adapter target mismatch')
 for p in model.parameters():p.requires_grad_(False)
 layers={}
 for n,base in targets.items():
  saved=adapter[n] if adapter is not None else None
  if saved and saved.get('structure')!='dense':raise ValueError('Only original Dense adapters supported')
  if saved:
   state=saved['state'];config=saved['config']
   if set(state)!={'weight_a','weight_b'}:raise ValueError('Expanded/block adapters not supported')
   r=config['lora_rank'];scale=saved['scaling'];dp=config['lora_dropout']
  else:r,scale,dp=rank,alpha/rank,dropout
  layer=DenseAdapter(base,r,scale*r,dp)
  if saved:
   for k,p in [('weight_a',layer.weight_a),('weight_b',layer.weight_b)]:
    v=state[k]
    if v.shape!=p.shape or not torch.isfinite(v).all():raise ValueError('Invalid saved factors '+n)
    with torch.no_grad():p.copy_(v)
  parent,_,child=n.rpartition('.');setattr(model.get_submodule(parent) if parent else model,child,layer);layers[n]=layer
 return layers

def factor_pairs(layers):return [(n,m.weight_a,m.weight_b) for n,m in layers.items()]
def adapter_state(layers):return {n:dict(spec=m.spec(),A=m.weight_a.detach().cpu().clone(),B=m.weight_b.detach().cpu().clone()) for n,m in layers.items()}
def restore_factors(layers,saved):
 if set(layers)!=set(saved):raise ValueError('Checkpoint target mismatch')
 for n,m in layers.items():
  if m.spec()!=saved[n]['spec']:raise ValueError('Checkpoint structure/scaling mismatch '+n)
  for k,p in [('A',m.weight_a),('B',m.weight_b)]:
   v=saved[n][k]
   if p.shape!=v.shape or not torch.isfinite(v).all():raise ValueError('Invalid factor '+n)
 with torch.no_grad():
  for n,m in layers.items():m.weight_a.copy_(saved[n]['A']);m.weight_b.copy_(saved[n]['B'])

def export_adapter(layers,base_model):
 """Legacy Dense factor container; optimizer/RNG lives only in full checkpoint.
 Existing legacy Dense loaders use the same cast/scale/add ordering.
 """
 return dict(format_version=1,base_model=str(base_model),forward_semantics='historical_lora_cast_scale_add',layers={n:dict(
  structure='dense',scaling=m.scaling,
  config=dict(in_features=m.weight_a.shape[1],out_features=m.weight_b.shape[0],bias=m.bias is not None,lora_rank=m.rank,lora_scaler=m.scaling*m.rank,lora_dropout=m.dropout.p,weight_a_init_method='kaiming',weight_b_init_method=None,run_lora_in_fp32=True,quant=False,quant_type='NF4'),
  state=dict(weight_a=m.weight_a.detach().cpu().clone(),weight_b=m.weight_b.detach().cpu().clone())) for n,m in layers.items()})
