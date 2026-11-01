"""Matched allocation-only rank expansion at historical step100. No gradient fitting."""
import argparse,copy,json,random,time,traceback,shutil
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,set_seed
import global84_temporal_from_zero as z
import global84_experiment as w
import global84_rank_math as m
import global84_temporal_rank_demand as t
ARMS=('base','random','static_C','current_C','static_DE','current_DE')

class SharedDropoutExpansion(nn.Module):
    """Original Dense forward arithmetic; exactly ONE dropout shared by both branches."""
    def __init__(self,layer,k,name,seed):
        super().__init__();self.base_layer=layer
        rows=[]
        for j in range(k):
            g=torch.Generator().manual_seed(m.stream(seed,f'added:{name}:{j}'))
            v=torch.randn(layer.in_features,generator=g);rows.append(v/v.norm())
        self.add_a=nn.Parameter(torch.stack(rows).to(layer.weight.device));self.add_b=nn.Parameter(torch.zeros(layer.out_features,k,device=layer.weight.device))
    def forward(self,x):
        l=self.base_layer;result=F.linear(x,l.weight,l.bias)
        if l.disable_lora or not l.has_lora_weights:raise ValueError('Disabled Dense adapter unsupported')
        dtype=l._get_lora_dtype();dropped=l.lora_dropout(x.to(dtype))
        original=F.linear(F.linear(dropped,l.weight_a.to(dtype)),l.weight_b.to(dtype)).to(result.dtype)
        result=result+l.lora_scaler*original
        added=F.linear(F.linear(dropped.float(),self.add_a),self.add_b).to(result.dtype)
        return result+added

def gradient_contribution(before, after, grad_norm, lr, weight_decay):
    """AdamW decay-only counterfactual, evaluated in the actual factor dtype."""
    decay_only=before.clone().mul_(1-lr*weight_decay)
    contribution=float((after.detach().double()-decay_only.double()).norm())
    w.require(grad_norm>0 and contribution>0,'No measurable gradient contribution beyond weight decay')
    return dict(gradient_norm=grad_norm,decay_only_change_norm=float((decay_only.double()-before.double()).norm()),
                update_beyond_decay_norm=contribution)

def order(seed):
    # DataLoader consumes a base seed before RandomSampler, like the historical entry.
    return [int(x.item()) for x in DataLoader(range(2000),batch_size=1,shuffle=True,generator=torch.Generator().manual_seed(seed))][100:400]

def inspect(args):
    plans=[];states={};allocations={};hashes={}
    initial=torch.load(args.initial_adapter,map_location='cpu',weights_only=True)
    ih=w.shared.sha256(args.initial_adapter)
    w.require(json.loads((args.initial_adapter.parent/'initial_gid.json').read_text())['sha256']==ih,'Initial SHA mismatch')
    for seed in (42,43,44):
        folder=args.source_root/f'seed{seed}';report=json.loads((folder/'report.json').read_text());plan=json.loads((folder/'plan.json').read_text())
        w.require(report['status']=='PASS' and report['seed']==seed and report['mode']=='train','Wrong historical run')
        w.require(report['initial_adapter_sha256']==ih,'Different initial adapter')
        w.check_hashes(plan['input_sha256']);w.check_hashes(plan['code_sha256'],w.ROOT);plans.append(plan)
        file=folder/'adapter_step_100.pt';h=w.shared.sha256(file)
        w.require(json.loads((folder/'state_100.json').read_text())['sha256']==h,'Step100 SHA mismatch')
        states[seed]=torch.load(file,map_location='cpu',weights_only=True)
        allocations[seed]=json.loads((folder/'allocation_by_stage.json').read_text())
        for p in (file,folder/'plan.json',folder/'report.json',folder/'allocation_by_stage.json',folder/'state_100.json'):hashes[str(p.resolve())]=w.shared.sha256(p)
    w.require(plans[0]==plans[1]==plans[2],'Different historical protocols')
    for p in (args.train_data,args.dev_data,args.source_data):w.require(plans[0]['input_sha256'].get(str(p.resolve()))==w.shared.sha256(p),'Data differs from historical run')
    w.require(Path(plans[0]['historical_config']['model_name_or_path']).resolve()==args.model.resolve(),'Wrong base')
    rows=[]
    for r in initial['records']:
        n=r['module'];a,b=initial['factors'][n]['weight_a'],initial['factors'][n]['weight_b']
        rows.append(dict(module=n,d_in=a.shape[1],d_out=b.shape[0],rank=a.shape[0],cost_per_added_rank=a.shape[1]+b.shape[0]))
        for seed,s in states.items():
            w.require(set(s)==set(initial['factors']),'Missing modules')
            for k,v in initial['factors'][n].items():w.require(s[n][k].shape==v.shape and torch.isfinite(s[n][k]).all(),'Malformed snapshot')
    w.require(len(rows)==84,'84 targets required')
    selected={}
    for seed in states:
        a=allocations[seed]
        selected[seed]=dict(base={r['module']:0 for r in rows},random=m.random_allocation(rows,seed),static_C=a['0']['C'],current_C=a['100']['C'],static_DE=a['0']['DE'],current_DE=a['100']['DE'])
        for arm,allocation in selected[seed].items():validate_budget(allocation,rows,arm=='base')
    hashes[str(args.initial_adapter.resolve())]=ih
    return initial,states,selected,rows,hashes,plans[0]

def validate_budget(a,rows,base=False):
    w.require(set(a)=={r['module'] for r in rows} and all(type(k)==int and 0<=k<=4 for k in a.values()),'Invalid allocation')
    for kind in ('q_proj','v_proj'):w.require(sum(a[r['module']] for r in rows if r['module'].endswith(kind))==(0 if base else 4),'Type budget mismatch')
    cost=sum(a[r['module']]*r['cost_per_added_rank'] for r in rows)
    w.require(cost==(0 if base else 25600),'Actual dimension budget mismatch');return cost

def install(model,layers,allocation,seed):
    added={}
    for n,k in allocation.items():
        if k:
            wrap=SharedDropoutExpansion(layers[n],k,n,seed);parent,_,child=n.rpartition('.');setattr(model.get_submodule(parent),child,wrap);added[n]=wrap
    return added

def factors(model,layers,added):
    for p in model.parameters():p.requires_grad_(False)
    ps=[getattr(l,k) for l in layers.values() for k in ('weight_a','weight_b')]+[getattr(l,k) for l in added.values() for k in ('add_a','add_b')]
    w.require(len(ps)==len({id(p) for p in ps}),'Duplicate parameters')
    for p in ps:p.requires_grad_(True)
    return ps

def evaluate(model,ds,meta):
    result=w.shared.evaluate(model,ds,'cuda')
    for row,source in zip(result['samples'],meta):row['source']=source
    result['mean_question_nll']=sum(x['nll'] for x in result['samples'])/len(result['samples'])
    result['by_family']={f:sum(x['nll_sum'] for x in result['samples'] if x['source']['family']==f)/sum(x['effective_tokens'] for x in result['samples'] if x['source']['family']==f) for f in ('GSM','MATH')}
    return result

def run(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False);report=dict(status='RUNNING',mode=args.mode)
    try:
        initial,states,allocations,rows,hashes,plan=inspect(args)
        report.update(inputs=hashes,seed=args.seed,code_sha256={__file__:w.shared.sha256(Path(__file__)),**w.code_hashes()},git_commit=z.w.subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),configuration=vars(args).copy(),fresh_AdamW=True,historical_moments_restored=False)
        report['configuration']={k:str(v) if isinstance(v,Path) else v for k,v in report['configuration'].items()}
        sequence=order(args.seed);w.write(out/'order.json',sequence);report['order_sha256']=w.shared.sha256(out/'order.json')
        report['pairing_note']='Same reconstructed DataLoader permutation and fresh dropout seed across arms; no historical Adam or RNG restoration. Historical files did not store explicit sample indices. No claim of historical dropout-mask continuation.'
        report['runtime_versions']={k:w.importlib.metadata.version(k) for k in ('torch','transformers','numpy')}
        w.write(out/'allocations.json',allocations[args.seed]);report['old_parameters']=initial['actual_parameters']
        w.require(shutil.disk_usage(out).free>2*2**30,'Need 2 GiB output headroom')
        if args.mode=='dry-run':report['status']='PASS';return
        report['gpu']=w.gpu_admission(30);model,base,report['loading']=z.base(args);layers=z.attach(model,initial['ranks'])
        tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=True)
        if tok.chat_template is None:tok.chat_template=(args.model/'chat_template.jinja').read_text()
        tok.pad_token=tok.eos_token
        train=w.shared.ChatTrainingDataset(args.train_data,tok,2048,dynamic_length=True);dev=w.shared.ChatTrainingDataset(args.dev_data,tok,2048,dynamic_length=True)
        w.require(len(train)==2000 and len(dev)==400,'Dataset sizes changed')
        for arm in args.arms:
            folder=out/arm;folder.mkdir();snapshot=states[args.seed]
            w.restore_layers(model,layers,snapshot);scaling={n:l.lora_scaler for n,l in layers.items()}
            insertion={}
            example=w.shared.batch(train[sequence[0]],'cuda')
            for mode in (False,True):
                model.train(mode)
                with torch.random.fork_rng(devices=[0]),torch.no_grad():insertion[mode]=model(**example).logits.detach().clone()
            added=install(model,layers,allocations[args.seed][arm],args.seed)
            for mode in (False,True):
                model.train(mode)
                with torch.random.fork_rng(devices=[0]),torch.no_grad():logits=model(**example).logits
                w.require(torch.equal(logits,insertion[mode]),f'Insertion logits changed, training={mode}')
            w.require(all(torch.equal(getattr(layers[n],k).detach().cpu(),v) for n,s in snapshot.items() for k,v in s.items()),'Insertion changed A/B')
            ps=factors(model,layers,added);added_count=sum(wr.add_a.numel()+wr.add_b.numel() for wr in added.values())
            w.require(added_count==validate_budget(allocations[args.seed][arm],rows,arm=='base'),'New factor count differs')
            w.require(sum(p.numel() for p in ps)==initial['actual_parameters']+added_count,'Trainable count mismatch')
            optimizer=torch.optim.AdamW(ps,lr=5e-5*(1-100/2000),weight_decay=.01)
            set_seed(args.seed);start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            evaluations={};steps=2 if args.mode=='smoke' else 300
            if args.mode=='train':evaluations['0']=evaluate(model,dev,plan['metadata']['dev'])
            early_gradients={};decay_audit={};activity={};all_gradient_max=[0.]*len(ps);start_params={id(p):p.detach().clone() for p in ps}
            for step,index in enumerate(sequence[:steps],1):
                model.train();optimizer.zero_grad(set_to_none=True);lr=5e-5*(1-(100+step-1)/2000)
                for group in optimizer.param_groups:group['lr']=lr
                loss=model(**w.shared.batch(train[index],'cuda')).loss;w.require(torch.isfinite(loss),'Invalid loss');loss.backward()
                w.require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps),'Invalid gradients')
                for j,p in enumerate(ps):all_gradient_max[j]=max(all_gradient_max[j],float(p.grad.norm()))
                for n,wr in added.items():
                    for key in ('add_a','add_b'):
                        v=float(getattr(wr,key).grad.norm());activity[n+'.'+key]=max(activity.get(n+'.'+key,0),v)
                        if step==1 and key=='add_b':w.require(v>0,'New B has no first-step gradient')
                if step<=2:
                    early_gradients[str(step)]={n:{key:float(getattr(wr,key).grad.norm()) for key in ('add_a','add_b')} for n,wr in added.items()}
                before_a={n:wr.add_a.detach().clone() for n,wr in added.items()} if step==2 else {}
                optimizer.step()
                if step==2:
                    decay_audit={n:gradient_contribution(before_a[n],wr.add_a,early_gradients['2'][n]['add_a'],lr,.01) for n,wr in added.items()}
                if step==1:w.require(all(torch.count_nonzero(wr.add_b)>0 for wr in added.values()),'New B failed to update on first step')
                if step==2:w.require(all(not torch.equal(wr.add_a.detach(),start_params[id(wr.add_a)]) for wr in added.values()),'New A failed to update by second step')
                with (folder/'train.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,global_step=100+step,index=index,loss=float(loss.detach()),lr=lr))+'\n')
                if args.mode=='train' and step in (100,200,300):evaluations[str(step)]=evaluate(model,dev,plan['metadata']['dev'])
            w.require(all(v>0 for v in activity.values()) and all(v>0 for v in all_gradient_max),'Inactive factor; decay is not gradient learning')
            w.require(all(not torch.equal(p.detach(),start_params[id(p)]) for p in ps),'Unchanged trainable factor')
            w.require(all(l.lora_scaler==scaling[n] for n,l in layers.items()),'Scaling changed')
            verified=w.shared.verify_disk_weights(base,args.model)
            result=dict(status='PASS',arm=arm,seed=args.seed,evaluations=evaluations,added_parameters=added_count,trainable_parameters=sum(p.numel() for p in ps),activity=activity,first_two_step_gradient_norms=early_gradients,second_step_A_decay_counterfactual=decay_audit,order_sha256=report['order_sha256'],all_factor_max_gradients=all_gradient_max,allocation=allocations[args.seed][arm],
                added_BA_norms={n:float(torch.trace((wr.add_b.detach().double().T@wr.add_b.detach().double())@(wr.add_a.detach().double()@wr.add_a.detach().double().T)).clamp_min(0).sqrt()) for n,wr in added.items()},
                optimizer_state_bytes=sum(v.numel()*v.element_size() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)),seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),base_verified=verified,insertion_train_eval_logits_equal=True)
            w.write(folder/'result.json',result)
            torch.save(dict(old=t.snapshot(layers),added={n:dict(A=wr.add_a.detach().cpu(),B=wr.add_b.detach().cpu()) for n,wr in added.items()}),folder/'adapter.pt')
            del optimizer,ps,added,start_params,insertion
        w.check_hashes(hashes);report['status']='PASS'
    except BaseException as e:report.update(status='FAIL',error=str(e),traceback=traceback.format_exc());raise
    finally:w.write(out/'report.json',report)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['dry-run','smoke','train']);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--initial-adapter',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--train-data',type=Path,required=True);p.add_argument('--dev-data',type=Path,required=True);p.add_argument('--source-data',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seed',type=int,choices=[42,43,44],default=42);p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS));args=p.parse_args();w.require(len(args.arms)==len(set(args.arms)),'Duplicate arms');run(args)
if __name__=='__main__':main()
