"""CPU protocol-v2 regression tests, including actual q/v matrix dimensions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import global84_experiment as w
import global84_rank_math as m
import global84_statistics as stats
from global84_solver_benchmark import benchmark_shape
from test_minicpm_single_layer_rank_gain import TinyLM, examples


class RevisionTests(unittest.TestCase):
    def test_actual_q_dimensions_reference(self):
        result=benchmark_shape(2048)
        self.assertEqual(result['shape'],[2048,2048])
        self.assertTrue(all(x['converged'] for x in result['solver'].values()))

    def test_actual_v_dimensions_reference(self):
        result=benchmark_shape(256)
        self.assertEqual(result['shape'],[256,2048])
        self.assertTrue(all(x['orthogonality_error']<1e-7 for x in result['solver'].values()))

    def test_independent_alignment_protocol(self):
        self.assertEqual(w.ROLES,('construction','alignment','selection','audit'))
        self.assertEqual(sum(w.PROTOCOL[r+'_samples'] for r in w.ROLES),128)
        # Selection changes scores, not the already fitted frame.
        torch.manual_seed(123)
        a=torch.randn(2,12); b=torch.randn(8,2)
        gc=torch.randn(8,12,dtype=torch.float64); alignment=torch.randn_like(gc)
        frames,_=m.candidate_frames(a,b,gc,alignment,4,42,max_basis=12)
        original=frames['cross_alignment'].clone()
        p1,_=m.proxy_prefixes(gc,[gc]*4,[1]*4,original,4096)
        p2,_=m.proxy_prefixes(gc,[-gc]*4,[1]*4,original,4096)
        self.assertTrue(torch.equal(original,frames['cross_alignment']))
        self.assertNotEqual(p1,p2)

    def test_real_analysis_path_selection_and_audit_cannot_refit(self):
        torch.manual_seed(19)
        name='toy.q_proj'; row=dict(module=name,d_in=12,d_out=8,cost_per_added_rank=4096)
        payload=dict(layers={name:dict(state=dict(weight_a=torch.randn(2,12),weight_b=torch.randn(8,2)))})
        plan=dict(common=dict(tokens={r:[2]*32 for r in w.ROLES}))
        blocks={r:[torch.randn(8,12,dtype=torch.float64) for _ in range(4)] for r in w.ROLES}
        def get(folder,artifacts,role,index,shape,device): return blocks[role]
        with patch.object(w,'load_gradient_blocks',side_effect=get):
            frames,first=w.analyze_module(None,None,plan,payload,0,row,42,'cpu')
            blocks['selection']=[-100*g for g in blocks['selection']]
            frames2,second=w.analyze_module(None,None,plan,payload,0,row,42,'cpu')
            for f in frames: self.assertTrue(torch.equal(frames[f],frames2[f]))
            self.assertNotEqual(first['DE_prefix_options'],second['DE_prefix_options'])
            blocks['audit']=[-200*g for g in blocks['audit']]
            frames3,third=w.analyze_module(None,None,plan,payload,0,row,42,'cpu')
            self.assertEqual(second['DE_prefix_options'],third['DE_prefix_options'])
            self.assertEqual(second['C_prefix_options'],third['C_prefix_options'])
            for f in frames: self.assertTrue(torch.equal(frames[f],frames3[f]))
            blocks['alignment']=[-3*g for g in blocks['alignment']]
            frames4,_=w.analyze_module(None,None,plan,payload,0,row,42,'cpu')
            self.assertTrue(torch.equal(frames['residual_svd'],frames4['residual_svd']))
            self.assertFalse(torch.equal(frames['cross_alignment'],frames4['cross_alignment']))

    def test_cached_module_requires_hash_and_convergence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'module_cache').mkdir(); (root/'directions').mkdir()
            frame=root/'directions/00.pt'; torch.save({'random':torch.eye(4)},frame)
            record=root/'module_cache/00.record.json'
            w.write(record,dict(solver={f:dict(converged=True,relative_ritz_residuals=[0.]*4,tolerance=1e-4) for f in m.FAMILIES if f!='random'}))
            marker=root/'module_cache/00.json'
            def save(): w.write(marker,dict(status='PASS',key='key',files={str(p.relative_to(root)):w.shared.sha256(p) for p in (record,frame)}))
            save(); self.assertIsNotNone(w.reuse_analysis(root,0,'key'))
            self.assertIsNone(w.reuse_analysis(root,1,'key'))
            with self.assertRaisesRegex(ValueError,'mismatch'): w.reuse_analysis(root,0,'other')
            frame.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'changed'): w.reuse_analysis(root,0,'key')
            save(); r=json.loads(record.read_text()); r['solver']['cross_alignment']['converged']=False
            w.write(record,r); save()
            with self.assertRaisesRegex(ValueError,'Unconverged'): w.reuse_analysis(root,0,'key')

    def test_cpu_full_allocation_benchmark_and_reuse(self):
        # Real 84-module/type budget contract, small synthetic gradient tensors.
        # Real matrix dimensions are exercised separately above.
        from test_global84_experiment import inventory
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'probe'; source.mkdir()
            rows=[dict(r,d_in=12,d_out=8) for r in inventory()]
            a=torch.eye(12)[:2]; b=torch.eye(8)[:,:2]
            payload=dict(layers={r['module']:dict(state=dict(weight_a=a,weight_b=b)) for r in rows})
            adapter=root/'adapter.pt'; torch.save(payload,adapter)
            plan=dict(fingerprint='synthetic-cpu-only',common=dict(inventory=dict(adapter=str(adapter),modules=rows),
                tokens={role:[2]*32 for role in w.ROLES}))
            w.write(source/'report.json',dict(status='PASS',mode='probe',total_backward_calls=128,total_forward_calls=128))
            artifacts={}
            # Preserve distinct role/block values; no 84-layer model or GPU.
            generator=torch.Generator().manual_seed(42)
            for role in w.ROLES:
                for block in range(4):
                    d=source/'gradients'/role/str(block); d.mkdir(parents=True)
                    g=torch.randn(8,12,generator=generator)
                    for i in range(84):
                        f=d/f'{i:02d}.pt'; torch.save(g,f)
                        artifacts[str(f.relative_to(source))]=w.shared.sha256(f)
            w.write(source/'gradient_artifacts.json',artifacts)
            def args(mode,out,reuse=None):
                argv=[mode,'--probe-dir',str(source),'--output-dir',str(out)]
                if reuse: argv+=['--reuse-dir',str(reuse)]
                if mode=='benchmark': argv+=['--module',rows[0]['module']]
                out.mkdir(); return w.parser().parse_args(argv)
            first=root/'first'; second=root/'second'; bench=root/'bench'
            with patch.object(w,'load_plan',return_value=plan),patch('builtins.print'):
                report={}; w.allocate(args('allocate',first),report)
                self.assertEqual(report['status'],'PASS')
                allocation=json.loads((first/'allocation.json').read_text())
                self.assertEqual(allocation['allocations']['D'],allocation['allocations']['E'])
                for arm in 'BCDE':
                    self.assertEqual(sum(allocation['allocations'][arm][r['module']]*r['cost_per_added_rank'] for r in rows),25600)
                with patch.object(w,'analyze_module',side_effect=AssertionError('Must reuse completed module')):
                    report={}; w.allocate(args('allocate',second,first),report)
                    self.assertEqual(len(report['reused_modules']),84)
                    report={}; w.allocate(args('benchmark',bench,first),report,benchmark=True)
                    self.assertTrue(report['cpu_only']); self.assertFalse(report['allocation_eligible'])
                self.assertEqual(json.loads((second/'allocation.json').read_text())['allocations'],allocation['allocations'])

    def test_smoke_cannot_allocate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); w.write(root/'report.json',dict(status='PASS',mode='gpu-smoke'))
            args=w.parser().parse_args(['allocate','--probe-dir',folder,'--output-dir',str(root/'out')])
            with patch.object(w,'load_plan',return_value={}):
                with self.assertRaisesRegex(ValueError,'ineligible'): w.allocate(args,{})

    def test_inactive_factors_fail_without_manufactured_updates(self):
        model=TinyLM(); initial=copy.deepcopy(model.state_dict())
        # Zero real loss gradient; AdamW decay alone must not masquerade as activity.
        original=model.forward
        def zero(*args,**kwargs):
            out=original(*args,**kwargs); out.loss=out.loss*0; return out
        with patch.object(model,'forward',side_effect=zero):
            with self.assertRaisesRegex(ValueError,'Inactive factor'):
                w.train_adapters(model,{'target':model.target},{},examples(),[0,1],'cpu')
        self.assertTrue(torch.equal(model.target.weight,initial['target.weight']))

    def test_meaningful_effect_and_multiple_testing(self):
        n=np.array([1.,2.,3.,4.]); x=np.ones((1,3,5,4))*n
        meta=[dict(original_question_sha256=str(i),family='GSM' if i<2 else 'MATH') for i in range(4)]
        x[:,:,4,:]-=.0001*n
        r=stats.analyze(x,n,meta,repeats=80)
        self.assertEqual(r['scientific_interpretation'],'主要比较的收益证据不足')
        self.assertEqual(r['statistical_protocol']['minimum_meaningful_improvement'],w.PROTOCOL['minimum_meaningful_nll_improvement'])
        self.assertIsNotNone(r['overall']['comparisons']['E-D']['primary_ci97_5_bonferroni'])
        self.assertIsNone(r['overall']['comparisons']['D-B']['primary_ci97_5_bonferroni'])
        x[:,:,4,:]-=.0005*n
        r=stats.analyze(x,n,meta,repeats=80)
        self.assertIn('观察到改善',r['scientific_interpretation'])


if __name__=='__main__': unittest.main()
