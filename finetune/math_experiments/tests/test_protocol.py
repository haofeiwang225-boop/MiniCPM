import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dev_accuracy import load_dev, build_request, score_response
from compare import paired
from adaptive_lora.data_provenance import save_source_metadata


class ProtocolTests(unittest.TestCase):
    def test_request_is_existing_author_protocol(self):
        request=build_request(dict(served_model_name='math-model',enable_thinking=True,generation=dict(temperature=1.,top_p=.95,n=1)),'2+3?')
        self.assertEqual(set(request),{'model','messages','temperature','top_p','n','chat_template_kwargs'})
        self.assertEqual(len(request['messages']),1)
        self.assertEqual(request['messages'][0]['role'],'user')
        self.assertIn('\\boxed{}',request['messages'][0]['content'])
        self.assertEqual(request['chat_template_kwargs'],{'enable_thinking':True})

    def test_unfinished_never_graded_as_answer(self):
        final,finished,parsed,grade=score_response('The answer is: 5','2+3?','5')
        self.assertFalse(finished);self.assertFalse(grade['correct']);self.assertEqual(final,'')
        self.assertEqual(parsed['reason'],'unfinished_thinking')

    def test_metamath_reference_and_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); rows=[dict(messages=[dict(role='user',content='2+3?'),dict(role='assistant',content='work\n#### 5\nThe answer is: 5')])]
            (d/'dev.json').write_text(json.dumps(rows))
            (d/'src.jsonl').write_text(json.dumps(dict(query='2+3?',response=rows[0]['messages'][1]['content'],type='GSM_AnsAug',original_question='2+3?')))
            save_source_metadata(d/'dev.json',d/'src.jsonl',d/'meta.json')
            samples=load_dev(d/'dev.json',d/'meta.json')
            self.assertEqual(samples[0]['answer'],'5')
            self.assertEqual(samples[0]['family'],'GSM')
            rows[0]['messages'][1]['content']='unmarked 5'
            (d/'dev.json').write_text(json.dumps(rows))
            with self.assertRaises(ValueError):load_dev(d/'dev.json',d/'meta.json')

    def test_paired_counts_and_no_intersection(self):
        base=[dict(question_id=i,sample_sha256=str(i),answer='5',family='GSM',correct=x) for i,x in enumerate([True,False,True,False])]
        geo=[dict(r,correct=x) for r,x in zip(base,[False,True,True,False])]
        stats=paired(base,geo)
        for k in ('rescued','harmed','common_correct','common_wrong'):self.assertEqual(stats[k],1)
        with self.assertRaises(ValueError):paired(base,geo[:-1])
