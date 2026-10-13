"""Request and scoring-boundary regression checks; no model or network."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate


class EndpointProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = {'enable_thinking': True, 'served_model_name': 'MiniCPM5-2B',
                       'generation': {'temperature': 1.0, 'top_p': 0.95, 'n': 1}}

    def test_exact_wire_payload(self):
        self.assertEqual(evaluate.build_request(self.config, 'problem'), {
            'model': 'MiniCPM5-2B', 'messages': [{'role': 'user', 'content':
                'problem\nPlease reason step by step, and put your final answer within \\boxed{}.'}],
            'temperature': 1.0, 'top_p': 0.95, 'n': 1,
            'chat_template_kwargs': {'enable_thinking': True}})

    def test_reject_old_generation_controls(self):
        for key in ('top_k', 'max_tokens', 'max_new_tokens', 'stop', 'stop_strings', 'reasoning_effort'):
            config = {**self.config, 'generation': {**self.config['generation'], key: None}}
            with self.assertRaises(ValueError):
                evaluate.build_request(config, 'problem')

    def test_only_closed_final_reaches_parser(self):
        with patch.object(evaluate, 'parse_answer', return_value={'answer': '2', 'reason': 'boxed'}) as parse, \
                patch.object(evaluate, 'grade_answer', return_value={'correct': True}) as grade:
            final, finished, _, _ = evaluate.score_response('reason</think>discard</think>final', 'p', 'g')
            self.assertEqual(final, 'final')
            self.assertTrue(finished)
            parse.assert_called_once_with('final', thinking_expected=False, problem='p')

    def test_unclosed_box_is_not_used(self):
        with patch.object(evaluate, 'parse_answer') as parse:
            final, finished, extraction, grading = evaluate.score_response(r'thinking \boxed{2}', 'p', '2')
            parse.assert_not_called()
            self.assertEqual(final, '')
            self.assertFalse(finished)
            self.assertIsNone(extraction['answer'])
            self.assertFalse(grading['correct'])


if __name__ == '__main__':
    unittest.main()
