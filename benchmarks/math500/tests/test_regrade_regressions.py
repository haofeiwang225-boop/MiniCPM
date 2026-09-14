"""Formatting equivalence must preserve mathematical structure and finality."""
import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from benchmarks.math500.answer_parser import parse_answer
from benchmarks.math500.grader import _grade, normalize_latex


class RegradeRegressions(unittest.TestCase):
    def test_offline_audit_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'results.jsonl', root / 'regraded'
            rows = [dict(question_id=i, problem='Compute the number.', gold_answer='0.75',
                         model_output=output, enable_thinking=True, correct=correct,
                         parsed_answer=None, parse_reason='old', grading={'reason': 'old'})
                    for i, (output, correct) in enumerate([
                        (r'</think>\boxed{\dfrac34}', False),
                        (r'</think>\boxed{\dfrac43}', True),
                        (r'<think>\boxed{\dfrac34}', False)])]
            original = ''.join(json.dumps(row) + '\n' for row in rows).encode()
            source.write_bytes(original)
            script = Path(__file__).resolve().parents[1] / 'regrade.py'
            command = [sys.executable, str(script), '--input', str(source),
                       '--output_dir', str(target), '--workers', '1']
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(source.read_bytes(), original)
            summary = json.loads((target / 'summary.json').read_text())
            self.assertEqual(summary['false_to_true_count'], 1)
            self.assertEqual(summary['true_to_false_count'], 1)
            self.assertEqual(summary['unfinished_thinking'], 1)
            comparisons = [json.loads(line) for line in (target / 'comparisons.jsonl').read_text().splitlines()]
            self.assertEqual(len(comparisons), 3)
            self.assertEqual(comparisons[2]['after']['parse_reason'], 'unfinished_thinking')
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_formats_and_counterexamples(self):
        cases = [
            (r'\dfrac34', r'\frac{3}{4}', True),
            (r'\tfrac{\sqrt{13}}7', r'\frac{\sqrt{13}}{7}', True),
            (r'\frac\sqrt{13}7', r'\frac{\sqrt{13}}{7}', True),
            (r'\frac{1}{\frac23}', '1.5', True),
            (r'\dfrac34', r'\frac43', False),
            (r'\sqrt21', r'\sqrt{21}', False),
            (r'\sqrt21', r'\sqrt{2}1', True),
            (r'\sqrt[3]8', '2', True),
            (r'\frac{1}{2}\,\!\;\quad', '0.5', True),
            (' West ', r'\text{west}', True),
            ('west', 'east', False), ('X', 'x', False),
            ('a-b', 'a - b', True), ('ab', 'ba', False),
            (r'\text{not equal}', r'\text{equal not}', False),
            ('**7**.', '7', True), ('7!', '7', False),
            ('(a,b)=(2,7)', '(2,7)', True),
            ('(a,b)=(2,7)', '(7,2)', False),
            ('(a,b)=(2,7)', '(b,a)=(2,7)', False),
            ('(a,b)=(2,7,8)', '(2,7,8)', False),
            ('(a,b)=(x,7)', '(x,7)', False),
            ('(a,a)=(2,7)', '(2,7)', False),
            (r'\mathbf{u}=\begin{pmatrix}2\\7\end{pmatrix}',
             r'\begin{pmatrix}2\\7\end{pmatrix}', True),
            (r'\begin{pmatrix}\dfrac14\\[4pt]\dfrac34\end{pmatrix}',
             r'\begin{pmatrix}0.25\\0.75\end{pmatrix}', True),
            (r'\begin{pmatrix}2&7\end{pmatrix}', r'\begin{pmatrix}2\\7\end{pmatrix}', False),
            (r'\begin{pmatrix}2\\7\end{pmatrix}', r'\begin{pmatrix}7\\2\end{pmatrix}', False),
            ('2+5=7', '7', True), ('2+5=8', '8', False),
            ('2*x=10', '5', False), ('x=y', 'y', False),
            ('x y=6', '6', False),
            (r'\text{Side length}=7\sqrt2', r'7\sqrt2', True),
            (r'\Bigl[\frac{\pi^2}8,\frac{3\pi^2}4\Bigr]',
             r'[\frac{\pi^2}{8},\frac{3\pi^2}{4}]', True),
            ('[0,2]', '(0,2)', False),
            ('2,7', '7,2', False),
            ('10 cm', '10 m', False), ('10 cm', '10', False),
            (r'30^\circ', '30', False),
        ]
        for prediction, gold, expected in cases:
            with self.subTest(prediction=prediction, gold=gold):
                self.assertEqual(_grade(prediction, gold)['correct'], expected)
        self.assertEqual(normalize_latex(r'\sqrt21'), r'\sqrt{2}1')

    def test_contextual_roots(self):
        problem = 'Find the real roots, separated by commas.'
        self.assertTrue(_grade(r'2-\sqrt3,2+\sqrt3', r'2\pm\sqrt3', problem)['correct'])
        self.assertFalse(_grade(r'2-\sqrt3,2+\sqrt3,9', r'2\pm\sqrt3', problem)['correct'])
        self.assertFalse(_grade('(2,7)', '(7,2)', problem)['correct'])
        self.assertTrue(_grade(r'30^\circ', '30', 'Find the angle in degrees.')['correct'])

    def test_finality_and_explanation(self):
        for output in [r'\boxed{7}', r'<think>The answer is 7.', r'<think>\boxed{7}']:
            self.assertEqual(parse_answer(output, thinking_expected=True),
                             {'answer': None, 'reason': 'unfinished_thinking'})
        self.assertEqual(parse_answer(r'<think>\boxed{7}</think>\boxed{8}')['answer'], '8')
        self.assertEqual(parse_answer(r'</think>\[v=(2,7)\] Indeed, this satisfies the constraint.')['answer'], 'v=(2,7)')
        self.assertNotEqual(parse_answer(r'</think>\[7\] Indeed, correction: the answer is 8.')['answer'], '7')
        problem = 'Alice and Bob compare scores. Who wins?'
        self.assertEqual(parse_answer('</think>Bob. His score is larger.', problem=problem)['answer'], 'Bob')
        self.assertNotEqual(parse_answer('</think>Carol. Her score is larger.', problem=problem)['answer'], 'Carol')


if __name__ == '__main__':
    unittest.main()
