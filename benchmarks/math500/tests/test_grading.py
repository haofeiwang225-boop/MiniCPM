"""Regression tests for answer extraction and conservative equivalence."""
import unittest
from benchmarks.math500.answer_parser import parse_answer, extract_answer
from benchmarks.math500.grader import _grade, grade_answer


class GradingTests(unittest.TestCase):
    def test_equivalence(self):
        cases = [
            (r'(3,\tfrac{\pi}{2})', r'\left(3,\frac{\pi}{2}\right)', True),
            ('Evelyn.', r'\text{Evelyn}', True),
            ('90', r'90^\circ', True), ('42', '43', False),
            (r'\text{no}', r'\text{on}', False),
            (r'\text{not equal}', r'\text{equal not}', False),
            (r'\text{a b}', r'\text{ab}', False),
            (r'\{1,2\}', r'\{2,1\}', True),
            (r'\{1,2\}', r'\{1,3\}', False),
            ('(1,2)', '(2,1)', False),
            ('x=1', '1=x', True), ('2*x=2', 'x=1', True),
            ('x*(x-1)=0', 'x-1=0', False),
            (r'\sin(90^\circ)', '1', True),
            (r'\frac12', '0.5', True),
            ('1.2', '12', False), ('5!', '5', False),
            ('1,', '1', False),
        ]
        for prediction, gold, expected in cases:
            with self.subTest(prediction=prediction, gold=gold):
                self.assertEqual(_grade(prediction, gold)['correct'], expected)

    def test_final_response(self):
        self.assertEqual(extract_answer(r'Work: \boxed{2}</think>The answer is: 3'), '3')
        self.assertEqual(extract_answer('The answer is: 2\nCorrection. The answer is: 3'), '3')
        self.assertEqual(extract_answer(r'\boxed{2} Correction. Answer: 3'), '3')
        self.assertIsNone(extract_answer(r'Work: \boxed{2}', thinking_expected=True))
        self.assertIsNone(extract_answer(r'<think>\boxed{2}'))
        self.assertIsNone(extract_answer(r'</think>\boxed{2}\boxed{3'))
        self.assertEqual(parse_answer('</think>')['reason'], 'empty_final_response')
        self.assertEqual(extract_answer('</think>\\[\n\\boxed{\\frac{1}{2}}\n\\]<|im_end|>'), r'\frac{1}{2}')

    def test_worker(self):
        self.assertTrue(grade_answer(r'\frac12', '0.5')['correct'])
        self.assertEqual(grade_answer('1', '1', timeout_seconds=0.001)['reason'], 'timeout')
        self.assertEqual(grade_answer(None, '1')['reason'], 'missing_answer')
        self.assertEqual(_grade(r'\unsupported', r'\unsupported')['reason'], 'string_fallback')

    def test_math_formats(self):
        for a, b, expected in [
            (r'11\sqrt2', r'11\sqrt{2}', True),
            (r'\sqrt[3]8', '2', True),
            (r'\sqrt[3]{8}', '2', True),
            ('5', 'x=5', True), ('7', 'y=7', True),
            ('5', 'x=6', False), ('5', '2*x=10', False),
            ('5', 'x=y', False), ('x=5', 'y=5', False),
            (r'\begin{pmatrix}-\frac13\\[2pt]\frac23\\\frac53\end{pmatrix}',
             r'\begin{pmatrix}-1/3\\2/3\\5/3\end{pmatrix}', True),
            (r'\begin{pmatrix}1&2\end{pmatrix}', r'\begin{pmatrix}1\\2\end{pmatrix}', False),
            (r'\begin{pmatrix}1&3\end{pmatrix}', r'\begin{pmatrix}1&2\end{pmatrix}', False),
        ]:
            with self.subTest(a=a, b=b):
                self.assertEqual(_grade(a, b)['correct'], expected)

    def test_problem_context(self):
        self.assertTrue(_grade('1,-2', '-2,1', 'Find all integers that solve the equation.')['correct'])
        self.assertFalse(_grade('1,-2', '-2,1', 'Find the ordered pair.')['correct'])
        self.assertFalse(_grade('(1,-2)', '(-2,1)', 'Find all values of the coordinates.')['correct'])
        self.assertTrue(_grade(r'10\%', '10', 'By what percent does the area decrease?')['correct'])
        self.assertFalse(_grade(r'10\%', '10', 'Find the number.')['correct'])
        self.assertTrue(_grade(r'10\%', '0.1')['correct'])
        self.assertTrue(grade_answer('7', 't=7')['correct'])

    def test_question_choices(self):
        problem = 'Which direction (north, south, east or west) is she facing?'
        for response, expected in [
            ('She will be facing **east**. (After the rotation.)', 'east'),
            ('She will be facing **west**.', 'west'),
        ]:
            self.assertEqual(parse_answer('</think>' + response, problem=problem)['answer'], expected)
        for response in ['She is not facing east.', 'She might be facing east.',
                         'She is facing east or west.', 'If she is facing east, continue.']:
            self.assertNotEqual(parse_answer('</think>' + response, problem=problem)['reason'],
                                'explicit_question_choice')
        self.assertIsNone(parse_answer('She is facing east.', thinking_expected=True, problem=problem)['answer'])

    def test_intervals_and_infinity(self):
        cases = [
            (r'(2,\infty)', r'(2,\infty)', True),
            (r'(-\infty,0]', r'(-\infty, 0]', True),
            (r'(\frac35,\frac83]', r'(0.6,\frac83]', True),
            (r'(0,9)\cup(9,36)', r'(9,36)\cup(0,9)', True),
            (r'x\in [-2,7]', '[-2,7]', True),
            ('[0,1]', '(0,1)', False),
            ('(0,1]', '[0,1)', False),
            (r'(0,1)\cup(1,2)', '(0,2)', False),
            (r'(-\infty,0)', r'(0,\infty)', False),
            ('[2,1]', '[1,2]', False),
            ('(1,2)', '(2,1)', False),
        ]
        for a, b, expected in cases:
            with self.subTest(a=a, b=b):
                self.assertEqual(_grade(a, b, 'Find the range in interval notation.')['correct'], expected)
        self.assertTrue(_grade(r'(1,\infty)', r'(1,\infty)')['correct'])

    def test_units_and_number_annotations(self):
        cases = [
            ('864', r'864\mbox{ inches}^2', 'Find the area in square inches.', True),
            (r'36\text{ seconds}', '36', 'After how many seconds?', True),
            (r'12\pi\text{ inches per second}', r'12\pi', 'Find the speed in inches per second.', True),
            ('10 meters', '10', 'How many meters long?', True),
            ('10 centimeters', '10', 'How many meters long?', False),
            ('10 cm', '10 m', 'Find the length.', False),
            ('10 inches^2', '10 inches', 'Find the measurement.', False),
            ('10 cats', '10', 'How many cats?', False),
            ('2m', '2', 'Simplify the expression.', False),
            (r'\$36', '36', 'How many dollars?', True),
            (r'12^{\text{th}}\text{ grade}', '12', 'For which grade?', True),
            (r'10,\!080', '10080', 'In how many ways?', True),
            ('58,500', '58{,}500', 'How many plates?', True),
            ('1,234', '1234', 'Find all integer roots.', False),
            ('1,23', '123', 'Find the number.', False),
            ('(1,234)', '(1234)', 'Find the ordered pair.', False),
            ('-2,1', r'-2,\;1', 'Find all the integer roots, separated by commas.', True),
        ]
        for a, b, problem, expected in cases:
            with self.subTest(a=a, b=b):
                self.assertEqual(_grade(a, b, problem)['correct'], expected)

    def test_answer_labels_and_matrix_chain(self):
        problem = 'Which equation? (A) first (B) second (C) third'
        self.assertTrue(_grade('C', r'\text{(C)}', problem)['correct'])
        self.assertTrue(_grade(r'\text{(B)}\;3x=7', r'\text{(B)}', problem)['correct'])
        self.assertFalse(_grade('A', r'\text{(C)}', problem)['correct'])
        self.assertFalse(_grade('(B) or (C)', '(C)', problem)['correct'])
        self.assertFalse(_grade('not (C)', '(C)', problem)['correct'])
        self.assertTrue(_grade(r'\sin T=\frac{\sqrt{21}}5', r'\frac{\sqrt{21}}5', r'What is $\sin{T}$?')['correct'])
        self.assertFalse(_grade(r'\cos T=\frac{\sqrt{21}}5', r'\frac{\sqrt{21}}5', r'What is $\sin{T}$?')['correct'])
        self.assertTrue(_grade(r'2\begin{pmatrix}1\\2\end{pmatrix}=\begin{pmatrix}2\\4\end{pmatrix}',
                               r'\begin{pmatrix}2\\4\end{pmatrix}')['correct'])
        self.assertFalse(_grade(r'2\begin{pmatrix}1\\2\end{pmatrix}=\begin{pmatrix}2\\5\end{pmatrix}',
                                r'\begin{pmatrix}2\\5\end{pmatrix}')['correct'])

    def test_explained_final_answers(self):
        self.assertEqual(parse_answer(r'</think>\[\sqrt{53}\] Since 53 is prime.')['answer'], r'\sqrt{53}')
        self.assertNotEqual(parse_answer(r'</think>\[2\] However, the answer is 3.')['answer'], '2')
        problem = 'Alice and Bob compare wages. Who earns the most?'
        self.assertEqual(parse_answer('</think>Alice would earn the most.**', problem=problem)['answer'], 'Alice')
        self.assertNotEqual(parse_answer('</think>Alice would not earn the most.', problem=problem)['reason'], 'explicit_person_answer')
        self.assertNotEqual(parse_answer('</think>Carol would earn the most.', problem=problem)['reason'], 'explicit_person_answer')


if __name__ == '__main__':
    unittest.main()
