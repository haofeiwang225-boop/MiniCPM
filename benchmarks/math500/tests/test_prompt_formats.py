"""General notation equivalences and conservative counterexamples."""
import unittest
from benchmarks.math500.grader import _grade


class PromptFormatTests(unittest.TestCase):
    def test_intervals(self):
        for a, b, expected in [
            (r'2<t\le7', '(2,7]', True),
            (r'7\ge t>2', '(2,7]', True),
            (r'2\leq\alpha<7', '[2,7)', True),
            ('2<t<7', '[2,7]', False),
            ('7<t<2', '(2,7)', False),
            ('2<t>7', '(2,7)', False),
            ('2<x+y<7', '(2,7)', False),
        ]:
            with self.subTest(a=a, b=b):
                self.assertEqual(_grade(a, b)['correct'], expected)

    def test_thousands(self):
        for a, b, expected in [
            (r'27\,600', '27600', True),
            (r'1\,234\,567', '1234567', True),
            (r'27\,60', '2760', False),
            (r'(27,600)', '27600', False),
            ('27 600', '27600', False),
        ]:
            with self.subTest(a=a):
                self.assertEqual(_grade(a, b)['correct'], expected)

    def test_explicit_sets(self):
        for a, b, expected in [
            (r'\{2\pm\sqrt3,5\}', r'\{5,2-\sqrt3,2+\sqrt3\}', True),
            (r'\{2\pm\sqrt3,5\}', r'\{5,2+\sqrt3\}', False),
            (r'\{2\pm\sqrt3,5\}', r'\{5,2-\sqrt3,2+\sqrt3,9\}', False),
            ('(2,5)', '(5,2)', False),
        ]:
            with self.subTest(a=a, b=b):
                self.assertEqual(_grade(a, b)['correct'], expected)

    def test_vector_coordinates(self):
        for a, b, context, expected in [
            ('(2,5)', r'\begin{pmatrix}2\\5\end{pmatrix}', 'Find the vector.', True),
            ('(5,2)', r'\begin{pmatrix}2\\5\end{pmatrix}', 'Find the vector.', False),
            ('(2,5)', r'\begin{pmatrix}2&5\end{pmatrix}', 'Find the vector.', False),
            ('(2,5,7)', r'\begin{pmatrix}2\\5\end{pmatrix}', 'Find the vector.', False),
            ('(2,5)', r'\begin{pmatrix}2\\5\end{pmatrix}', 'Find an ordered pair.', False),
        ]:
            with self.subTest(a=a, b=b, context=context):
                self.assertEqual(_grade(a, b, context)['correct'], expected)

    def test_contextual_length_units(self):
        for a, b, context, expected in [
            ('8 cm', '8', 'A shape has perimeter 32 cm. How long is each side?', True),
            ('8 m', '8', 'A shape has perimeter 32 cm. How long is each side?', False),
            ('8 cm', '8', 'Lengths are 32 cm and 2 m. How long is the result?', False),
            ('8 cm', '8', 'How long is it?', False),
            ('8 cm', '8', 'A 32 cm wire is cut. How many pieces?', False),
            ('8 cm', '8', 'A 32 cm wire is cut. How long is the result in meters?', False),
            ('8 cm', '8 m', 'A length is 32 cm. How long is it?', False),
        ]:
            with self.subTest(a=a, context=context):
                self.assertEqual(_grade(a, b, context)['correct'], expected)


if __name__ == '__main__':
    unittest.main()
