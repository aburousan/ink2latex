import unittest

from ink2latex.tokenizer import detokenize, tokenize


class CanonicalTest(unittest.TestCase):
    CASES = {
        "x^2": "x ^ { 2 }",
        r"\nabla ^ 2 V = - \frac { \rho } { \epsilon _ 0 }": r"\nabla ^ { 2 } V = - \frac { \rho } { \epsilon _ { 0 } }",
        r"\frac a b": r"\frac { a } { b }",
        r"\sqrt[3]{x}": r"\sqrt [ 3 ] { x }",
        r"a_{p}^{\dagger}": r"a _ { p } ^ { \dagger }",
        r"x^\prime": r"x ^ { \prime }",
        r"x^\hat a": r"x ^ { \hat { a } }",
        r"\frac{m}{\sqrt{\frac{e^{2}}{n^{2}}-1}}": r"\frac { m } { \sqrt { \frac { e ^ { 2 } } { n ^ { 2 } } - 1 } }",
    }

    def test_old_labels_get_mathwriting_braces(self):
        for source, expected in self.CASES.items():
            self.assertEqual(detokenize(tokenize(source)), expected, source)

    def test_idempotent_and_lists(self):
        for expected in self.CASES.values():
            self.assertEqual(tokenize(expected.split()), expected.split())

    def test_unbalanced_input_does_not_crash(self):
        for source in ("{", "}", "x^", r"\frac{a", "^{"):
            tokenize(source)


if __name__ == "__main__":
    unittest.main()


class CompactLatexTest(unittest.TestCase):
    def test_compact_and_round_trip(self):
        from ink2latex.tokenizer import to_latex, tokenize
        cases = {
            r"\begin { m a t r i x } 3 x - y \\ 2 x + y \end { m a t r i x }": r"\begin{matrix}3x-y\\2x+y\end{matrix}",
            r"\frac { d y } { d x } = 2 x": r"\frac{dy}{dx}=2x",
            r"\alpha x + \beta": r"\alpha x+\beta",
            r"\lim _ { x \rightarrow 3 } \sin x": r"\lim_{x\rightarrow3}\sin x",
            r"a ^ { \prime } \\ \alpha": r"a^{\prime}\\\alpha",
        }
        for spaced, compact in cases.items():
            tokens = spaced.split()
            self.assertEqual(to_latex(tokens), compact)
            self.assertEqual(tokenize(to_latex(tokens)), tokens)


class TidyTest(unittest.TestCase):
    def test_function_names(self):
        from ink2latex.tokenizer import tidy, to_latex
        self.assertEqual(to_latex(tidy("l i m _ { x \\rightarrow 3 } ( x )".split())), r"\lim_{x\rightarrow3}(x)")
        self.assertEqual(to_latex(tidy("s i n x + c o s y".split())), r"\sin x+\cos y")
        self.assertEqual(to_latex(tidy("s i n h x".split())), r"\sinh x")   # the longer name wins
        self.assertEqual(tidy("a s i n".split()), list("asin"))            # joined letters stay
        self.assertEqual(tidy(["\\sin", "x"]), ["\\sin", "x"])
