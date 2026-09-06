"""Tests for the shared emission module (5.2.md §2).

The emission budget is a *feasibility constraint on the solver*: a J-term
Gaussian sum nests additions, and the parser rewrites every ``-`` to ``+-``
(unary negation), which corrupts scientific notation. The formatter is plain
decimal only; the balanced-tree nesting keeps the parse depth at log2(J)
instead of J (the M5.3 AST-depth cap makes depth a hard constraint).
"""

from __future__ import annotations

import math

import pytest

from graphwar_sim import config
from graphwar_sim.emission import balanced_sum, format_literal, gauss_term
from graphwar_sim.parser import MalformedFunction, PolishNotationFunction

SAMPLE_X = (-25.0, -12.3, -0.5, 0.0, 3.7, 11.4286, 20.9, 25.0)


def _evaluate(expr: str, xs: tuple[float, ...]) -> list[float]:
    return [PolishNotationFunction(expr).evaluate(x) for x in xs]


def test_format_literal_is_plain_decimal() -> None:
    """Never an exponent: the parser's ``-``→``+-`` rewrite corrupts ``1e-06``."""
    assert format_literal(1e-6) == "0.000001"
    assert format_literal(-2.5) == "-2.5"
    assert format_literal(0.0) == "0"
    assert "e" not in format_literal(1.23e-11).lower().replace("e^", "")


@pytest.mark.parametrize(
    ("w", "c", "b"),
    [
        (1.0, 0.0, 0.5),
        (-0.75, -12.5, 2.0),  # negative weight AND negative centre
        (0.001, 25.0, 0.03125),  # tiny weight, far centre, sigma=4
    ],
)
def test_gauss_term_round_trip(w: float, c: float, b: float) -> None:
    expr = gauss_term(w, c, b)
    f = PolishNotationFunction(expr)  # must not raise
    for x in SAMPLE_X:
        assert f.evaluate(x) == pytest.approx(w * math.exp(-b * (x - c) ** 2), rel=1e-9, abs=1e-9)


def test_balanced_sum_matches_left_linear() -> None:
    """Balanced-tree nesting round-trips to the SAME values as left-linear
    nesting (5.2.md §2.3), for J from 1 to 33 (both tree shapes)."""
    terms = [gauss_term(0.5 * (i + 1), -10.0 + 1.3 * i, 0.5) for i in range(33)]
    for j in range(1, len(terms) + 1):
        subset = terms[:j]
        left_linear = "+".join(subset)
        balanced = balanced_sum(subset)
        f_lin = _evaluate(left_linear, SAMPLE_X)
        f_bal = _evaluate(balanced, SAMPLE_X)
        assert f_bal == pytest.approx(f_lin, rel=1e-12, abs=1e-12), f"J={j}"


def test_balanced_sum_bracket_depth_is_log2() -> None:
    """Maximum bracket nesting of the balanced form is O(log2 J), not O(J).

    The parser's evaluateRec recursion depth equals the bracket nesting of the
    emitted string, so this pins the M5.3 AST-depth budget observable.
    """

    def max_depth(expr: str) -> int:
        depth = 0
        best = 0
        for ch in expr:
            if ch == "(":
                depth += 1
                best = max(best, depth)
            elif ch == ")":
                depth -= 1
        return best

    for j in (2, 4, 8, 16, 32, 64):
        terms = [gauss_term(1.0, float(i), 0.5) for i in range(j)]
        expr = balanced_sum(terms)
        PolishNotationFunction(expr)  # parses
        # ceil(log2(J)) tree levels + 3 for the Gaussian term's own parens
        # (weight, e^(...), centre).
        assert max_depth(expr) <= math.ceil(math.log2(j)) + 3, f"J={j}"


def test_balanced_depth_fits_the_m53_ast_cap() -> None:
    """The two depth accounts compose: the balanced tree's ``ceil(log2 J) + 3``
    bracket depth stays under ``config.MAX_AST_DEPTH`` for every J the CCF
    emission budget could ever ask for (J <= j_max ≈ 55 ⇒ depth <= 9), with
    wide margin. Breaks if either the formatter's nesting or the cap changes.
    """

    def max_depth(expr: str) -> int:
        depth = 0
        best = 0
        for ch in expr:
            if ch == "(":
                depth += 1
                best = max(best, depth)
            elif ch == ")":
                depth -= 1
        return best

    for j in (2, 4, 8, 16, 32, 64, 128):
        terms = [gauss_term(1.0, float(i), 0.5) for i in range(j)]
        expr = balanced_sum(terms)
        bound = math.ceil(math.log2(j)) + 3
        assert bound <= config.MAX_AST_DEPTH, f"J={j}: bound {bound} exceeds the cap"
        PolishNotationFunction(expr)  # the parser's own guard accepts it
        assert max_depth(expr) <= bound, f"J={j}"


def test_balanced_sum_single_term() -> None:
    """One term: no wrapping, no change."""
    term = gauss_term(1.0, 3.0, 0.5)
    assert balanced_sum([term]) == term
    with pytest.raises(MalformedFunction):
        PolishNotationFunction(balanced_sum([]))


def test_balanced_sum_with_negative_constant_term() -> None:
    """CCF appends the affine constant ``(b2)`` — possibly negative — into the
    balanced tree; the ``-``→``+-`` rewrite must not corrupt it."""
    terms = [gauss_term(1.0, 2.0, 0.5), f"({format_literal(-3.25)})*x", f"({format_literal(-0.5)})"]
    expr = balanced_sum(terms)
    f = PolishNotationFunction(expr)
    for x in SAMPLE_X:
        expected = math.exp(-0.5 * (x - 2.0) ** 2) + (-3.25) * x + (-0.5)
        assert f.evaluate(x) == pytest.approx(expected, rel=1e-9, abs=1e-9)
