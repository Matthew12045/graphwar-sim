"""Numeric literal and term emission for Graphwar expressions.

Single source of truth for turning solver/CCF numbers into legal Graphwar
``y = f(x)`` strings. Critical constraints (see ``docs/GROUND_TRUTH.md`` §3 and
``docs/OPEN_QUESTIONS.md``):

- The game parser rewrites every ``-`` to ``+-`` (unary negation), which
  corrupts scientific notation (``1e-06`` -> ``1e+-06`` misparses as
  ``1*e - 6``). Every numeric literal is therefore a **plain decimal**, never
  an exponent (``format_literal``).
- A negative centre or constant inside ``x - c`` would double-negate under the
  same rewrite, so signs are folded into parentheses (``gauss_term``).
- A J-term sum nests additions; nesting as a **balanced tree** keeps the parse
  depth at log2(J) instead of J (``balanced_sum``). The parser's
  ``reorderRec`` handles bracketed sub-expressions and produces a balanced
  prefix tree, so the emitted form round-trips to exactly the left-linear
  values (pinned by ``tests/test_emission.py``).

Validated against the real parser by the M2 round-trip tests and the M5.2
emission-budget tests.
"""

from __future__ import annotations


def format_literal(v: float) -> str:
    """Format a numeric literal as plain decimal, never scientific notation.

    The parser rewrites every ``-`` to ``+-`` (unary negation), so a literal
    such as ``1e-06`` would become ``1e+-06`` and misparse as ``1*e - 6``.
    We therefore never emit an exponent. Twelve decimal places round-trip a
    double far inside the ~1e-9 tolerance the round-trip tests assert.
    """
    s = f"{v:.12f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def gauss_term(w: float, c: float, b: float) -> str:
    """Emit one Gaussian term ``w·e^(-b(x-c)²)``, folding the centre's sign
    into the operator (a negative centre must not double-negate under the
    ``-``→``+-`` rewrite)."""
    center = f"x-({format_literal(c)})" if c >= 0 else f"x+({format_literal(-c)})"
    return f"({format_literal(w)})*e^(-{format_literal(b)}({center})^2)"


def balanced_sum(terms: list[str]) -> str:
    """Join term strings with additions nested as a **balanced binary tree**.

    ``((t1+t2)+(t3+t4))`` instead of ``t1+t2+t3+t4``: the parser's
    ``evaluateRec`` recursion depth equals the bracket nesting of the emitted
    string, so the balanced form parses with depth log2(J) — the emission-side
    half of the M5.3 AST-depth cap. Verified value-identical to left-linear
    nesting by ``tests/test_emission.py``.
    """
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    mid = len(terms) // 2
    return f"({balanced_sum(terms[:mid])}+{balanced_sum(terms[mid:])})"


__all__ = ["balanced_sum", "format_literal", "gauss_term"]
