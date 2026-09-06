"""Parser (tokenizer + prefix reordering) accept/reject table.

The reference tokenizer is a security surface (M3 will feed LLM strings into
it), so the accept/reject boundary is pinned down here. Valid inputs must
parse and evaluate correctly; malformed inputs must raise
:class:`MalformedFunction` (never crash, never reach ``eval``/``exec``).
"""

from __future__ import annotations

import pytest

from graphwar_sim import MalformedFunction, PolishNotationFunction

# (func, expected_value_at_x) — accepted, and evaluated at x=2.0.
ACCEPT = [
    ("x", 2.0),
    ("x^2*0.01", 0.04),
    ("x*0.1", 0.2),
    ("x^3*0.001", 0.008),
    ("x+1", 3.0),
    ("x-1", 1.0),
    ("x/2", 1.0),
    ("2*x", 4.0),  # implicit multiplication (number * variable)
    ("x^2+10", 14.0),
    ("sqrt(x)", 1.4142135623730951),
    ("sqrt(abs(x-3))+1", 2.0),
    ("log(x)", 0.3010299956639812),  # base-10
    ("ln(x)", 0.6931471805599453),  # natural
    ("abs(x-5)", 3.0),
    ("sin(x)", 0.9092974268256817),
    ("cos(x)*2", -0.8322936730942848),
    ("tan(x)*0.1", -0.2185039863261519),
    ("x^2*0.01-10", -9.96),
    ("(x+1)*(x-1)", 3.0),
    ("pi", 3.141592653589793),
    ("e", 2.718281828459045),
    ("x^2*0.01+10", 10.04),
]


@pytest.mark.parametrize("func,expected", ACCEPT)
def test_accept_and_evaluate(func: str, expected: float) -> None:
    f = PolishNotationFunction(func)
    assert f.evaluate(2.0) == pytest.approx(expected, rel=1e-12)


# Inputs that must be rejected: empty, a lone bracket, or a dangling operator
# (an operator with no operand to bind). Confirmed against the reference:
# ``Graphwar.GoldenShot`` raises on exactly these.
REJECT = [
    "",
    "(",
    ")",
    "x +",
    "x *",
    "x^",
    "sqrt(",
]


# The reference tokenizer is *lenient*: unmatched parentheses are silently
# dropped (the ``find()`` loop skips non-matching characters), and adjacent
# operands imply multiplication. These are therefore ACCEPTED, not rejected —
# pinning that down so a future "fix" that makes the parser strict doesn't
# silently change what a player's typed function means.
LENIENT = [
    ("(x+1", 3.0),
    ("x+1)", 3.0),
    ("x x", 4.0),
    ("x^2*0.01)", 0.04),
    ("((x)", 2.0),
]


@pytest.mark.parametrize("func,expected", LENIENT)
def test_lenient_accepts(func: str, expected: float) -> None:
    f = PolishNotationFunction(func)
    assert f.evaluate(2.0) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("func", REJECT)
def test_reject(func: str) -> None:
    with pytest.raises(MalformedFunction):
        PolishNotationFunction(func)


def test_no_eval_or_exec_in_module() -> None:
    """Guard the security invariant: the parser must not call eval/exec."""
    import inspect

    import graphwar_sim.parser as p

    src = inspect.getsource(p)
    # ``eval``/``exec`` may appear in comments/docstrings; assert they are not
    # *called*. A call is ``eval(`` or ``exec(`` not preceded by an identifier
    # char (which would make it a different name like ``evaluate``).
    for bad in ("eval(", "exec("):
        # ``evaluate(`` contains ``eval`` but not ``eval(`` (it's ``evaluate(``).
        assert bad not in src, f"parser.py appears to call {bad}"
