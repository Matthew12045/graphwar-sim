"""The compiled evaluator must equal the reference interpreter bit for bit."""

from __future__ import annotations

import math
import pickle
import random

import pytest

from graphwar_sim.parser import PolishNotationFunction

_EXPRS = [
    "x",
    "2",
    "-x",
    "x/4",
    "20x",
    "x^2/40 - 3",
    "sin(x)*3+0.1x^2",
    "1/x",
    "0/0+x",
    "sqrt(x)",
    "ln(x)",
    "log(x)",
    "tan(x)",
    "abs(x-1.429)+(x-1.429)",
    "(-2)^x",
    "x^0.5",
    "e^(-0.35(x+20.97)^2)*(-3.5)+(9.34)*e^(-0.35(x-(13.71))^2)",
    "2(3+4)x",
    "-(-(-x))",
    "pi*cos(x/pi)+sen(x)+tg(x)",
    "(x+1)(x-1)/(x-1)",
    "10^(10^x)",
]


def _same(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)


@pytest.mark.parametrize("expr", _EXPRS)
def test_compiled_matches_reference(expr: str) -> None:
    f = PolishNotationFunction(expr)
    rng = random.Random(expr)
    xs = [-25.0, -1.0, -0.0, 0.0, 1.0, 1.429, 25.0, math.inf, -math.inf, math.nan]
    xs += [rng.uniform(-30, 30) for _ in range(400)]
    for x in xs:
        assert _same(f.evaluate(x), f.evaluate_reference(x)), (expr, x)


def test_pickles_by_source() -> None:
    f = pickle.loads(pickle.dumps(PolishNotationFunction("x^2/4")))
    assert f.evaluate(2.0) == 1.0
