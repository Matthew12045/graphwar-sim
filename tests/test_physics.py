"""Regression tests for the Java ``(int)`` cast semantics in the physics.

The reference casts the trajectory's double plane coords before
``collidePoint`` with Java's ``(int)`` narrowing (JLS 5.1.3: NaN -> 0,
+-Inf saturate) — it can never raise. Python's ``int()`` raises on NaN, so
a curve that evaluates to NaN mid-flight (e.g. ``log(x-30)``, negative
past x=30) used to crash every unguarded consumer (``Game.fire``, the UI
fire path, the simulate oracle) with ``ValueError: cannot convert float
NaN to integer``. Reported divergences are tuned away only when the
reference cannot exhibit them — and the JVM cannot raise here.
"""

from __future__ import annotations

from graphwar_sim import Game, PolishNotationFunction
from graphwar_sim.physics import _java_int_cast, process_function_range


def test_java_int_cast_semantics() -> None:
    assert _java_int_cast(float("nan")) == 0
    assert _java_int_cast(float("inf")) == 2147483647
    assert _java_int_cast(float("-inf")) == -2147483648
    assert _java_int_cast(3.7) == 3
    assert _java_int_cast(-3.7) == -3


def test_nan_midflight_curve_terminates_without_raising() -> None:
    """A curve that goes NaN mid-flight terminates the shot (the NaN check
    right after collidePoint) instead of raising out of ``Game.fire``."""
    game = Game.create(21, num_soldiers=2)
    f = PolishNotationFunction("log(x-30)")  # NaN for x <= 30
    shooter = game.state.current_team().current_soldier()
    result = process_function_range(f, shooter, game.all_soldiers(), game.terrain, inverted=False)
    assert result.num_steps < 20000  # terminated early, not the full run
    # The full public fire path is crash-free for the same input.
    game2 = Game.create(21, num_soldiers=2)
    game2.fire("log(x-30)")
    game2.state.advance_turn()
