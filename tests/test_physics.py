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


def test_stop_cause_terrain_on_seeded_rock() -> None:
    """Pinned seed: a flat line into seeded rock dies with stop=terrain."""
    from graphwar_sim.physics import _java_int_cast

    game = Game.create(21, num_soldiers=2)
    f = PolishNotationFunction("0*x")
    shooter = game.state.current_team().current_soldier()
    result = process_function_range(f, shooter, game.all_soldiers(), game.terrain, inverted=False)
    assert result.stop == "terrain"
    # The break site is in-bounds rock (not an OOB mislabel).
    ix, iy = _java_int_cast(result.last_x), _java_int_cast(result.last_y)
    assert 0 <= ix < 770 and 0 <= iy < 450


def test_stop_cause_oob_on_empty_board() -> None:
    """Empty terrain: a steep climb can only die leaving the plane (oob)."""
    from graphwar_sim.state import make_circle_obstacle

    game = Game.create(21, num_soldiers=2)
    game.terrain = make_circle_obstacle([])
    f = PolishNotationFunction("10*x")
    shooter = game.state.current_team().current_soldier()
    result = process_function_range(f, shooter, game.all_soldiers(), game.terrain, inverted=False)
    assert result.stop == "oob"


def test_stop_cause_nan_on_empty_board() -> None:
    """Empty terrain: an always-NaN curve reaches the NaN check (stop=nan)."""
    from graphwar_sim.state import make_circle_obstacle

    game = Game.create(21, num_soldiers=2)
    game.terrain = make_circle_obstacle([])
    f = PolishNotationFunction("sqrt(x-1000)")
    shooter = game.state.current_team().current_soldier()
    result = process_function_range(f, shooter, game.all_soldiers(), game.terrain, inverted=False)
    assert result.stop == "nan"


def test_stop_cause_taxonomy_closure() -> None:
    """Every shot records a cause from the taxonomy (never empty/unknown).

    The ``steps`` bucket covers the adaptive step-halving floor and step
    exhaustion — nearly unreachable on real boards (poles die as ``oob``
    via the saturating int cast first), so this pins the contract, not a
    specific expression, plus the dataclass default for synthetic results.
    """
    from graphwar_sim.physics import ShotResult

    game = Game.create(21, num_soldiers=2)
    shooter = game.state.current_team().current_soldier()
    for expr in ("0*x", "10*x", "tan(x)", "log(x-30)", "sqrt(x-1000)", "1/(x-100)"):
        f = PolishNotationFunction(expr)
        result = process_function_range(
            f, shooter, game.all_soldiers(), game.terrain, inverted=False
        )
        assert result.stop in {"terrain", "oob", "nan", "steps"}, expr
    assert ShotResult().stop == "steps"
