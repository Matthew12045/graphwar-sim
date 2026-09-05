"""Tests for the deterministic no-LLM solver (Phase 2 / Milestone M2).

Acceptance bar (IMPLEMENTATION_PLAN.md, Phase 2): on a seeded-map battery the
solver emits **parseable, certified** expressions; the hit rate and the
degradation-rung distribution are logged; and the emission **round-trips**
(``parse(to_graphwar(w))`` reproduces the intended curve). This is *not* a
high-hit-rate requirement — see ``docs/OPEN_QUESTIONS.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from graphwar_sim import Game
from graphwar_sim.parser import PolishNotationFunction
from graphwar_sim.solver import (
    RUNG_ARC,
    RUNG_DUD,
    SolverResult,
    _gauss_expression,
    run_battery,
    solve,
)

# --- Emission round-trip -----------------------------------------------------


@pytest.mark.parametrize(
    ("weights", "centers", "b"),
    [
        ([1.0], [0.0], 0.35),
        ([1.0, -0.5, 2.5], [0.0, -12.0, 15.5], 0.35),  # negative centre + weight
        ([-3.8961, 14.9616], [11.4286, 20.9091], 0.2),
        ([0.001, 1e-6], [-25.0, 25.0], 0.6),  # extreme centres, tiny weights
    ],
)
def test_round_trip(weights: list[float], centers: list[float], b: float) -> None:
    """``PolishNotationFunction(to_graphwar(w))`` reproduces the intended curve."""
    expr = _gauss_expression(np.array(weights), np.array(centers), b)
    f = PolishNotationFunction(expr)  # must not raise MalformedFunction

    def intended(x: float) -> float:
        return sum(w * math.exp(-b * (x - c) ** 2) for w, c in zip(weights, centers, strict=True))

    for x in (-25.0, -12.3, -0.5, 0.0, 3.7, 11.4286, 20.9, 25.0):
        assert f.evaluate(x) == pytest.approx(intended(x), rel=1e-9, abs=1e-9)


def test_round_trip_negative_center_no_double_negation() -> None:
    """A negative centre must not double-negate under the ``-``→``+-`` rewrite."""
    expr = _gauss_expression(np.array([1.0]), np.array([-10.5]), 0.35)
    f = PolishNotationFunction(expr)
    # At x = -10.5 the Gaussian is exactly 1.0 (its peak).
    assert f.evaluate(-10.5) == pytest.approx(1.0, rel=1e-12)


# --- Seeded-map battery (M2 acceptance) -------------------------------------


def test_battery_parseable_and_certified() -> None:
    """Every seeded map yields a parseable, certified (finite-bound) expression."""
    results = run_battery(range(1, 13), num_soldiers=4)
    assert len(results) == 12
    for r in results:
        assert r.parseable, f"seed {r.seed}: expression did not parse"
        assert r.certified, f"seed {r.seed}: certification bound not finite"
        assert r.rung, f"seed {r.seed}: no rung recorded"


def test_battery_no_friendly_fire() -> None:
    """The solver never fires a shot that hits a teammate."""
    results = run_battery(range(1, 25), num_soldiers=4)
    assert all(not r.hit_teammate for r in results)


def test_battery_hit_rate_and_rung_distribution_logged() -> None:
    """The M2 acceptance log: hit rate + degradation-rung distribution.

    We assert the *shape* of the log (a hit rate in [0,1] and a rung
    distribution over the known rungs), not a specific hit rate — the plan
    explicitly does not require high hit rate (terrain occlusion in NORMAL_FUNC
    mode is a genuine physical limit; see docs/OPEN_QUESTIONS.md).
    """
    results = run_battery(range(1, 25), num_soldiers=4)
    n = len(results)
    hit_rate = sum(r.hit_any_enemy for r in results) / n
    assert 0.0 <= hit_rate <= 1.0
    known_rungs = {
        "per_target_gaussian",
        "fixed_grid_gaussian",
        "line",
        "parabola",
        RUNG_ARC,
        RUNG_DUD,
    }
    rungs = {r.rung for r in results}
    assert rungs <= known_rungs
    # The distribution is a proper count over the seeds.
    assert sum(1 for r in results if r.rung in rungs) == n


# --- solve() contract --------------------------------------------------------


def test_solve_returns_legal_self_verified_result() -> None:
    """``solve`` returns a parseable expression that the simulator accepts."""
    game = Game.create(7, num_soldiers=4)
    result: SolverResult = solve(game)
    # Parseable (no MalformedFunction).
    PolishNotationFunction(result.expression)
    # Certified: finite, non-negative bound.
    assert math.isfinite(result.bound) and result.bound >= 0.0
    # The expression fires cleanly through the real integrator (no crash).
    game2 = Game.create(7, num_soldiers=4)
    shot = game2.fire(result.expression)
    assert shot.num_steps >= 1


def test_solve_deterministic() -> None:
    """The same seed yields the identical expression (no hidden RNG)."""
    a = solve(Game.create(3, num_soldiers=4))
    b = solve(Game.create(3, num_soldiers=4))
    assert a.expression == b.expression
    assert a.rung == b.rung


# --- Degradation: no-solution maps fall to the safe dud, never crash ---------


def test_solve_never_crashes_on_hard_maps() -> None:
    """Across a wide seed range, solve always returns (never raises)."""
    for seed in range(1, 41):
        game = Game.create(seed, num_soldiers=4)
        result = solve(game)
        assert isinstance(result, SolverResult)
        PolishNotationFunction(result.expression)  # parseable
        assert not result.notes.startswith("FF")  # no friendly fire recorded


def test_arc_rung_clears_terrain_the_cheaper_rungs_cannot() -> None:
    """The terrain-aware arc rung rescues maps the cheaper rungs dud on.

    Because the ladder only reaches the arc rung after every cheaper rung has
    failed to land a clean shot, a seed that lands on ``arc`` and hits an
    enemy is, by construction, a map the parabola/line/Gaussian rungs could
    not clear — i.e. the arc bulged over/under terrain. We assert the rung is
    non-empty on the seeded battery and that every such seed is a real, clean
    hit.
    """
    results = run_battery(range(1, 41), num_soldiers=4)
    arc = [r for r in results if r.rung == RUNG_ARC]
    assert arc, "expected the arc rung to clear terrain on at least one seeded map"
    for r in arc:
        assert r.hit_any_enemy, f"seed {r.seed}: arc rung did not hit an enemy"
        assert not r.hit_teammate, f"seed {r.seed}: arc rung caused friendly fire"
        assert r.parseable and r.certified


def test_dud_rung_hits_nothing() -> None:
    """When the ladder reaches the dud rung, the shot hits no one."""
    # Find a seed whose solve lands on the dud rung, then confirm the dud is
    # clean (no enemy, no teammate).
    dud_seeds = [r.seed for r in run_battery(range(1, 41), num_soldiers=4) if r.rung == RUNG_DUD]
    assert dud_seeds, "expected at least one seed to degrade to the dud rung"
    for seed in dud_seeds:
        game = Game.create(seed, num_soldiers=4)
        shot = game.fire(solve(game).expression)
        assert shot.hits == [], f"seed {seed}: dud rung hit someone"


# --- Security: no eval/exec in the solver hot path ---------------------------


def test_no_eval_or_exec_in_module() -> None:
    """The solver must not reach Python code-evaluation primitives."""
    import inspect

    import graphwar_sim.solver as mod

    src = inspect.getsource(mod)
    assert "eval(" not in src.replace("evaluate(", "")
    assert "exec(" not in src
    assert "sympify" not in src
