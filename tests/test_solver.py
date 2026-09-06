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
    RUNG_CCF,
    RUNG_PASS_UNREACHABLE,
    RUNG_SOLVER_FAILED,
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
        RUNG_CCF,
        RUNG_ARC,
        RUNG_PASS_UNREACHABLE,
        RUNG_SOLVER_FAILED,
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


def test_pass_unreachable_on_full_wall() -> None:
    """PASS_UNREACHABLE fires when the corridor proves no monotone-x path
    exists: a terrain wall spanning the whole vertical band between the teams
    is such a proof. The solver passes (safe flat dud), not a solver failure.
    """
    from graphwar_sim.corridor import reachability
    from graphwar_sim.physics import Soldier as PhysSoldier
    from graphwar_sim.state import GameState, Team, make_circle_obstacle

    wall = [(385, 225, 230)]  # px 155..615: the full vertical band at its core
    game = Game(
        GameState(
            teams=[
                Team(name="t1", team=1, soldiers=[PhysSoldier(x=50.0, y=400.0, alive=True)]),
                Team(name="t2", team=2, soldiers=[PhysSoldier(x=720.0, y=50.0, alive=True)]),
            ],
            current_turn=0,
        ),
        make_circle_obstacle(wall),
        wall,
    )
    assert not any(r.reachable for r in reachability(game))
    result = solve(game)
    assert result.rung == RUNG_PASS_UNREACHABLE
    # The pass turn's emission is a safe flat dud that hits no one.
    shot = game.fire(result.expression)
    assert shot.hits == []


def test_solver_failed_seeds_are_corridor_reachable() -> None:
    """The corridor refutes the M2 claim that six 2-soldier seeds are a
    genuine physical limit: those seeds are reachable by an arbitrary
    monotone-x path and the dud bucket's other half (SOLVER_FAILED) is a fit
    gap, recorded here as the M5.1 finding (docs/OPEN_QUESTIONS.md)."""
    from graphwar_sim.corridor import reachability

    results = run_battery(range(1, 41), num_soldiers=2)
    failed = [r for r in results if r.rung == RUNG_SOLVER_FAILED]
    assert failed, "expected SOLVER_FAILED seeds on the 2-soldier battery"
    for r in failed:
        game = Game.create(r.seed, num_soldiers=2)
        assert any(x.reachable for x in reachability(game)), (
            f"seed {r.seed}: SOLVER_FAILED but corridor says unreachable"
        )


def test_unreachable_proof_is_against_the_whole_ladder() -> None:
    """On a PASS_UNREACHABLE map no solver rung can land a clean hit — the
    corridor pre-check is a proof, not a solver failure. Assert by firing the
    whole ladder directly and confirming nothing hits."""
    from graphwar_sim.corridor import reachability
    from graphwar_sim.solver import _build_frame, _ordered_candidates, _verify

    results = run_battery(range(1, 41), num_soldiers=4)
    pass_seeds = [r.seed for r in results if r.rung == RUNG_PASS_UNREACHABLE]
    if not pass_seeds:
        return  # nothing to prove on this battery range; see the wall test
    for seed in pass_seeds:
        game = Game.create(seed, num_soldiers=4)
        assert not any(x.reachable for x in reachability(game))
        frame = _build_frame(game)
        for cand in _ordered_candidates(frame):
            hit_e, hit_t, _r = _verify(game, frame, cand)
            assert not hit_e and not hit_t, f"seed {seed}: reachable candidate on unreachable map"


# --- Security: no eval/exec in the solver hot path ---------------------------


def test_no_eval_or_exec_in_module() -> None:
    """The solver must not reach Python code-evaluation primitives."""
    import inspect

    import graphwar_sim.solver as mod

    src = inspect.getsource(mod)
    assert "eval(" not in src.replace("evaluate(", "")
    assert "exec(" not in src
    assert "sympify" not in src


# --- M5.2 CCF rung integration ----------------------------------------------


def test_ordered_candidates_is_lazy_and_ordered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ladder generator yields rungs in the 5.2.md §1 order and is lazy:
    pulling just the first candidate must not push the generator into CCF."""
    from graphwar_sim.solver import (
        RUNG_ARC,
        RUNG_CCF,
        RUNG_FIXED_GRID,
        RUNG_LINE,
        RUNG_PARABOLA,
        RUNG_PER_TARGET,
        _Candidate,
        _Frame,
        _ordered_candidates,
    )

    class _StubSoldier:
        x = 100.0
        y = 300.0
        player_index = 0
        soldier_index = 0
        alive = True

    frame = _Frame(
        mx=-5.0,
        my=0.0,
        targets=[(10.0, 5.0), (20.0, 8.0), (30.0, 3.0)],
        shooter=_StubSoldier(),
        enemies=[_StubSoldier(), _StubSoldier(), _StubSoldier()],
        inverted=False,
        circles=(),
        teammates=[],
    )

    # Laziness at the generator level: stub the CCF rung to record calls AND
    # return a cheap sentinel candidate; pulling only the first candidate must
    # never invoke it (the generator stops as soon as solve has its winner).
    calls: list[_Frame] = []

    def stub_ccf(f: _Frame) -> list[_Candidate]:
        calls.append(f)
        return [_Candidate(expression="0*x", rung=RUNG_CCF, m_bound=0.0)]

    monkeypatch.setattr("graphwar_sim.solver._ccf_candidates", stub_ccf)
    first = next(iter(_ordered_candidates(frame)))
    assert first.rung == RUNG_PER_TARGET
    assert calls == [], "CCF was invoked while producing the first candidate"

    # Full materialisation order: fixed-grid / arc also stubbed to cheap
    # sentinels so no expensive scipy / real LPs run at all.
    def stub_fixed(f: _Frame, b: float) -> _Candidate | None:
        return _Candidate(expression="0*x", rung=RUNG_FIXED_GRID, m_bound=0.0)

    def stub_arc(f: _Frame) -> list[_Candidate]:
        return [_Candidate(expression="0*x", rung=RUNG_ARC, m_bound=0.0)]

    monkeypatch.setattr("graphwar_sim.solver._fixed_grid_gaussians", stub_fixed)
    monkeypatch.setattr("graphwar_sim.solver._arc_candidates", stub_arc)
    kinds = [c.rung for c in _ordered_candidates(frame)]
    assert kinds == [
        RUNG_PER_TARGET,
        RUNG_PER_TARGET,
        RUNG_PER_TARGET,
        RUNG_PARABOLA,
        RUNG_LINE,
        RUNG_CCF,
        RUNG_FIXED_GRID,
        RUNG_ARC,
    ]


def test_ccf_rung_is_lazy_until_cheaper_rungs_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The solver never pays for CCF while a cheaper rung lands a verified hit,
    but does reach the (stubbed) CCF rung when no cheap rung does."""
    from graphwar_sim.physics import ShotResult
    from graphwar_sim.solver import RUNG_CCF, _Candidate

    # (a) Every candidate "hits": the first cheap rung wins, CCF never fires.
    calls_hit: list[object] = []

    def recording_ccf_hit(f: object) -> list[_Candidate]:
        calls_hit.append(f)
        return []

    monkeypatch.setattr("graphwar_sim.solver._ccf_candidates", recording_ccf_hit)
    monkeypatch.setattr(
        "graphwar_sim.solver._verify",
        lambda _game, _frame, _cand: (True, False, ShotResult()),
    )
    res_hit = solve(Game.create(7, num_soldiers=4))
    assert res_hit.rung != RUNG_CCF
    assert calls_hit == [], "CCF rung was invoked despite a cheaper verified hit"

    # (b) No candidate hits (every _verify is a clean miss): the ladder runs
    # through fixed_grid and arc, invoking the CCF stub, before exhausting.
    calls_miss: list[object] = []

    def recording_ccf_miss(f: object) -> list[_Candidate]:
        calls_miss.append(f)
        return []

    monkeypatch.setattr("graphwar_sim.solver._ccf_candidates", recording_ccf_miss)
    monkeypatch.setattr(
        "graphwar_sim.solver._verify",
        lambda _game, _frame, _cand: (False, False, ShotResult()),
    )
    res_miss = solve(Game.create(7, num_soldiers=4))
    assert res_miss.rung == RUNG_SOLVER_FAILED
    assert calls_miss != [], "CCF rung was never reached on a no-hit frame"


def test_solve_ccf_smoke() -> None:
    """A plain solve on a couple of real seeds still returns a parseable,
    verified expression; if the CCF rung wins, its certificate is attached."""
    from graphwar_sim.solver import RUNG_CCF

    for seed in (4, 11):
        game = Game.create(seed, num_soldiers=4)
        res = solve(game)
        PolishNotationFunction(res.expression)
        assert math.isfinite(res.bound) and res.bound >= 0.0
        if res.rung == RUNG_CCF:
            assert res.cert is not None
        shot = game.fire(res.expression)
        assert shot.num_steps >= 1
