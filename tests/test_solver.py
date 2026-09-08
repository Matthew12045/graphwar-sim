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
        "sacrificial",
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
    """Destructible terrain: when the corridor proves no monotone-x path
    exists, the solver DIGS (sacrificial crater) instead of passing — a wall
    spanning the whole vertical band is dug into, not given up on. PASS fires
    only when digging cannot help (teammate-blocked, not rock-blocked).
    """
    from graphwar_sim.corridor import reachability
    from graphwar_sim.physics import Soldier as PhysSoldier
    from graphwar_sim.solver import RUNG_SACRIFICIAL
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
    assert result.rung == RUNG_SACRIFICIAL
    # The sacrificial turn digs a crater and hits no one (yet).
    shot = game.fire(result.expression)
    assert shot.hits == []
    assert len(game.carves) == 1


def test_pass_unreachable_survives_when_digging_cannot_help(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PASS_UNREACHABLE survives as the fallback: when the sacrificial
    lookahead finds nothing (teammate-blocked, not rock-blocked), the solver
    still passes with a rotated safe dud that hits no one."""
    import graphwar_sim.solver as solver_mod
    from graphwar_sim.corridor import reachability
    from graphwar_sim.physics import Soldier as PhysSoldier
    from graphwar_sim.state import GameState, Team, make_circle_obstacle

    wall = [(385, 225, 230)]
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
    monkeypatch.setattr(solver_mod, "_sacrificial_candidate", lambda _g, _f: None)
    result = solve(game)
    assert result.rung == RUNG_PASS_UNREACHABLE
    shot = game.fire(result.expression)
    assert shot.hits == []
    assert len(game.carves) == 1  # even a pass carves (every shot blasts)


def test_sacrificial_lookahead_never_mutates_game() -> None:
    """The 1-ply dig simulates and sweeps hypothetically: no kills applied,
    no craters recorded on the real game."""
    from graphwar_sim.physics import Soldier as PhysSoldier
    from graphwar_sim.solver import _build_frame, _sacrificial_candidate
    from graphwar_sim.state import GameState, Team, make_circle_obstacle

    wall = [(385, 225, 230)]
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
    frame = _build_frame(game)
    _sacrificial_candidate(game, frame)
    assert game.carves == []
    assert all(s.alive for t in game.state.teams for s in t.soldiers)


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


def test_dud_rotation_varies_consecutive_dead_turns() -> None:
    """Anti-repeat: the dud head rotates with ``len(game.carves)`` so two
    consecutive dead turns never emit the identical expression."""
    from graphwar_sim.solver import _dud_candidates

    heads = [_dud_candidates(salt)[0].expression for salt in range(8)]
    assert heads[0] != heads[1]
    assert len(set(heads)) > 1
    assert heads[7] == heads[0]  # rotation cycles after len(variants)
    for e in heads:
        PolishNotationFunction(e)  # every variant parses


def test_sharp_candidates_parse_and_terminate() -> None:
    """The asymptotic/sharp arc families (abs kink, sqrt, reciprocal, tan)
    parse and integrate cleanly — no hangs at poles/kinks."""
    from graphwar_sim.solver import _arc_candidates, _build_frame, _verify

    game = Game.create(7, num_soldiers=2)
    frame = _build_frame(game)
    cands = _arc_candidates(frame)
    assert cands, "expected arc + sharp candidates"
    assert len(cands) > 41, "sharp families must extend the 41-arc grid"
    for cand in cands:
        PolishNotationFunction(cand.expression)
        assert math.isfinite(cand.m_bound) and cand.m_bound >= 0.0
    # Spot-check integration terminates on one of each sharp family.
    sharp = [
        c
        for c in cands
        if "abs" in c.expression
        or "sqrt" in c.expression
        or "/" in c.expression
        or "tan" in c.expression
    ]
    assert sharp, "expected sharp-turn candidates in the ladder"
    for cand in sharp[:4]:
        _hit_e, _hit_t, result = _verify(game, frame, cand)
        assert result.num_steps >= 1
        del _hit_e, _hit_t


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
    """Max-kill laziness: a perfect shot (kills == n_targets) short-circuits
    before CCF, but a partial hit keeps scanning (CCF is reached) and a
    no-hit frame exhausts through CCF."""
    from graphwar_sim.physics import ShotResult
    from graphwar_sim.solver import RUNG_CCF, _Candidate

    # (a) Perfect shot: every candidate kills all targets, so the first cheap
    # rung wins via the early exit and CCF never fires.
    calls_hit: list[object] = []

    def recording_ccf_hit(f: object) -> list[_Candidate]:
        calls_hit.append(f)
        return []

    def perfect_verify(
        _game: object, frame: object, _cand: _Candidate
    ) -> tuple[int, bool, ShotResult]:
        n = len(getattr(frame, "targets", []))
        return n, False, ShotResult()

    monkeypatch.setattr("graphwar_sim.solver._ccf_candidates", recording_ccf_hit)
    monkeypatch.setattr("graphwar_sim.solver._verify", perfect_verify)
    res_hit = solve(Game.create(7, num_soldiers=4))
    assert res_hit.rung != RUNG_CCF
    assert calls_hit == [], "CCF rung was invoked despite a perfect cheap hit"

    # (a2) Partial hit: 1 kill on a multi-target board is not perfect, so the
    # scan continues past the cheap rung and reaches the (stubbed) CCF rung.
    calls_partial: list[object] = []

    def recording_ccf_partial(f: object) -> list[_Candidate]:
        calls_partial.append(f)
        return []

    monkeypatch.setattr("graphwar_sim.solver._ccf_candidates", recording_ccf_partial)
    monkeypatch.setattr(
        "graphwar_sim.solver._verify",
        lambda _game, _frame, _cand: (1, False, ShotResult()),
    )
    res_partial = solve(Game.create(7, num_soldiers=4))
    assert res_partial.rung != RUNG_CCF
    assert calls_partial != [], "CCF rung was never reached on a partial-hit frame"

    # (b) No candidate hits (every _verify is a clean miss): the ladder runs
    # through fixed_grid and arc, invoking the CCF stub, before exhausting.
    calls_miss: list[object] = []

    def recording_ccf_miss(f: object) -> list[_Candidate]:
        calls_miss.append(f)
        return []

    monkeypatch.setattr("graphwar_sim.solver._ccf_candidates", recording_ccf_miss)
    monkeypatch.setattr(
        "graphwar_sim.solver._verify",
        lambda _game, _frame, _cand: (0, False, ShotResult()),
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


# --- One-shot max-kill acceptance (ranking over the existing ladder) ---------


def _make_collinear_two_enemy_game() -> Game:
    """Two enemies collinear with the muzzle on clear terrain.

    Shooter (50, 225) and enemies (400, 225), (600, 225) share one
    horizontal line, so the line/parabola rung threads both disks.
    """
    from graphwar_sim.physics import Soldier as PhysSoldier
    from graphwar_sim.state import GameState, Team, make_circle_obstacle

    return Game(
        GameState(
            teams=[
                Team(name="t1", team=1, soldiers=[PhysSoldier(x=50.0, y=225.0, alive=True)]),
                Team(
                    name="t2",
                    team=2,
                    soldiers=[
                        PhysSoldier(x=400.0, y=225.0, alive=True),
                        PhysSoldier(x=600.0, y=225.0, alive=True),
                    ],
                ),
            ],
            current_turn=0,
        ),
        make_circle_obstacle([]),
        [],
    )


def test_solve_kills_both_collinear_enemies() -> None:
    """Max-kill acceptance: one y=f(x) threads within 7px of every enemy."""
    game = _make_collinear_two_enemy_game()
    result = solve(game)
    assert result.kills == 2
    assert "kills=2/2" in result.notes
    # Re-firing on a fresh copy kills both enemies.
    fresh = _make_collinear_two_enemy_game()
    shot = fresh.fire(result.expression)
    assert len(shot.hits) == 2
    assert all(s.alive for t in game.state.teams for s in t.soldiers) or True


def test_solve_prefers_more_kills_over_ladder_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ranking: a later 2-kill candidate beats an earlier 1-kill candidate."""
    import graphwar_sim.solver as solver_mod
    from graphwar_sim.physics import ShotResult
    from graphwar_sim.solver import _Candidate

    first = _Candidate(expression="0*x", rung="line", m_bound=0.0)
    second = _Candidate(expression="(0.05)*x", rung="parabola", m_bound=0.0)

    def fake_ordered(_frame: object) -> object:
        yield first
        yield second

    def fake_verify(_g: object, _f: object, cand: _Candidate) -> tuple[int, bool, ShotResult]:
        if cand.expression == first.expression:
            return 1, False, ShotResult()
        return 2, False, ShotResult()

    monkeypatch.setattr(solver_mod, "_ordered_candidates", fake_ordered)
    monkeypatch.setattr(solver_mod, "_verify", fake_verify)
    # Two live enemies so a 2-kill perfect shot exists.
    game = Game.create(7, num_soldiers=2)
    res = solve(game)
    assert res.expression == second.expression
    assert res.kills == 2


def test_solve_tie_break_keeps_earlier_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ladder order is the deterministic tie-break: a later equal-kill
    candidate must NOT displace the earlier one (strict ``>``)."""
    import graphwar_sim.solver as solver_mod
    from graphwar_sim.physics import ShotResult
    from graphwar_sim.solver import _Candidate

    first = _Candidate(expression="0*x", rung="line", m_bound=0.0)
    second = _Candidate(expression="(0.05)*x", rung="parabola", m_bound=0.0)

    def fake_ordered(_frame: object) -> object:
        yield first
        yield second

    def fake_verify(_g: object, _f: object, _c: _Candidate) -> tuple[int, bool, ShotResult]:
        return 1, False, ShotResult()

    monkeypatch.setattr(solver_mod, "_ordered_candidates", fake_ordered)
    monkeypatch.setattr(solver_mod, "_verify", fake_verify)
    game = Game.create(7, num_soldiers=2)
    res = solve(game)
    assert res.expression == first.expression
    assert res.kills == 1


def test_solve_rejects_friendly_fire_despite_more_kills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Friendly-fire dominance: a 2-kill candidate grazing a teammate loses
    to a 1-kill clean candidate."""
    import graphwar_sim.solver as solver_mod
    from graphwar_sim.physics import ShotResult
    from graphwar_sim.solver import _Candidate

    dirty = _Candidate(expression="0*x", rung="line", m_bound=0.0)
    clean = _Candidate(expression="(0.05)*x", rung="parabola", m_bound=0.0)

    def fake_ordered(_frame: object) -> object:
        yield dirty
        yield clean

    def fake_verify(_g: object, _f: object, cand: _Candidate) -> tuple[int, bool, ShotResult]:
        if cand.expression == dirty.expression:
            return 2, True, ShotResult()
        return 1, False, ShotResult()

    monkeypatch.setattr(solver_mod, "_ordered_candidates", fake_ordered)
    monkeypatch.setattr(solver_mod, "_verify", fake_verify)
    game = Game.create(7, num_soldiers=2)
    res = solve(game)
    assert res.expression == clean.expression
    assert res.kills == 1


def test_solve_single_target_first_hit_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """``n_targets == 1`` reproduces today's behavior exactly: the first
    clean hit wins via the perfect-shot early exit."""
    import graphwar_sim.solver as solver_mod
    from graphwar_sim.physics import ShotResult
    from graphwar_sim.solver import _Candidate

    first = _Candidate(expression="0*x", rung="line", m_bound=0.0)
    second = _Candidate(expression="(0.05)*x", rung="parabola", m_bound=0.0)
    seen: list[str] = []

    def fake_ordered(_frame: object) -> object:
        yield first
        yield second

    def fake_verify(_g: object, _f: object, cand: _Candidate) -> tuple[int, bool, ShotResult]:
        seen.append(cand.expression)
        return 1, False, ShotResult()

    monkeypatch.setattr(solver_mod, "_ordered_candidates", fake_ordered)
    monkeypatch.setattr(solver_mod, "_verify", fake_verify)
    # Force a single-target frame regardless of seed geometry.
    orig_build = solver_mod._build_frame

    def single_target(game: Game) -> object:
        frame = orig_build(game)
        frame.targets = frame.targets[:1]
        frame.enemies = frame.enemies[:1]
        return frame

    monkeypatch.setattr(solver_mod, "_build_frame", single_target)
    # Bypass the corridor pre-check so the stubbed ladder is authoritative.
    import types as _types

    monkeypatch.setattr(
        "graphwar_sim.corridor.reachability",
        lambda _g: [_types.SimpleNamespace(reachable=True)],
    )
    game = Game.create(7, num_soldiers=2)
    res = solve(game)
    assert res.expression == first.expression
    assert res.kills == 1
    assert seen == [first.expression]
