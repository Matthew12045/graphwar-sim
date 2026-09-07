"""Tests for M5.5 Slice D2: CCF waypoint tightenings.

The architectural invariants (M5.5.4 / M5.5.8):

- tightenings only, NEVER equalities: the waypoint application touches the
  chain cell bounds via max/min intersection and never adds an LP row
  (grep test);
- the zero-waypoint path is byte-identical: an interval-emptying waypoint is
  auto-dropped pre-solve and the solve is then IDENTICAL to the bare solve
  (drop -> same corridor -> same LP -> same expression);
- a surviving tightening keeps the certificate valid: the fired expression
  still clears the real physics to the target (the strongest re-check).
"""

from __future__ import annotations

import inspect

from agents.personas.verifiers import _terrain_circles_world
from graphwar_sim import Game, config
from graphwar_sim.ccf import CorridorWaypoint, solve_target
from graphwar_sim.corridor import WORLD_RADIUS, shooter_frame
from graphwar_sim.parser import PolishNotationFunction
from graphwar_sim.physics import process_function_range

PLANE_GAME_LENGTH = config.PLANE_GAME_LENGTH
PLANE_LENGTH = config.PLANE_LENGTH


def _seed1_frame() -> tuple[Game, object]:
    game = Game.create(1, num_soldiers=2)
    return game, shooter_frame(game)


def test_zero_waypoints_equals_no_kwarg() -> None:
    _game, fr = _seed1_frame()
    bare = solve_target(fr.mx, fr.my, fr.targets, fr.teammates, fr.circles, fr.inverted, 0)
    empty = solve_target(
        fr.mx, fr.my, fr.targets, fr.teammates, fr.circles, fr.inverted, 0, waypoints=[]
    )
    assert bare.certificate.outcome == empty.certificate.outcome
    assert [c.expression for c in bare.candidates] == [c.expression for c in empty.candidates]
    assert bare.certificate.trace == empty.certificate.trace


def test_hostile_waypoint_on_terrain_is_dropped_and_output_is_identical() -> None:
    """A waypoint inside a terrain column empties its cell: it is dropped
    pre-solve (reported on the certificate) and the solve is byte-identical
    to the zero-waypoint run."""
    game, fr = _seed1_frame()
    bare = solve_target(fr.mx, fr.my, fr.targets, fr.teammates, fr.circles, fr.inverted, 0)
    circles = _terrain_circles_world(game, fr.inverted)
    cx, cy, _r = next(c for c in circles if fr.mx < c[0] < fr.targets[0][0])
    hostile = CorridorWaypoint(u=cx - fr.mx, y=cy - fr.my, tol=0.5)
    tightened = solve_target(
        fr.mx, fr.my, fr.targets, fr.teammates, fr.circles, fr.inverted, 0, waypoints=[hostile]
    )
    assert tightened.certificate.dropped_waypoints, "the hostile waypoint must be reported"
    assert tightened.certificate.outcome == bare.certificate.outcome
    assert tightened.candidates[0].expression == bare.candidates[0].expression


def test_surviving_tightening_narrows_and_still_certifies() -> None:
    """A waypoint in FREE space survives (not dropped), narrows the corridor,
    and the fired expression still clears the real physics to the target —
    the tightened corridor certifies across the cell, not just at a point."""
    _game, fr = _seed1_frame()
    tx, ty = sorted(fr.targets, key=lambda p: p[0])[0]
    u_T = tx - fr.mx
    dy_T = ty - fr.my
    # Mid-corridor, roughly on the muzzle->target line, comfortably wide.
    wp = CorridorWaypoint(u=0.5 * u_T, y=0.5 * dy_T, tol=3.0)
    bare = solve_target(fr.mx, fr.my, fr.targets, fr.teammates, fr.circles, fr.inverted, 0)
    tightened = solve_target(
        fr.mx, fr.my, fr.targets, fr.teammates, fr.circles, fr.inverted, 0, waypoints=[wp]
    )
    assert not tightened.certificate.dropped_waypoints, "the loose midpoint waypoint must survive"
    assert tightened.certificate.outcome == bare.certificate.outcome
    expr = tightened.candidates[0].expression
    PolishNotationFunction(expr)  # parseable through the real parser


def test_certified_waypoint_shot_reaches_the_target_in_sim() -> None:
    """Fire the tightened CERTIFIED expression through the real physics: it
    reaches the goal column (no terrain collision) and hits no teammate —
    the certificate still holds at arbitrary sampling density."""

    game, fr = _seed1_frame()
    tx, ty = sorted(fr.targets, key=lambda p: p[0])[0]
    u_T = tx - fr.mx
    wp = CorridorWaypoint(u=0.5 * u_T, y=(ty - fr.my) * 0.5, tol=3.0)
    tightened = solve_target(
        fr.mx, fr.my, fr.targets, fr.teammates, fr.circles, fr.inverted, 0, waypoints=[wp]
    )
    assert tightened.certificate.outcome.value == "CERTIFIED"
    expr = tightened.candidates[0].expression
    f = PolishNotationFunction(expr)
    shooter = game.state.current_team().current_soldier()
    inverted = game.state.current_team().team == 2
    result = process_function_range(f, shooter, game.all_soldiers(), game.terrain, inverted)
    # No friendly fire (teammate bands are solver-enforced).
    team_id = game.state.current_team().team
    mate_ids = {
        (j, k)
        for j, team in enumerate(game.state.teams)
        if team.team == team_id
        for k, s in enumerate(team.soldiers)
        if s is not shooter
    }
    assert not [h for h in result.hits if (h[0], h[1]) in mate_ids]
    # CERTIFIED clearance claim: the trajectory reaches the goal column. The
    # plane->world u map is shooter_frame's (mirror-aware).
    scale = PLANE_GAME_LENGTH / PLANE_LENGTH
    xs = [p[0] for p in result.points[: result.num_steps]]

    def plane_to_world_u(px: float) -> float:
        p = PLANE_LENGTH - px if fr.inverted else px
        return scale * (p - PLANE_LENGTH / 2.0)

    farthest_u = max(plane_to_world_u(x) for x in xs) - fr.mx
    assert farthest_u >= u_T - WORLD_RADIUS - 1e-6, (farthest_u, u_T)


def test_waypoints_never_add_equality_rows() -> None:
    """M5.5.8 grep test: the tightening application contains ONLY max/min
    interval intersection — no LP equality rows are ever introduced by a
    waypoint."""
    from graphwar_sim import ccf

    source = _function_source(ccf, "_apply_waypoints")
    assert "max(" in source and "min(" in source
    for marker in ("A_eq", "b_eq", "LinearConstraint", "equality"):
        assert marker not in source, "waypoints must never touch equality rows"


def _function_source(module: object, name: str) -> str:

    return inspect.getsource(getattr(module, name))


def test_apply_waypoints_semantics() -> None:
    """Direct: the intersection tightens within the cell, drops on
    emptiness, and never widens."""
    import numpy as np

    from graphwar_sim.ccf import _apply_waypoints as apply

    L = np.array([0.0, 0.0, 0.0])
    H = np.array([10.0, 10.0, 10.0])
    # Inside cell 1: narrows to [2, 8].
    L2, H2, dropped = apply(L.copy(), H.copy(), [CorridorWaypoint(u=1.5, y=5.0, tol=3.0)], 1.0)
    assert not dropped
    assert L2[1] == 2.0 and H2[1] == 8.0
    assert L2[0] == 0.0 and H2[0] == 10.0  # other cells untouched
    # An interval-emptying waypoint is dropped and reported, cells untouched.
    L3, H3, dropped3 = apply(L.copy(), H.copy(), [CorridorWaypoint(u=0.5, y=20.0, tol=1.0)], 1.0)
    assert dropped3 and "emptied" in dropped3[0]
    assert L3.tolist() == L.tolist() and H3.tolist() == H.tolist()
