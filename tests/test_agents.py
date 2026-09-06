"""Tests for M3: agent protocol, world-frame observation, frame round trip.

Acceptance (IMPLEMENTATION_PLAN.md, Phase 3 minimal slice):

- Every agent plays a full seeded match headless without crashing.
- The **frame round-trip test** is green: a known board state round-trips
  through ``observation -> agent -> expression -> solver/physics`` with the
  geometry preserved (this catches shooter-relative vs centered-world frame
  confusion before any LLM work).
- Parse-failure / retry counts are logged by the agents.

The handcrafted board in the round-trip tests uses **empty terrain**
(``make_circle_obstacle([])``), so the only thing that can stop a straight shot
is geometry — the test asserts the observation's world coordinates are the ones
the physics actually integrates.

Expected world coordinates are hand-computed from the documented plane->world
transform (GROUND_TRUTH.md §1.3): ``gx = 50·(px-385)/770``, ``gy = 50·(-py+225)/770``.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from agents import (
    Agent,
    AgentStats,
    Observation,
    RandomAgent,
    SolverAgent,
    StraightShotAgent,
    observe,
)
from graphwar_sim import (
    SOLDIER_RADIUS,
    TEAM1,
    TEAM2,
    Game,
    PolishNotationFunction,
)
from graphwar_sim.physics import Soldier
from graphwar_sim.state import GameState, Team, make_circle_obstacle

# Hand-computed world coordinates (independent of the implementation).
SHOOTER_T1_WORLD = (-18.506493506493507, -4.870129870129870)  # plane (100, 300)
ENEMY_T2_WORLD = (18.506493506493507, 4.870129870129870)  # plane (670, 150)
# TEAM2 shooter at plane (670, 150) mirrors to (100, 150): x flips, y unchanged.
SHOOTER_T2_WORLD = (-18.506493506493507, 4.870129870129870)
ENEMY_T1_WORLD = (18.506493506493507, -4.870129870129870)  # plane (100, 300) mirrored


def _empty_terrain_board() -> Game:
    """A handcrafted 1-vs-1 board on empty terrain (no circles)."""
    teams = [
        Team(name="team0", team=TEAM1, soldiers=[Soldier(x=100.0, y=300.0, alive=True)]),
        Team(name="team1", team=TEAM2, soldiers=[Soldier(x=670.0, y=150.0, alive=True)]),
    ]
    return Game(state=GameState(teams=teams, current_turn=0), terrain=make_circle_obstacle([]))


def _board_with_turn(board: Game, turn: int) -> Game:
    """Same board, different current turn (fresh object: fire mutates kills)."""
    teams = [
        Team(name="team0", team=TEAM1, soldiers=[Soldier(x=100.0, y=300.0, alive=True)]),
        Team(name="team1", team=TEAM2, soldiers=[Soldier(x=670.0, y=150.0, alive=True)]),
    ]
    return Game(state=GameState(teams=teams, current_turn=turn), terrain=make_circle_obstacle([]))


# --- Frame: observation presents the centered world frame --------------------


def test_observe_presents_centered_world_frame() -> None:
    """The observation reports the board in centered world coords (x in [-25,25],
    y up) — the space where ``f`` is evaluated, not shooter-relative."""
    obs = observe(_empty_terrain_board())
    assert obs.frame == "centered_world_shooter_facing"
    assert obs.shooter == pytest.approx(SHOOTER_T1_WORLD, abs=1e-9)
    assert _approx_points(obs.enemy_soldiers, [ENEMY_T2_WORLD], tol=1e-9)
    assert SHOOTER_T1_WORLD in obs.own_soldiers


def test_observe_mirrors_team2_shooter() -> None:
    """A TEAM2 shooter is mirrored about the vertical centre: its world position
    is the left side, and the enemy appears on the right (Function.java:188-191).
    Without the mirror, the shooter would appear at x=+18.5 with the enemy at
    x=-18.5 — the frame round trip would aim the wrong way."""
    obs = observe(_board_with_turn(_empty_terrain_board(), turn=1))
    assert obs.shooter == pytest.approx(SHOOTER_T2_WORLD, abs=1e-9)
    assert _approx_points(obs.enemy_soldiers, [ENEMY_T1_WORLD], tol=1e-9)


def test_observe_marks_current_shooter_in_ascii() -> None:
    """The ASCII board names the current shooter 'M' and enemies 'E'."""
    obs = observe(_empty_terrain_board())
    assert "M" in obs.ascii_board
    assert "E" in obs.ascii_board


def _approx_points(
    got: Sequence[tuple[float, float]],
    expected: Sequence[tuple[float, float]],
    tol: float = 1e-9,
) -> bool:
    """Compare a sequence of points pairwise with pytest.approx (which cannot
    nest inside tuples)."""
    if len(got) != len(expected):
        return False
    return all(
        got_p == pytest.approx(exp_p, abs=tol) for got_p, exp_p in zip(got, expected, strict=True)
    )


# --- Frame round trip: observation -> agent -> expression -> physics ---------


def test_frame_round_trip_straight_shot_hits_aimed_enemy() -> None:
    """The full round trip: observe the board in the centered world frame, have
    StraightShotAgent aim at the enemy, fire through the real physics, and the
    recorded hit is exactly that enemy — geometry preserved end to end."""
    game = _empty_terrain_board()
    obs: Observation = observe(game)
    expr = StraightShotAgent().act(game, obs)
    # Parseable, and it is a straight line through the muzzle.
    PolishNotationFunction(expr)
    assert "x" in expr

    result = game.fire(expr)
    # The shot starts at the muzzle (within the soldier radius of the shooter).
    shot_start = result.points[0]
    assert (shot_start[0] - 100.0) ** 2 + (shot_start[1] - 300.0) ** 2 < (2 * SOLDIER_RADIUS) ** 2
    # It hits the TEAM2 soldier — the enemy the observation placed at
    # (+18.506, +4.870) and the agent aimed at.
    assert any(player == 1 for player, _soldier, _pos in result.hits), (
        f"expected a hit on the TEAM2 enemy, got {result.hits!r}"
    )
    assert all(0 <= _pos < result.num_steps for _p, _s, _pos in result.hits)


def test_frame_round_trip_team2_mirror_hits_enemy() -> None:
    """Same round trip with the TEAM2 shooter: the mirrored observation must aim
    the curve so the physics' mirror brings it onto the TEAM1 enemy."""
    game = _board_with_turn(_empty_terrain_board(), turn=1)
    obs = observe(game)
    expr = StraightShotAgent().act(game, obs)
    result = game.fire(expr)
    assert any(player == 0 for player, _soldier, _pos in result.hits)


# --- Agent contract ----------------------------------------------------------


def test_all_agents_emit_parseable_expressions() -> None:
    """Every roster agent returns a parseable expression the simulator accepts,
    on a spread of seeded maps."""
    for seed in range(1, 8):
        game = Game.create(seed, num_soldiers=2)
        obs = observe(game)
        for agent in (SolverAgent(), RandomAgent(seed=seed), StraightShotAgent()):
            expr = agent.act(game, obs)
            PolishNotationFunction(expr)  # must not raise MalformedFunction
            fresh = Game.create(seed, num_soldiers=2)
            shot = fresh.fire(expr)
            assert shot.num_steps >= 1, f"seed {seed}: shot did not integrate"


def test_solver_agent_wraps_solve() -> None:
    """SolverAgent returns exactly ``solve(game).expression`` (the M2 solver)."""
    from graphwar_sim.solver import solve

    game = Game.create(13, num_soldiers=2)
    obs = observe(game)
    assert SolverAgent().act(game, obs) == solve(game).expression


def test_straight_shot_agent_aims_at_nearest_enemy() -> None:
    """The straight shot's slope is the world-frame secant to the nearest enemy."""
    game = _empty_terrain_board()
    obs = observe(game)
    expr = StraightShotAgent().act(game, obs)
    m = float(expr.replace("(", "").replace(")*x", "").replace(")", ""))
    mx, my = obs.shooter
    tx, ty = obs.enemy_soldiers[0]
    assert m == pytest.approx((ty - my) / (tx - mx), rel=1e-6)


def test_no_enemies_falls_back_to_safe_dud() -> None:
    """With no enemy alive, StraightShotAgent emits a parseable harmless line."""
    board = _empty_terrain_board()
    board.state.teams[1].soldiers[0].alive = False
    obs = observe(board)
    expr = StraightShotAgent().act(board, obs)
    PolishNotationFunction(expr)
    assert "0" in expr or "x" in expr


# --- Parse-failure / retry logging -------------------------------------------


def test_random_agent_tracks_parse_failures_and_retries() -> None:
    """RandomAgent validates every emit; parse-failure and retry counters move
    together and are visible via ``stats()`` (the M3 logging requirement)."""
    agent = RandomAgent(seed=42)
    board = _empty_terrain_board()
    for _ in range(20):
        expr = agent.act(board, observe(board))
        PolishNotationFunction(expr)  # every emit is parseable
    stats: AgentStats = agent.stats()
    # The validation loop never hands an unparseable expression back, but its
    # counters are the observable record of the (defensive) retry machinery.
    assert stats.parse_failures == stats.retries
    assert stats.parse_failures >= 0


def test_agents_log_parse_failures_through_matches() -> None:
    """Playing many turns across seeded maps, the agent never raises and keeps
    its counters consistent (the runner records them per match)."""
    agent = RandomAgent(seed=7)
    for seed in range(1, 6):
        game = Game.create(seed, num_soldiers=2)
        for _ in range(10):
            if game.finished():
                break
            expr = agent.act(game, observe(game))
            game.fire(expr)
            game.state.advance_turn()
    stats = agent.stats()
    assert stats.retries == stats.parse_failures


# --- Full seeded match, headless, no crash (M3 acceptance) -------------------


def _play_match(
    game: Game,
    agents_by_team: dict[int, Agent],
    cap: int = 50,
) -> int | None:
    """Minimal local match driver (the M4 runner is the harness version)."""
    for _ in range(cap):
        if game.finished():
            break
        team = game.state.current_team()
        agent = agents_by_team[team.team]
        expr = agent.act(game, observe(game))
        game.fire(expr)
        game.state.advance_turn()
    return game.winner()


@pytest.mark.parametrize(
    ("a", "b", "name"),
    [
        (SolverAgent(), RandomAgent(), "solver_vs_random"),
        (SolverAgent(), StraightShotAgent(), "solver_vs_straight"),
        (RandomAgent(), StraightShotAgent(), "random_vs_straight"),
    ],
)
def test_every_agent_plays_full_seeded_match(a: Agent, b: Agent, name: str) -> None:
    """Every roster pairing plays a full seeded match headless without crashing
    (the winner may be None on a dud-heavy draw at the turn cap)."""
    for seed in (1, 2, 3):
        game = Game.create(seed, num_soldiers=2)
        winner = _play_match(game, {TEAM1: a, TEAM2: b})
        assert winner in (TEAM1, TEAM2, None)
        assert game.finished() or len(game.all_soldiers()) == 4
