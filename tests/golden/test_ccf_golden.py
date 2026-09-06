"""Golden tests for M5.2 CCF: in-sim behaviour + parity with the Java reference.

Mirrors ``tests/golden/test_golden.py``. Each stored scenario in
``data/ccf_golden.json`` was captured from the CCF solver and then fired through
the *reference* ``Graphwar.GoldenShot`` harness (see
``tools/golden/generate_ccf_golden.py``) for up to two candidate expressions.
This suite asserts two things per candidate:

1. **In-sim, through the real physics** (the M5.2 "verified in-sim" half).
   We rebuild the :class:`Game` from the stored seed and fire the candidate's
   expression through ``physics.process_function_range`` (the real integrator,
   the same oracle ``solver._verify`` uses, solver.py:458-480). We assert:
   - **no friendly fire**: no hit lands on a teammate (corridor excludes
     teammate disks, corridor.py:86-97; the physics' only skip is the shooter
     itself, physics.py:229-233), and
   - **for CERTIFIED candidates, zero terrain collisions**: the trajectory
     reaches its goal column. The integrator advances ``x`` monotonically and
     stops at the first terrain pixel, truncating the trajectory
     (physics.py:245-247); asserting the trajectory's farthest world-``u``
     reached ``u_T - WORLD_RADIUS`` proves no terrain pixel blocked the path to
     the goal. UNCERTIFIED candidates make no clearance claim, so they get only
     the friendly-fire and parity assertions, not the clearance one.
2. **Parity with the Java reference** (the "against graphwar.jar" half), the
   exact way ``tests/golden/test_golden.py`` does it: rebuild the flat soldier
   list (one soldier per player, so the (player, soldier) hit identity matches
   the Java harness), reconstruct terrain from the reference's byte-identical
   collidePoint grid, fire through the real physics, and compare every field
   (numSteps, lastX, lastY, hits, points) to the stored reference output.

The grid stored per scenario is the anti-aliased Java oval fill (byte-identical
to ``GoldenShot``), so both replays use the true reference terrain; the CCF
certificate itself is expressed against the simulator's crisp circle model
(state.py:132-165), which is why the in-sim clearance check fires through the
rebuild ``Game`` rather than the grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphwar_sim import config
from graphwar_sim.corridor import WORLD_RADIUS, shooter_frame
from graphwar_sim.parser import PolishNotationFunction
from graphwar_sim.physics import Obstacle, Soldier, process_function_range
from graphwar_sim.state import Game

DATA_PATH = Path(__file__).resolve().parent / "data" / "ccf_golden.json"

REL_TOL = 1e-9
ABS_TOL = 1e-6


def _load() -> list[dict]:
    with DATA_PATH.open() as f:
        return json.load(f)


SCENARIOS = _load()
SCENARIO_IDS = [s["name"] for s in SCENARIOS]


def _candidate_cases():
    """(scenario, candidate, name) triples for pytest parametrization."""
    for s in SCENARIOS:
        for i, c in enumerate(s["candidates"]):
            yield s, c, f"{s['name']}#{i}"


_CASES3 = list(_candidate_cases())
_CASE_IDS = [name for _s, _c, name in _CASES3]


def _param():
    """Parametrize with 2 names: list of (scenario, candidate) pairs."""
    return [(s, c) for s, c, _name in _CASES3]


CASES = _param()
CASE_IDS = _CASE_IDS


def _make_obstacle(grid: list[str] | None) -> Obstacle:
    """Build terrain from the reference's exact ``collidePoint`` grid (identical
    to ``tests/golden/test_golden.py:_make_obstacle``)."""
    rows = ["0" * 770] * 450 if grid is None else grid

    def collide_point(x: int, y: int) -> bool:
        if x < 0 or x >= 770 or y < 0 or y >= 450:
            return True
        return rows[y][x] == "1"

    return Obstacle(collide_point=collide_point)


def _replay_java_parity(scenario: dict, candidate: dict) -> dict:
    """Fire one candidate exactly like ``test_golden._replay`` (flat soldiers,
    one player each) so the hit identity matches the Java harness."""
    func = PolishNotationFunction(candidate["expression"])
    inverted = bool(scenario["inverted"])
    shooter_xy = scenario["shooter"]

    soldiers: list[Soldier] = []
    shooter: Soldier | None = None
    for i, (sx, sy, alive) in enumerate(scenario["soldiers"]):
        s = Soldier(x=float(sx), y=float(sy), alive=bool(alive),
                     player_index=i, soldier_index=0)
        soldiers.append(s)
        if (sx, sy) == tuple(shooter_xy):
            shooter = s
    assert shooter is not None, "shooter not found among soldiers"

    obstacle = _make_obstacle(scenario["grid"])
    result = process_function_range(func, shooter, soldiers, obstacle, inverted)
    return {
        "numSteps": result.num_steps,
        "lastX": result.last_x,
        "lastY": result.last_y,
        "hits": [list(h) for h in result.hits],
        "points": [list(p) for p in result.points],
    }


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))


@pytest.mark.parametrize(("scenario", "candidate"), CASES, ids=CASE_IDS)
def test_java_reference_parity(scenario: dict, candidate: dict) -> None:
    """Python simulator replays the stored CCF expression in the flat-reader
    convention and must match the reference (Java) output field-for-field."""
    ref = candidate["ref"]
    got = _replay_java_parity(scenario, candidate)

    assert got["numSteps"] == ref["numSteps"], (
        f"{scenario['name']}: numSteps {got['numSteps']} != {ref['numSteps']}"
    )
    assert _close(got["lastX"], ref["lastX"]), (
        f"{scenario['name']}: lastX {got['lastX']!r} != {ref['lastX']!r}"
    )
    assert _close(got["lastY"], ref["lastY"]), (
        f"{scenario['name']}: lastY {got['lastY']!r} != {ref['lastY']!r}"
    )
    assert got["hits"] == ref["hits"], (
        f"{scenario['name']}: hits {got['hits']} != {ref['hits']}"
    )
    assert len(got["points"]) == len(ref["points"]), (
        f"{scenario['name']}: {len(got['points'])} points != {len(ref['points'])}"
    )
    for i, (gp, rp) in enumerate(zip(got["points"], ref["points"], strict=True)):
        if not (_close(gp[0], rp[0]) and _close(gp[1], rp[1])):
            raise AssertionError(f"{scenario['name']}: point {i} {gp} != {rp}")


def _fire_in_sim(seed: int, num_soldiers: int, expression: str):
    """Rebuild the :class:`Game` and fire ``expression`` through the real
    integrator. Returns ``(result, frame, game, shooter)`` with the muzzle, the
    real trajectory (:class:`ShotResult`) and the current-turn shooter."""
    game = Game.create(seed, num_soldiers=num_soldiers)
    fr = shooter_frame(game)
    shooter = game.state.current_team().current_soldier()
    f = PolishNotationFunction(expression)
    result = process_function_range(
        f, shooter, game.all_soldiers(), game.terrain, fr.inverted
    )
    return result, fr, game, shooter


def _reached_u(result, fr) -> float:
    """Farthest world-``u`` reached by a real trajectory (mirror-aware; undoes
    the plane transform and TEAM2 mirror of physics.py:222-226)."""
    px = result.points[-1][0]
    if fr.inverted:
        px = config.PLANE_LENGTH - px
    world_x = config.PLANE_GAME_LENGTH * (px - config.PLANE_LENGTH / 2.0) / config.PLANE_LENGTH
    return world_x - fr.mx


@pytest.mark.parametrize(("scenario", "candidate"), CASES, ids=CASE_IDS)
def test_in_sim_no_teammates_and_certified_clearance(
    scenario: dict, candidate: dict
) -> None:
    """Rebuild the Game, fire through the real physics: no friendly fire for
    every candidate, and zero terrain collisions (trajectory reaches the goal
    column) for CERTIFIED candidates."""
    result, fr, game, shooter = _fire_in_sim(
        scenario["seed"], scenario["num_soldiers"], candidate["expression"]
    )

    # --- no friendly fire -----------------------------------------------
    team_id = game.state.current_team().team
    mate_ids = {
        (j, k)
        for j, team in enumerate(game.state.teams)
        if team.team == team_id
        for k, s in enumerate(team.soldiers)
        if s is not shooter
    }
    mate_hits = [h for h in result.hits if (h[0], h[1]) in mate_ids]
    assert not mate_hits, (
        f"{scenario['name']}: friendly fire {mate_hits} (certified={candidate['certified']})"
    )

    # --- CERTIFIED candidates claim clearance: no terrain collision -----
    if candidate["certified"]:
        tx, _ty = fr.targets[candidate["target_index"]]
        u_T = tx - fr.mx
        reached = _reached_u(result, fr)
        assert reached >= u_T - WORLD_RADIUS, (
            f"{scenario['name']}: CERTIFIED shot stopped at u={reached:.3f} "
            f"(goal u_T={u_T:.3f}) -- terrain collision before the goal"
        )


def test_at_least_8_scenarios() -> None:
    """M5.2 §11 golden coverage: >= 8 scenarios, all with at least one CCF
    candidate."""
    assert len(SCENARIOS) >= 8, f"expected >=8 CCF golden scenarios, found {len(SCENARIOS)}"
    for s in SCENARIOS:
        assert s["candidates"], f"{s['name']}: no candidates stored"
        assert s["grid"] is not None, f"{s['name']}: expected a terrain grid"


def test_both_cert_and_uncert_covered() -> None:
    """The golden reference should span both certificate outcomes where the
    seeds provide them."""
    cert = any(c["certified"] for s in SCENARIOS for c in s["candidates"])
    uncert = any(not c["certified"] for s in SCENARIOS for c in s["candidates"])
    assert cert, "no CERTIFIED candidate in the CCF golden reference"
    assert uncert, "no UNCERTIFIED candidate in the CCF golden reference"
