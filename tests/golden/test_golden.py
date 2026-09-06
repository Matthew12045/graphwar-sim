"""Golden tests: Python simulator vs the compiled Java reference.

Each scenario in ``data/golden.json`` was captured from the *reference*
``Graphwar.GoldenShot`` harness (see ``tools/golden/generate_golden.py``). This
suite replays the identical inputs through the Python simulator and asserts
parity on ``numSteps``, ``lastX``/``lastY``, the hit list, and the full
trajectory.

Divergence is reported, not tuned away: if a scenario fails, the failure
message shows the exact mismatch. Record any genuine divergence in
``docs/OPEN_QUESTIONS.md`` rather than loosening a tolerance to force a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphwar_sim import (
    Obstacle,
    PolishNotationFunction,
    Soldier,
    process_function_range,
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "golden.json"

# Tolerances. The port uses the same double-precision arithmetic as the
# reference in the same order, so exact equality is expected; a tiny tolerance
# guards against platform libm differences (sin/cos/tan/pow) without masking a
# real divergence.
REL_TOL = 1e-9
ABS_TOL = 1e-6


def _load() -> list[dict]:
    with DATA_PATH.open() as f:
        return json.load(f)


SCENARIOS = _load()
SCENARIO_IDS = [s["name"] for s in SCENARIOS]


def _make_obstacle(grid: list[str] | None) -> Obstacle:
    """Build terrain from the reference's exact ``collidePoint`` grid.

    ``grid`` is a list of ``PLANE_HEIGHT`` strings, each ``PLANE_LENGTH`` chars
    of ``'0'``/``'1'`` (row = y, char = x). Out-of-bounds returns ``True`` to
    match the reference (Obstacle.java:99-103). When ``grid`` is ``None``
    (no circles) the field is empty except for the out-of-bounds boundary.
    """
    rows = ["0" * 770] * 450 if grid is None else grid

    def collide_point(x: int, y: int) -> bool:
        if x < 0 or x >= 770 or y < 0 or y >= 450:
            return True
        return rows[y][x] == "1"

    return Obstacle(collide_point=collide_point)


def _replay(scenario: dict) -> dict:
    """Replay one scenario in the Python simulator and return a dict shaped
    like the reference result for easy comparison."""
    func = PolishNotationFunction(scenario["func"])
    inverted = bool(scenario["inverted"])
    shooter_xy = scenario["shooter"]

    soldiers: list[Soldier] = []
    shooter: Soldier | None = None
    for i, (sx, sy, alive) in enumerate(scenario["soldiers"]):
        s = Soldier(
            x=float(sx),
            y=float(sy),
            alive=bool(alive),
            player_index=i,
            soldier_index=0,
        )
        soldiers.append(s)
        if (sx, sy) == tuple(shooter_xy):
            shooter = s
    assert shooter is not None, "shooter not found among soldiers"

    obstacle = _make_obstacle(scenario["ref"]["grid"])
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


@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_IDS)
def test_shot_parity(scenario: dict) -> None:
    ref = scenario["ref"]
    got = _replay(scenario)

    # numSteps must match exactly (it drives the point count and lastX/lastY).
    assert got["numSteps"] == ref["numSteps"], (
        f"{scenario['name']}: numSteps {got['numSteps']} != {ref['numSteps']}"
    )

    assert _close(got["lastX"], ref["lastX"]), (
        f"{scenario['name']}: lastX {got['lastX']!r} != {ref['lastX']!r}"
    )
    assert _close(got["lastY"], ref["lastY"]), (
        f"{scenario['name']}: lastY {got['lastY']!r} != {ref['lastY']!r}"
    )

    # Hits: exact (player, soldier, position) triples, in order.
    assert got["hits"] == ref["hits"], f"{scenario['name']}: hits {got['hits']} != {ref['hits']}"

    # Trajectory: same length, each point within tolerance.
    assert len(got["points"]) == len(ref["points"]), (
        f"{scenario['name']}: {len(got['points'])} points != {len(ref['points'])}"
    )
    for i, (gp, rp) in enumerate(zip(got["points"], ref["points"], strict=True)):
        if not (_close(gp[0], rp[0]) and _close(gp[1], rp[1])):
            # Raise (not `assert False`, which `-O` strips) with the first
            # divergent point for a compact, actionable message.
            raise AssertionError(f"{scenario['name']}: point {i} {gp} != {rp}")


def test_at_least_20_scenarios() -> None:
    assert len(SCENARIOS) >= 20, f"expected >=20 golden scenarios, found {len(SCENARIOS)}"


def test_required_dimensions_covered() -> None:
    """The M1 acceptance criteria call for specific coverage."""
    names = {s["name"] for s in SCENARIOS}
    # (a) left-facing shooter (mirror / inverted=1)
    assert any(s["inverted"] == 1 for s in SCENARIOS), "no mirrored shooter"
    # (b) large f(0) auto-offset
    assert "large_f0_offset" in names and "negative_f0_offset" in names
    # (c) steep curve triggering step-halving
    assert "steep_parabola" in names and "quartic" in names
    # hit bookkeeping
    assert "guaranteed_hit" in names and "multi_soldier" in names
    # terrain (exact grid)
    assert any(s["ref"]["grid"] is not None for s in SCENARIOS), "no terrain scenario"
    # dead-soldier skip
    assert "dead_soldier" in names
