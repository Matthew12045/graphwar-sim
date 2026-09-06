#!/usr/bin/env python3
"""Generate the golden reference data for the M5.2 CCF golden tests.

Mirrors :file:`tools/golden/generate_golden.py`. For each seeded map where the
CCF solver (:func:`graphwar_sim.ccf.solve_for_game`) yields at least one
candidate, this invokes the *compiled Java reference* (``Graphwar.GoldenShot``
in ``ref/graphwar/bin``) on the candidate's emitted expression and captures its
JSON output. The captured reference results are written to
``tests/golden/data/ccf_golden.json``. The pytest suite
(``tests/golden/test_ccf_golden.py``) then replays each scenario in the Python
simulator and asserts parity against this checked-in data (in-sim clearance +
no friendly fire + exact parity with the Java reference).

Keeping the reference output in a checked-in data file means the golden tests
run without a JVM; re-run this script
(``python3 tools/golden/generate_ccf_golden.py``) only to regenerate the
reference after a reference change.

Scenario selection
------------------
Seeds are probed in increasing order (``num_soldiers=2``) until >= 8 scenarios
each contributing at least one CCF candidate are captured; the underlying
seeds are whatever the deterministic generator produces. Up to two candidates
per scenario are stored (best-first: CERTIFIED before UNCERTIFIED).

A CERTIFIED candidate is only pinned as a golden case if it *in-sim clears*:
the fired expression, resimulated through the real integrator
(``physics.process_function_range``), reaches its goal column (no terrain
termination before the target). CERTIFIED candidates that do not clear are
genuine "CERTIFIED-but-collided" certificate defects and are owned by the
property test (``tests/test_ccf_property.py``), not by the golden reference —
a golden reference pins *valid* reference behavior. UNCERTIFIED candidates
make no clearance claim, so they are stored as-is (their trajectory still
replays, providing Java-parity coverage without a clearance assertion).

Usage:
    python3 tools/golden/generate_ccf_golden.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from graphwar_sim import ccf, config
from graphwar_sim.corridor import WORLD_RADIUS, shooter_frame
from graphwar_sim.parser import PolishNotationFunction
from graphwar_sim.physics import process_function_range
from graphwar_sim.state import Game

# Repository root is two levels up from this file (tools/golden/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
REF_DIR = REPO_ROOT / "ref" / "graphwar"
OUT_PATH = REPO_ROOT / "tests" / "golden" / "data" / "ccf_golden.json"

_NUM_SOLDIERS = 2
# Spec minimum for the golden §11 coverage. # TUNABLE (spec floor is 8).
_MIN_SCENARIOS = 8
# Candidates stored per scenario (best-first).
_MAX_CANDIDATES = 2


def _run_reference(
    func: str, inverted: int, shooter: tuple[int, int],
    soldiers: list[list[int]], circles: list[tuple[int, int, int]],
) -> list[str]:
    """Invoke the compiled Java reference and return its stdout lines.

    Identical mechanism to ``tools/golden/generate_golden.py``.
    """
    args = ["java", "-cp", "bin", "Graphwar.GoldenShot", func, str(inverted),
            f"{shooter[0]},{shooter[1]}"]
    for s in soldiers:
        args.append(f"{s[0]},{s[1]},{s[2]}")
    for c in circles:
        args.append(f"C:{c[0]},{c[1]},{c[2]}")
    proc = subprocess.run(args, cwd=REF_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"reference failed for func={func!r} rc={proc.returncode}\n"
            f"stderr:\n{proc.stderr}")
    return proc.stdout.strip().splitlines()


def _clears_in_sim(seed: int, cand: ccf.CCFCandidate) -> tuple[bool, float]:
    """Resimulate ``cand`` through the real integrator; return whether its
    trajectory reached the goal column plus the farthest ``u`` reached."""
    fresh = Game.create(seed, num_soldiers=_NUM_SOLDIERS)
    fr = shooter_frame(fresh)
    shooter = fresh.state.current_team().current_soldier()
    f = PolishNotationFunction(cand.expression)
    result = process_function_range(
        f, shooter, fresh.all_soldiers(), fresh.terrain, fr.inverted
    )
    px = result.points[-1][0]
    if fr.inverted:
        px = config.PLANE_LENGTH - px
    world_x = config.PLANE_GAME_LENGTH * (px - config.PLANE_LENGTH / 2.0) / config.PLANE_LENGTH
    reached = world_x - fr.mx
    tx, _ty = fr.targets[cand.target_index]
    return reached >= tx - fr.mx - WORLD_RADIUS, reached


def _probe_scenario(seed: int) -> tuple[dict, list[dict]] | None:
    """Build a golden scenario dict for ``seed``, or ``None`` if CCF yields no
    candidate worth pinning."""
    game = Game.create(seed, num_soldiers=_NUM_SOLDIERS)
    res = ccf.solve_for_game(game)
    fr = shooter_frame(game)
    team = game.state.current_team()
    shooter = team.current_soldier()
    inverted = 1 if team.team == config.TEAM2 else 0

    soldiers = [[int(s.x), int(s.y), 1] for t in game.state.teams for s in t.soldiers]
    circles = [list(c) for c in game.circles]

    kept: list[dict] = []
    for cand in res.candidates[: _MAX_CANDIDATES]:
        if cand.certified:
            ok, reached = _clears_in_sim(seed, cand)
            if not ok:
                # A real CLEARANCE claim failure: report loudly, don't pin it.
                print(f"    note: seed={seed} CERTIFIED candidate does not clear "
                      f"in-sim (reached u={reached:.3f}) -- not pinned as golden",
                      file=sys.stderr)
                continue
        kept.append({
            "expression": cand.expression,
            "target_index": cand.target_index,
            "sigma": cand.sigma,
            "mode": cand.mode,
            "certified": cand.certified,
        })
    if not kept:
        return None

    return {
        "name": f"seed_{seed}",
        "seed": seed,
        "num_soldiers": _NUM_SOLDIERS,
        "inverted": inverted,
        "shooter": [int(shooter.x), int(shooter.y)],
        "soldiers": soldiers,
        "circles": circles,
    }, kept


def main() -> int:
    if not (REF_DIR / "bin" / "Graphwar" / "GoldenShot.class").exists():
        print("error: GoldenShot.class not found; build the reference first "
              "(see ref/graphwar/compile.sh)", file=sys.stderr)
        return 1

    scenarios: list[dict] = []
    seed = 1
    while len(scenarios) < _MIN_SCENARIOS:
        out = _probe_scenario(seed)
        if out is None:
            seed += 1
            continue
        base, kept = out
        scenario = dict(base)
        candidates_out: list[dict] = []
        for cand_rec in kept:
            func = cand_rec["expression"]
            lines = _run_reference(func, scenario["inverted"], tuple(scenario["shooter"]),
                                   scenario["soldiers"], scenario["circles"])
            shot = json.loads(lines[0])
            grid = json.loads(lines[1]) if len(lines) > 1 else None
            cand_out = dict(cand_rec)
            cand_out["ref"] = {
                "numSteps": shot["numSteps"],
                "lastX": shot["lastX"],
                "lastY": shot["lastY"],
                "hits": shot["hits"],
                "points": shot["points"],
            }
            candidates_out.append(cand_out)
        scenario["candidates"] = candidates_out
        scenario["grid"] = grid
        scenarios.append(scenario)
        ncert = sum(c["certified"] for c in candidates_out)
        print(f"  seed {seed:4d} candidates={len(candidates_out)} certified={ncert}")
        seed += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(scenarios, f)
    size = OUT_PATH.stat().st_size
    total_cert = sum(1 for s in scenarios for c in s["candidates"] if c["certified"])
    total_uncert = sum(1 for s in scenarios for c in s["candidates"] if not c["certified"])
    print(f"\nwrote {len(scenarios)} scenarios to {OUT_PATH} "
          f"({size/1024:.1f} KiB); candidates: {total_cert} certified, "
          f"{total_uncert} uncertified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
