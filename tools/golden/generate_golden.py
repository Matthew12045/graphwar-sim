#!/usr/bin/env python3
"""Generate the golden reference data for the M1 golden tests.

For each scenario, this invokes the *compiled Java reference*
(``Graphwar.GoldenShot`` in ``ref/graphwar/bin``) and captures its JSON output.
The captured reference results are written to ``tests/golden/data/golden.json``.
The pytest suite (``tests/golden/test_golden.py``) then replays each scenario in
the Python simulator and asserts parity against this checked-in data.

Keeping the reference output in a checked-in data file means the golden tests
run without a JVM; re-run this script (``python3 tools/golden/generate_golden.py``)
only to regenerate the reference after a reference change.

Usage:
    python3 tools/golden/generate_golden.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Repository root is two levels up from this file (tools/golden/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
REF_DIR = REPO_ROOT / "ref" / "graphwar"
OUT_PATH = REPO_ROOT / "tests" / "golden" / "data" / "golden.json"

# Each scenario: (name, func, inverted, shooter, soldiers, circles)
#   func       firing function string (regular notation, as a player would type)
#   inverted   0/1 (TEAM2 mirror)
#   shooter    (x, y) plane/pixel coords of the current-turn soldier
#   soldiers   list of (x, y, alive); the shooter must be one of them
#   circles    list of (cx, cy, r) terrain circles (plane coords)
SCENARIOS: list[tuple[str, str, int, tuple[int, int], list[list[int]], list[list[int]]]] = [
    # --- Core trajectory shapes (no terrain) --------------------------------
    ("baseline_parabola", "x^2*0.01", 0, (50, 400), [(50, 400, 1)], []),
    ("mirror_parabola", "x^2*0.01", 1, (720, 400), [(720, 400, 1)], []),
    ("linear_shallow", "x*0.1", 0, (50, 400), [(50, 400, 1)], []),
    ("linear_mirror", "x*0.1", 1, (720, 400), [(720, 400, 1)], []),
    ("cubic", "x^3*0.001", 0, (50, 400), [(50, 400, 1)], []),
    ("sine", "sin(x)", 0, (50, 400), [(50, 400, 1)], []),
    ("cosine", "cos(x)*2", 0, (50, 400), [(50, 400, 1)], []),
    # --- Auto vertical offset: large f(0) forces the curve through the muzzle.
    ("large_f0_offset", "x^2*0.01+10", 0, (50, 400), [(50, 400, 1)], []),
    ("negative_f0_offset", "x^2*0.01-10", 0, (50, 400), [(50, 400, 1)], []),
    # --- Steep curves: exercise the adaptive step-halving loop.
    ("steep_parabola", "x^2*0.005", 0, (50, 400), [(50, 400, 1)], []),
    ("quartic", "x^4*0.0001", 0, (50, 400), [(50, 400, 1)], []),
    ("steep_line", "x*0.5", 0, (50, 400), [(50, 400, 1)], []),
    ("tan_asymptote", "tan(x)*0.1", 0, (50, 400), [(50, 400, 1)], []),
    ("sqrt_abs", "sqrt(abs(x))*0.5", 0, (50, 400), [(50, 400, 1)], []),
    ("log_offset", "log(abs(x)+1)*2", 0, (50, 400), [(50, 400, 1)], []),
    ("v_shape", "abs(x^2-100)*0.01", 0, (50, 400), [(50, 400, 1)], []),
    # --- Hit bookkeeping: multi-kill, dedup, dead-soldier skip.
    ("guaranteed_hit", "x^2*0.01", 0, (50, 400), [(50, 400, 1), (56, 403, 1)], []),
    # Soldiers placed on the baseline trajectory (sampled from the reference)
    # so the shot genuinely strikes several; also exercises dedup (a soldier
    # whose path is revisited is only recorded once).
    ("multi_soldier", "x^2*0.01", 0, (50, 400),
     [(50, 400, 1), (87, 415, 1), (118, 426, 1), (148, 436, 1)], []),
    ("dead_soldier", "x^2*0.01", 0, (50, 400), [(50, 400, 1), (56, 403, 0)], []),
    # --- Terrain: exact collidePoint grid (anti-aliased oval fill). Circles are
    # placed on the trajectory so the shot genuinely terminates at the terrain.
    ("terrain_block", "x^2*0.01", 0, (50, 400), [(50, 400, 1)],
     [(120, 430, 40)]),
    ("terrain_mirror", "x^2*0.01", 1, (720, 400), [(720, 400, 1)],
     [(650, 430, 40)]),
    ("terrain_hit", "x^2*0.01", 0, (50, 400), [(50, 400, 1), (87, 415, 1)],
     [(150, 438, 40)]),
    # --- Positional variety.
    ("center_shooter", "x*0.1", 0, (385, 400), [(385, 400, 1)], []),
    ("high_shooter", "x^2*0.01", 0, (50, 100), [(50, 100, 1)], []),
]


def _run_reference(func: str, inverted: int, shooter: tuple[int, int],
                   soldiers: list[list[int]], circles: list[list[int]]) -> list[str]:
    """Invoke the compiled Java reference and return its stdout lines."""
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


def main() -> int:
    if not (REF_DIR / "bin" / "Graphwar" / "GoldenShot.class").exists():
        print("error: GoldenShot.class not found; build the reference first "
              "(see ref/graphwar/compile.sh)", file=sys.stderr)
        return 1

    records: list[dict] = []
    for name, func, inverted, shooter, soldiers, circles in SCENARIOS:
        lines = _run_reference(func, inverted, shooter, soldiers, circles)
        shot = json.loads(lines[0])
        grid = json.loads(lines[1]) if len(lines) > 1 else None
        records.append({
            "name": name,
            "func": func,
            "inverted": inverted,
            "shooter": list(shooter),
            "soldiers": soldiers,
            "circles": circles,
            "ref": {
                "numSteps": shot["numSteps"],
                "lastX": shot["lastX"],
                "lastY": shot["lastY"],
                "hits": shot["hits"],
                "points": shot["points"],
                "grid": grid,
            },
        })
        npts = len(shot["points"])
        nhits = len(shot["hits"])
        print(f"  {name:20s} steps={shot['numSteps']:6d} hits={nhits} "
              f"pts={npts} grid={'yes' if grid else 'no'}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(records, f)
    size = OUT_PATH.stat().st_size
    print(f"\nwrote {len(records)} scenarios to {OUT_PATH} ({size/1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
