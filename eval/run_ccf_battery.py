"""M5.2 battery: per-shot rung histogram + CCF certificate data on the M4 seeds.

Replicates the M4 measurement context EXACTLY (eval/results/seeds.json): the
seed-1000, 3-roster round robin (5 matches/pair x2 sides = 30 planned matches),
``num_soldiers=2``, ``max_turns=100``. Only the 20 solver-involving matches are
played (random-vs-straight matches produce no solver rungs and are not part of
the M4 rung histogram). Each solver match is played through the unchanged
``eval.runner.play_match``; the *agent* is a thin recording wrapper around the
stock ``SolverAgent`` that additionally captures the ``SolverResult`` (and thus
the ``CCFCertificate`` when the CCF rung fired) of every turn.

Per solver turn we record: the degradation rung, the emitted expression length,
and (for CCF shots) the full certificate. The counting rule matches M4 exactly:
one row per solver turn, ``rung = solver.rung_history`` back-stop as
``play_match`` computes its ``rung_counts``.

Output: ``eval/results/ccf_battery.json`` (raw per-shot records) + a human
readable summary block on stdout.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from agents import RandomAgent, StraightShotAgent
from agents.base import Agent, Observation
from agents.solver_agent import SolverAgent
from eval.runner import MatchConfig, play_match
from graphwar_sim import Game
from graphwar_sim.ccf import CCFCertificate
from graphwar_sim.solver import (
    RUNG_ARC,
    RUNG_CCF,
    RUNG_FIXED_GRID,
    RUNG_SOLVER_FAILED,
)

SEED_FILE = Path(__file__).resolve().parent / "results" / "seeds.json"
OUT_FILE = Path(__file__).resolve().parent / "results" / "ccf_battery.json"

# Final rungs that prove the CCF rung was actually reached (the ladder is lazy:
# CCF sits after per_target/parabola/line, so a shot whose final rung is ccf,
# fixed_grid, arc, or SOLVER_FAILED got past the closed-form rungs and paid the
# CCF scipy work). A final rung among the closed-form rungs means CCF never ran.
_CCF_REACHED_RUNGS: frozenset[str] = frozenset(
    {RUNG_CCF, RUNG_FIXED_GRID, RUNG_ARC, RUNG_SOLVER_FAILED}
)


class RecordingSolverAgent(SolverAgent):
    """Stock SolverAgent that records every turn's SolverResult (rung, expr, cert)."""

    def __init__(self) -> None:
        super().__init__()
        self.turn_meta: list[dict[str, Any]] = []

    def act(self, game: Game, obs: Observation) -> str:
        from graphwar_sim.solver import solve

        result = solve(game)
        self.turn_meta.append(_shot_record(result.rung, result.expression, result.cert))
        self.rung_history.append(result.rung)
        return result.expression


def _shot_record(rung: str, expression: str, cert: CCFCertificate | None) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "rung": rung,
        "expr_len": len(expression),
        "expression": expression,
    }
    if cert is None:
        rec["cert"] = None
        return rec
    rec["cert"] = {
        "outcome": str(cert.outcome),
        "sigma": cert.sigma,
        "ladder_depth": cert.ladder_depth,
        "branch_index": cert.branch_index,
        "branch_count": cert.branch_count,
        "branch_kind": cert.branch_kind,
        "slack_total": cert.slack_total,
        "nonzero_weights": cert.nonzero_weights,
        "emitted_length": cert.emitted_length,
        "char_limit": cert.char_limit,
        "milp_fired": cert.milp_fired,
        "solve_seconds": cert.solve_seconds,
        "binding": cert.binding,
    }
    return rec


def _make_side_agent(name: str, seed: int) -> Agent:
    if name == "solver":
        return RecordingSolverAgent()
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "straight":
        return StraightShotAgent()
    raise ValueError(f"unknown agent: {name}")


def _main() -> None:
    data = json.loads(SEED_FILE.read_text())
    cfg = MatchConfig(num_soldiers=data["num_soldiers"], max_turns=data["max_turns"])
    matches = data["matches"]

    matches_out: list[dict[str, Any]] = []
    shots_out: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    solver_matches = [m for m in matches if m["a"] == "solver" or m["b"] == "solver"]

    for m in solver_matches:
        seed = int(m["seed"])
        agent_a = _make_side_agent(m["a"], seed)
        agent_b = _make_side_agent(m["b"], seed)
        result = play_match(seed, agent_a, agent_b, cfg)

        solver_agent = agent_a if m["a"] == "solver" else agent_b
        # solver_matches keeps only matches with a solver side, and
        # _make_side_agent returns the recording wrapper for "solver".
        assert isinstance(solver_agent, RecordingSolverAgent)
        for meta in solver_agent.turn_meta:
            rec = dict(meta)
            rec["seed"] = seed
            rec["match"] = f"{m['a']} vs {m['b']}"
            shots_out.append(rec)

        matches_out.append(
            {
                "seed": seed,
                "pair": f"{m['a']} vs {m['b']}",
                "winner": result.winner,
                "turns": result.turns,
                "draw_reason": result.draw_reason,
            }
        )
        _log_match(seed, result.turns, result.winner, len(solver_agent.turn_meta))

    elapsed = time.perf_counter() - t_start
    payload = {
        "meta": {
            "battery": "M5.2 CCF",
            "context": "M4 context: root_seed=1000, 3-roster round robin, 5 matches/pair x2, "
            "num_soldiers=2, max_turns=100 (from eval/results/seeds.json)",
            "matches_played": len(matches_out),
            "solver_turns": len(shots_out),
            "source_script": "eval/run_ccf_battery.py",
            "wall_seconds": round(elapsed, 3),
        },
        "matches": matches_out,
        "shots": shots_out,
    }
    OUT_FILE.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n=== CCF battery summary ===")
    print(f"matches played: {len(matches_out)}  solver turns: {len(shots_out)}")
    print(f"wall time: {elapsed:.1f}s")
    print("rung histogram (per solver turn):")
    for rung, n in Counter(s["rung"] for s in shots_out).most_common():
        print(f"  {rung}: {n}")
    print(f"wrote raw records -> {OUT_FILE}")


def _log_match(seed: int, turns: int, winner: int | None, n_turns: int) -> None:
    print(f"  seed {seed}: turns={turns} winner={winner} solver_turns={n_turns}", flush=True)


if __name__ == "__main__":
    _main()
