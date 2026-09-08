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

D4 hybrid waypoint battery (``--waypoints``): a plan file of per-turn LLM
plans (JSON — the raw M5.5.3 plan objects ``agents.waypoints.parse_plan``
validates; no schema logic here) is applied to every battery board (the
turn-0 boards of the solver-match seeds) through the M5.5.5 relaxation
ladder — ``agents.hybrid_agent.solve_plan_with_ladder``, so the drop logic
is NOT duplicated. Boards whose corridor sweep is ``UNREACHABLE`` are
skipped with ``reason: corridor_empty``. ``--compare`` additionally solves
the same board with the plan's waypoints DISABLED (a zero-waypoint twin with
the same target and branch hint) and reports the delta: certified-rate
change, slack change, and the waypoints applied-vs-dropped counts (the
M5.5.9 applied-vs-dropped ratio per persona lives in
``summary.waypoint_ratio_per_style``; the kill decision stays with the user).

Output: the M5.2 path writes ``eval/results/ccf_battery.json`` (raw per-shot
records); the D4 path writes ``eval/results/hybrid_battery.json``
(``--out`` overrides). Both print a human readable summary block on stdout.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from agents import RandomAgent, StraightShotAgent
from agents.base import Agent, Observation
from agents.hybrid_agent import _HybridSolve, solve_plan_with_ladder
from agents.personas import PERSONAS
from agents.solver_agent import SolverAgent
from agents.waypoints import PlanSchemaError, WaypointPlan, parse_plan
from eval.runner import MatchConfig, play_match
from graphwar_sim import Game
from graphwar_sim.ccf import CCFCertificate, CCFOutcome
from graphwar_sim.corridor import map_y_bounds, shooter_frame
from graphwar_sim.solver import (
    RUNG_ARC,
    RUNG_CCF,
    RUNG_FIXED_GRID,
    RUNG_SOLVER_FAILED,
)

SEED_FILE = Path(__file__).resolve().parent / "results" / "seeds.json"
OUT_FILE = Path(__file__).resolve().parent / "results" / "ccf_battery.json"
HYBRID_OUT_FILE = Path(__file__).resolve().parent / "results" / "hybrid_battery.json"

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


# --- D4 hybrid waypoint battery (M5.5.9 kill-criteria measurement) ------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m eval.run_ccf_battery",
        description=(
            "No arguments: the M5.2 CCF battery. --waypoints FILE: the D4 hybrid waypoint battery."
        ),
    )
    parser.add_argument(
        "--waypoints",
        type=Path,
        metavar="FILE",
        help=(
            "run the D4 hybrid battery: FILE holds per-turn LLM plans (a JSON array of "
            'raw M5.5.3 plan objects, or {"plans": [...]}) applied to every battery '
            "board through the M5.5.5 relaxation ladder"
        ),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "with --waypoints: also solve every board with the plan's waypoints DISABLED "
            "(a zero-waypoint twin with the same target and branch hint) and report the "
            "delta (certified rate, slack, applied-vs-dropped)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="with --waypoints: only the first N battery boards (smoke runs)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help="with --waypoints: output path (default: eval/results/hybrid_battery.json)",
    )
    args = parser.parse_args(argv)
    if args.compare and args.waypoints is None:
        parser.error("--compare requires --waypoints FILE")
    return args


def _load_plans(path: Path) -> list[Any]:
    """Load the plan file: a JSON array of raw M5.5.3 plan objects (or a top-level
    ``{"plans": [...]}`` object). NO schema validation here — :func:`parse_plan`
    validates each plan against the board it is applied to (the schema lives in
    :mod:`agents.waypoints` and is not re-implemented)."""
    raw: Any = json.loads(path.read_text())
    if isinstance(raw, dict) and isinstance(raw.get("plans"), list):
        return list(raw["plans"])
    if isinstance(raw, list):
        return list(raw)
    raise SystemExit(
        f'plan file must be a JSON array of plan objects or {{"plans": [...]}}: {path}'
    )


def _battery_seeds(matches: list[Any], limit: int | None) -> list[int]:
    """The battery boards: the seeds of the solver-involving matches (the M4
    measurement context), deduplicated and ascending; ``limit`` keeps the
    first N."""
    seeds = sorted(
        {
            int(m["seed"])
            for m in matches
            if isinstance(m, dict) and (m.get("a") == "solver" or m.get("b") == "solver")
        }
    )
    return seeds if limit is None else seeds[: max(limit, 0)]


def _solve_record(
    seed: int,
    plan_index: int,
    style: str | None,
    target_id: str,
    side: str,
    solve: _HybridSolve,
) -> dict[str, Any]:
    cert = solve.cert
    return {
        "seed": seed,
        "plan_index": plan_index,
        "style": style,
        "target_id": target_id,
        "side": side,
        "skipped": False,
        "outcome": cert.outcome.value,
        "sigma": cert.sigma,
        "slack_total": cert.slack_total,
        "binding": cert.binding,
        "emitted_length": cert.emitted_length,
        "applied": solve.applied,
        "dropped": solve.dropped,
        "expression_emitted": solve.expression is not None,
    }


def _run_hybrid_battery(
    plans_raw: list[Any],
    seeds: list[int],
    num_soldiers: int,
    compare: bool,
) -> list[dict[str, Any]]:
    """Every plan applied to every battery board as an independent per-turn
    shot (turn-0 board: all soldiers alive, the plan's ``enemy_i`` indexes the
    living enemies sorted by x — the same convention the HybridAgent turn
    message uses) through the M5.5.5 relaxation ladder."""
    records: list[dict[str, Any]] = []
    band_lo, band_hi = map_y_bounds()
    for seed in seeds:
        game = Game.create(seed, num_teams=2, num_soldiers=num_soldiers)
        fr = shooter_frame(game)
        targets = sorted(fr.targets, key=lambda p: p[0])
        target_u_T = {f"enemy_{i}": targets[i][0] - fr.mx for i in range(len(targets))}
        for plan_index, raw in enumerate(plans_raw):
            base: dict[str, Any] = {
                "seed": seed,
                "plan_index": plan_index,
                "style": raw.get("style") if isinstance(raw, dict) else None,
                "target_id": raw.get("target_id") if isinstance(raw, dict) else None,
            }
            try:
                plan = parse_plan(
                    raw,
                    target_u_T=target_u_T,
                    band_lo=band_lo - fr.my,
                    band_hi=band_hi - fr.my,
                    styles=set(PERSONAS),
                )
            except PlanSchemaError as exc:
                # The plan never reaches the solver on this board (e.g. a u
                # beyond this board's u_T) — recorded, not silently dropped.
                records.append(
                    {**base, "skipped": False, "outcome": "SCHEMA_ERROR", "schema_error": str(exc)}
                )
                continue
            if compare:
                # The bare twin: same board, same target, same branch hint,
                # waypoints DISABLED (zero waypoints solves byte-identical to
                # bare CCF — the M5.5.8 zero-waypoint guarantee).
                bare = solve_plan_with_ladder(
                    fr,
                    targets,
                    WaypointPlan(target_id=plan.target_id, branch_hint=plan.branch_hint),
                )
                if bare.outcome is CCFOutcome.UNREACHABLE:
                    # The corridor sweep is empty BEFORE any waypoint: skip the
                    # pair entirely — never the plan's fault (M5.5.5).
                    records.append({**base, "skipped": True, "reason": "corridor_empty"})
                    continue
                records.append(
                    _solve_record(seed, plan_index, plan.style, plan.target_id, "bare", bare)
                )
            planned = solve_plan_with_ladder(fr, targets, plan)
            if not compare and planned.outcome is CCFOutcome.UNREACHABLE:
                records.append({**base, "skipped": True, "reason": "corridor_empty"})
                continue
            records.append(
                _solve_record(seed, plan_index, plan.style, plan.target_id, "planned", planned)
            )
    return records


def _style_key(style: Any) -> str:
    return style if isinstance(style, str) else "none"


def _ratio_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The M5.5.9 applied-vs-dropped bookkeeping over one record group."""
    applied = sum(int(r["applied"]) for r in records)
    dropped = sum(int(r["dropped"]) for r in records)
    total = applied + dropped
    return {
        "applied": applied,
        "dropped": dropped,
        "applied_vs_dropped_ratio": (applied / total) if total else None,
        "solves": len(records),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _per_style(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The M5.5.9 ratio grouped by the plan's persona (unstyled plans -> "none")."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(_style_key(r.get("style")), []).append(r)
    return {style: _ratio_block(groups[style]) for style in sorted(groups)}


def _waypoint_summary(records: list[dict[str, Any]], compare: bool) -> dict[str, Any]:
    """Aggregate outcome histograms + the M5.5.9 applied-vs-dropped ratio
    (overall and per persona/style); with ``compare``, the bare-vs-planned
    delta over the shared (board, plan) pairs."""
    solved = [r for r in records if not r.get("skipped") and r.get("outcome") != "SCHEMA_ERROR"]
    planned = [r for r in solved if r.get("side") == "planned"]
    summary: dict[str, Any] = {
        "records": len(records),
        "solves": len(solved),
        "skipped_corridor_empty": sum(1 for r in records if r.get("skipped")),
        "schema_errors": sum(1 for r in records if r.get("outcome") == "SCHEMA_ERROR"),
        "outcomes_planned": dict(sorted(Counter(str(r["outcome"]) for r in planned).items())),
        "waypoint_ratio_planned": _ratio_block(planned),
        "waypoint_ratio_per_style": _per_style(planned),
    }
    if compare:
        bare = [r for r in solved if r.get("side") == "bare"]
        summary["outcomes_bare"] = dict(sorted(Counter(str(r["outcome"]) for r in bare).items()))
        by_key: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
        for r in solved:
            key = (int(r["seed"]), int(r["plan_index"]))
            by_key.setdefault(key, {})[str(r["side"])] = r
        pairs = [
            (sides["bare"], sides["planned"])
            for sides in by_key.values()
            if "bare" in sides and "planned" in sides
        ]
        n_pairs = len(pairs)
        cert_bare = sum(1 for b, _ in pairs if b["outcome"] == CCFOutcome.CERTIFIED.value)
        cert_planned = sum(1 for _, p in pairs if p["outcome"] == CCFOutcome.CERTIFIED.value)
        both = [
            (b, p)
            for b, p in pairs
            if b["outcome"] == CCFOutcome.CERTIFIED.value
            and p["outcome"] == CCFOutcome.CERTIFIED.value
        ]
        slack_bare = [float(b["slack_total"]) for b, _ in both if b["slack_total"] is not None]
        slack_planned = [float(p["slack_total"]) for _, p in both if p["slack_total"] is not None]
        mean_bare = _mean(slack_bare)
        mean_planned = _mean(slack_planned)
        summary["compare"] = {
            "pairs": n_pairs,
            "certified_bare": cert_bare,
            "certified_planned": cert_planned,
            "certified_rate_bare": (cert_bare / n_pairs) if n_pairs else None,
            "certified_rate_planned": (cert_planned / n_pairs) if n_pairs else None,
            "certified_rate_change": ((cert_planned - cert_bare) / n_pairs) if n_pairs else None,
            "both_certified_pairs": len(both),
            "slack_mean_bare": mean_bare,
            "slack_mean_planned": mean_planned,
            "slack_mean_change": (
                (mean_planned - mean_bare)
                if mean_bare is not None and mean_planned is not None
                else None
            ),
        }
    return summary


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def _print_ratio(label: str, block: dict[str, Any]) -> None:
    ratio = block["applied_vs_dropped_ratio"]
    suffix = f"  (ratio {ratio:.3f})" if ratio is not None else ""
    print(
        f"  {label}: applied {block['applied']} / dropped {block['dropped']}"
        f"{suffix}  over {block['solves']} solves"
    )


def _print_hybrid_summary(summary: dict[str, Any], out: Path, elapsed: float) -> None:
    print("\n=== hybrid waypoint battery summary ===")
    print(
        f"records: {summary['records']}  solves: {summary['solves']}  "
        f"skipped (corridor_empty): {summary['skipped_corridor_empty']}  "
        f"schema errors: {summary['schema_errors']}"
    )
    print("planned outcome histogram:")
    for outcome, n in summary["outcomes_planned"].items():
        print(f"  {outcome}: {n}")
    _print_ratio("waypoints (all)", summary["waypoint_ratio_planned"])
    for style, block in summary["waypoint_ratio_per_style"].items():
        _print_ratio(f"waypoints (style={style})", block)
    cmp_block = summary.get("compare")
    if isinstance(cmp_block, dict):
        print("bare-vs-planned delta (same boards):")
        print(f"  pairs: {cmp_block['pairs']}")
        print(
            f"  certified: bare {cmp_block['certified_bare']} "
            f"({_fmt(cmp_block['certified_rate_bare'])}) vs planned "
            f"{cmp_block['certified_planned']} ({_fmt(cmp_block['certified_rate_planned'])}) "
            f"— rate change {_fmt(cmp_block['certified_rate_change'])}"
        )
        print(
            f"  slack (both-certified pairs: {cmp_block['both_certified_pairs']}): "
            f"bare {_fmt(cmp_block['slack_mean_bare'])} vs planned "
            f"{_fmt(cmp_block['slack_mean_planned'])} — change "
            f"{_fmt(cmp_block['slack_mean_change'])}"
        )
    print(f"wall time: {elapsed:.1f}s")
    print(f"wrote records -> {out}")


def _main_waypoints(args: argparse.Namespace) -> None:
    assert args.waypoints is not None  # argparse guarantees it (--compare guards too)
    data = json.loads(SEED_FILE.read_text())
    num_soldiers = int(data["num_soldiers"])
    seeds = _battery_seeds(list(data["matches"]), args.limit)
    plans_raw = _load_plans(args.waypoints)
    print(
        f"hybrid battery: {len(plans_raw)} plan(s) x {len(seeds)} board(s) "
        f"(compare={bool(args.compare)})",
        flush=True,
    )
    t_start = time.perf_counter()
    records = _run_hybrid_battery(plans_raw, seeds, num_soldiers, bool(args.compare))
    elapsed = time.perf_counter() - t_start
    summary = _waypoint_summary(records, bool(args.compare))
    out = args.out if args.out is not None else HYBRID_OUT_FILE
    payload = {
        "meta": {
            "battery": "D4 hybrid waypoints (M5.5.9 kill-criteria measurement)",
            "plan_file": str(args.waypoints),
            "compare": bool(args.compare),
            "boards": len(seeds),
            "plans": len(plans_raw),
            "num_soldiers": num_soldiers,
            "context": (
                "turn-0 boards of the M4 battery seeds (eval/results/seeds.json); each "
                "plan applied to every board through the M5.5.5 relaxation ladder "
                "(agents.hybrid_agent.solve_plan_with_ladder)"
            ),
            "source_script": "eval/run_ccf_battery.py",
            "wall_seconds": round(elapsed, 3),
        },
        "summary": summary,
        "records": records,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    _print_hybrid_summary(summary, out, elapsed)


def _main_cli() -> None:
    args = _parse_args()
    if args.waypoints is not None:
        _main_waypoints(args)
    else:
        _main()


if __name__ == "__main__":
    _main_cli()
