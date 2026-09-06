"""Seeded match runner and leaderboard builder (Phase 4 / M4).

The runner is a pure function of its inputs: a list of planned matches
``(seed, agent_a, agent_b)`` and a :class:`MatchConfig`. Every match creates a
fresh ``Game`` from its seed and fresh agent instances (RandomAgent seeded per
match), so the whole leaderboard is reproducible **from the seed file alone**
(see :func:`run_from_seed_file` and ``eval/results/seeds.json``).

A match is 2 teams x ``num_soldiers``, turns alternating between the teams
until either side is wiped or the turn cap is hit (a draw). Turn advancement,
the win rule, and the physics are the M1/M2 simulator's; the runner adds
per-shot bookkeeping (expression, hit classification, parse failures) and a
per-match plot via :func:`graphwar_sim.render.render_match`.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, cast

from agents import (
    Agent,
    AgentStats,
    LLMAgent,
    RandomAgent,
    SolverAgent,
    StraightShotAgent,
    hit_team_counts,
    observe,
)
from graphwar_sim import TEAM1, TEAM2, Game, render
from graphwar_sim import config as sim_config
from graphwar_sim.parser import MalformedFunction
from graphwar_sim.physics import ShotResult
from graphwar_sim.solver import RUNG_PASS_UNREACHABLE, RUNG_SOLVER_FAILED

from .metrics import AgentLeaderRow, AgentMatchStats, ShotOutcome, team_label

# The M3 roster. Agent factories are seeded per match so RandomAgent (the only
# nondeterministic baseline) is reproducible from the seed file.
DEFAULT_ROSTER: tuple[str, ...] = ("solver", "random", "straight")

_AGENT_FACTORIES: dict[str, Callable[[int], Agent]] = {
    "solver": lambda _seed: SolverAgent(),
    "random": lambda seed: RandomAgent(seed=seed),
    "straight": lambda _seed: StraightShotAgent(),
}

# Seed layout: pair p, match m, side s in {0, 1} -> root + p * 1000 + m * 2 + s.
_PAIR_SEED_STRIDE: int = 1000


@dataclass(frozen=True)
class MatchConfig:
    """Per-match knobs (all # TUNABLE harness parameters, not game constants)."""

    num_soldiers: int = sim_config.INITIAL_NUM_SOLDIERS
    max_turns: int = 100  # shots per match before a draw is declared


@dataclass(frozen=True)
class PlannedMatch:
    """One planned match: ``a`` drives TEAM1, ``b`` drives TEAM2."""

    seed: int
    a: str
    b: str


# Stalemate rule (M5.1): this many consecutive PASS_UNREACHABLE turns from
# BOTH teams ends the match as a genuine draw (DRAW_STALEMATE), distinct from
# the turn-cap draw. # TUNABLE.
_STALL_LIMIT: int = 4
# Time budget for one trajectory evaluation (parse + integrate). Exceeding it
# classifies the turn TIMEOUT. # TUNABLE — the reference game's TURN_TIME is
# 60s per turn (Constants.java:46); headless agents should be far faster.
_SHOT_TIME_BUDGET: float = 2.0


@dataclass
class ShotRecord:
    """One fired shot, recorded for the match log and per-match plot."""

    agent: str
    team_id: int
    expression: str
    parse_failure: bool
    result: ShotResult = field(default_factory=ShotResult)
    outcome: str = ShotOutcome.MISS.value
    solver_rung: str | None = None


@dataclass
class MatchResult:
    """The outcome of one match plus its per-agent counters."""

    seed: int
    agent_a: str
    agent_b: str
    winner: int | None  # TEAM1 / TEAM2 / None (draw)
    turns: int
    stats: dict[str, AgentMatchStats]
    shots: list[ShotRecord] = field(default_factory=list)
    rung_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    plot_file: str = ""
    # M5.1: None (decisive) | "TURN_CAP" | "STALEMATE".
    draw_reason: str | None = None


@dataclass
class Leaderboard:
    """A full run: matched results + rolled-up per-agent rows + artifacts."""

    root_seed: int
    roster: tuple[str, ...]
    num_matches_per_pair: int
    config: MatchConfig
    matches: list[MatchResult]
    rows: list[AgentLeaderRow]
    out_dir: Path
    leaderboard_file: Path
    seed_file: Path


# --- Plan derivation ---------------------------------------------------------


def build_plan(
    root_seed: int,
    roster: Sequence[str],
    n_matches: int,
) -> list[PlannedMatch]:
    """The round-robin plan: every unordered pair, ``n_matches`` seeds, each
    played on both sides (a=TEAM1 and a=TEAM2) with unique derived seeds."""
    plan: list[PlannedMatch] = []
    for p, (a, b) in enumerate(combinations(roster, 2)):
        for m in range(n_matches):
            base = root_seed + p * _PAIR_SEED_STRIDE + m * 2
            plan.append(PlannedMatch(seed=base, a=a, b=b))
            plan.append(PlannedMatch(seed=base + 1, a=b, b=a))
    return plan


# --- One match ---------------------------------------------------------------


def _state_hash(game: Game) -> tuple[object, ...]:
    """M5.1 dedupe key: the board state that determines a shot's outcome.

    Positions + alive flags of every soldier, plus the current shooter
    identity (team + soldier). The global turn index is intentionally NOT
    included: a recurring (shooter, board) with an identical expression is a
    repeat attempt even though the turn counter advanced.
    """
    team = game.state.current_team()
    shooter = team.current_soldier()
    shooter_j = game.state.teams.index(team)
    shooter_k = team.soldiers.index(shooter)
    soldiers: list[tuple[int, int, bool, float, float]] = []
    for j, t in enumerate(game.state.teams):
        for k, s in enumerate(t.soldiers):
            # Identity by position: the lazy ``player_index``/``soldier_index``
            # fields are only assigned inside fire() (all_soldiers), so they
            # would make turn-0 and later hashes differ spuriously.
            soldiers.append((j, k, s.alive, s.x, s.y))
    return (team.team, shooter_j, shooter_k, tuple(soldiers))


def play_match(
    seed: int,
    agent_a: Agent,
    agent_b: Agent,
    cfg: MatchConfig,
) -> MatchResult:
    """Play one seeded match: ``agent_a`` drives TEAM1, ``agent_b`` TEAM2.

    Pure function of ``(seed, agent_a, agent_b, cfg)`` — no global RNG is
    touched (RandomAgent is seeded by the caller, typically
    :func:`_make_agents` with the match seed). Returns the result without
    writing any artifact.

    M5.1: every turn is classified into the :class:`ShotOutcome` taxonomy;
    duplicate attempts (unchanged board state + identical expression) are
    suppressed into ``repeat_suppressed``; and consecutive PASS_UNREACHABLE
    turns from both teams end the match as ``DRAW_STALEMATE``.
    """
    game = Game.create(seed, num_teams=2, num_soldiers=cfg.num_soldiers)
    agents_by_team = {TEAM1: agent_a, TEAM2: agent_b}
    stats = {
        agent_a.name: AgentMatchStats(agent=agent_a.name),
        agent_b.name: AgentMatchStats(agent=agent_b.name),
    }
    shots: list[ShotRecord] = []
    turns = 0
    draw_reason: str | None = None
    last_key: dict[str, tuple[object, ...]] = {}
    pass_run = 0
    pass_teams: set[int] = set()

    while not game.finished() and turns < cfg.max_turns:
        team = game.state.current_team()
        agent = agents_by_team[team.team]
        state_hash = _state_hash(game)
        obs = observe(game)
        expr = agent.act(game, obs)
        match_stats = stats[agent.name]
        solver_rung = _peek_solver_rung(agent)

        # M5.1 dedupe: unchanged board state + identical expression.
        key = (state_hash, expr)
        suppressed = last_key.get(agent.name) == key
        last_key[agent.name] = key

        t0 = time.perf_counter()
        try:
            result = game.fire(expr)
            parse_failure = False
        except MalformedFunction:
            # Defensive: a buggy agent must never crash a match. The safe dud
            # is the plan's last-resort emission; count the failure (M3 log).
            match_stats.parse_failures += 1
            result = game.fire("0*x")
            parse_failure = True
        elapsed = time.perf_counter() - t0

        enemy_hits, teammate_hits = hit_team_counts(game, result)
        outcome = _classify(solver_rung, parse_failure, enemy_hits, elapsed)

        # M5.1 stall rule: N consecutive PASS_UNREACHABLE from BOTH teams is a
        # genuine stalemate draw, distinct from the turn cap.
        if outcome is ShotOutcome.PASS_UNREACHABLE:
            pass_run += 1
            pass_teams.add(team.team)
        else:
            pass_run = 0
            pass_teams.clear()

        if suppressed:
            match_stats.repeat_suppressed += 1
        else:
            match_stats.shots += 1
            if enemy_hits:
                match_stats.enemy_hit_shots += 1
            match_stats.kills += enemy_hits
            if teammate_hits:
                match_stats.friendly_fire_shots += 1
            shots.append(
                ShotRecord(
                    agent=agent.name,
                    team_id=team.team,
                    expression=expr,
                    parse_failure=parse_failure,
                    result=result,
                    outcome=outcome.value,
                    solver_rung=solver_rung,
                )
            )
        # Outcome counters count every classified turn (they are outcomes,
        # not attempts; the dedupe rule only suppresses attempt recording).
        if outcome is ShotOutcome.PASS_UNREACHABLE:
            match_stats.pass_unreachable += 1
        elif outcome is ShotOutcome.SOLVER_FAILED:
            match_stats.solver_failed += 1
        elif outcome is ShotOutcome.TIMEOUT:
            match_stats.timeouts += 1
        game.state.advance_turn()
        turns += 1
        if pass_run >= _STALL_LIMIT and len(pass_teams) == 2:
            draw_reason = "STALEMATE"
            break

    if draw_reason is None and game.finished():
        draw_reason = None  # decisive
    elif draw_reason is None:
        draw_reason = "TURN_CAP"

    # Merge the agents' internal counters (M3 logging: parse failures / retries
    # the agent itself saw during emission, e.g. RandomAgent's validation loop).
    for agent in agents_by_team.values():
        internal: AgentStats = agent.stats()
        stats[agent.name].parse_failures += internal.parse_failures
        stats[agent.name].retries += internal.retries
        # M5.3: simulate-tool accounting (zero for agents without the wrapper).
        stats[agent.name].simulate_calls += internal.simulate_calls
        stats[agent.name].simulate_denied += internal.simulate_denied

    rung_counts: dict[str, dict[str, int]] = {}
    for agent in agents_by_team.values():
        history = getattr(agent, "rung_history", None)
        if isinstance(history, list):
            rung_counts[agent.name] = dict(Counter(history))

    return MatchResult(
        seed=seed,
        agent_a=agent_a.name,
        agent_b=agent_b.name,
        winner=game.winner(),
        turns=turns,
        stats=stats,
        shots=shots,
        rung_counts=rung_counts,
        draw_reason=draw_reason,
    )


def _peek_solver_rung(agent: Agent) -> str | None:
    """The degradation rung of the just-emitted shot, if the agent records one."""
    history = getattr(agent, "rung_history", None)
    if isinstance(history, list) and history:
        return str(history[-1])
    return None


def _classify(
    solver_rung: str | None,
    parse_failure: bool,
    enemy_hits: int,
    elapsed: float,
) -> ShotOutcome:
    """M5.1 outcome taxonomy for one turn (see eval/metrics.py)."""
    if parse_failure:
        return ShotOutcome.PARSE_ERROR
    if elapsed > _SHOT_TIME_BUDGET:
        return ShotOutcome.TIMEOUT
    if solver_rung == RUNG_PASS_UNREACHABLE:
        return ShotOutcome.PASS_UNREACHABLE
    if solver_rung == RUNG_SOLVER_FAILED:
        return ShotOutcome.SOLVER_FAILED
    return ShotOutcome.HIT if enemy_hits else ShotOutcome.MISS


def make_agent(name: str, seed: int) -> Agent:
    """Build one fresh agent instance from a roster entry.

    Exact-string factory lookup first (``_AGENT_FACTORIES``); otherwise an
    ``"llm:<model>"`` entry constructs :class:`~agents.llm_agent.LLMAgent`
    with ``model=<rest>`` (the part after the FIRST colon; ``seed`` is unused
    — LLM agents are deterministic given the model). The ``llm:`` prefix is
    opt-in: ``DEFAULT_ROSTER`` stays network-free, and a ``llm:`` entry
    raises at construction when no auth env var is set (fail fast, never
    mid-match). Unknown names raise ``ValueError``.
    """
    factory = _AGENT_FACTORIES.get(name)
    if factory is not None:
        return factory(seed)
    if name.startswith("llm:"):
        return LLMAgent(model=name.split(":", 1)[1])
    raise ValueError(f"unknown agent in roster: {name}")


def _make_agents(name_a: str, name_b: str, seed: int) -> tuple[Agent, Agent]:
    """Fresh, match-seeded agent instances (random baselines reproducible)."""
    return make_agent(name_a, seed), make_agent(name_b, seed)


# --- Leaderboard run ---------------------------------------------------------


def run_leaderboard(
    root_seed: int,
    roster: Sequence[str],
    n_matches: int,
    config: MatchConfig,
    out_dir: Path,
) -> Leaderboard:
    """Derive the plan from ``root_seed``, persist it as ``seeds.json``, and run
    every planned match into ``out_dir`` (plots + ``leaderboard.md``)."""
    roster_tuple = tuple(roster)
    plan = build_plan(root_seed, roster_tuple, n_matches)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_file = out / "seeds.json"
    _write_seed_file(seed_file, root_seed, roster_tuple, n_matches, config, plan)
    return _run_plan_to_disk(plan, roster_tuple, config, out, root_seed, n_matches, seed_file)


def run_from_seed_file(seed_file: Path, out_dir: Path) -> Leaderboard:
    """Re-run the evaluation from a saved ``seeds.json`` alone.

    Reads the exact match list and config from the file (no derivation step),
    so re-running reproduces the leaderboard and plots byte-for-byte from the
    committed artifact.
    """
    data = json.loads(Path(seed_file).read_text())
    roster = tuple(data["roster"])
    plan = [PlannedMatch(**m) for m in data["matches"]]
    cfg = MatchConfig(num_soldiers=data["num_soldiers"], max_turns=data["max_turns"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_matches = data["num_matches_per_pair"]
    root_seed = int(data["root_seed"])
    return _run_plan_to_disk(plan, roster, cfg, out, root_seed, n_matches, Path(seed_file))


def _run_plan_to_disk(
    plan: Sequence[PlannedMatch],
    roster: tuple[str, ...],
    cfg: MatchConfig,
    out_dir: Path,
    root_seed: int,
    n_matches: int,
    seed_file: Path,
) -> Leaderboard:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    matches: list[MatchResult] = []
    for i, planned in enumerate(plan):
        agent_a, agent_b = _make_agents(planned.a, planned.b, planned.seed)
        result = play_match(planned.seed, agent_a, agent_b, cfg)
        plot_file = f"match_{i:03d}_seed{planned.seed}.png"
        title = (
            f"{planned.a} (T1) vs {planned.b} (T2) — seed {planned.seed} — "
            f"{team_label(result.winner)} — {result.turns} turns"
        )
        _save_match_plot(out_dir, plots_dir / plot_file, agent_a, agent_b, result, title, cfg)
        result.plot_file = plot_file
        matches.append(result)

    rows = _aggregate(matches, roster)
    leaderboard_file = out_dir / "leaderboard.md"
    leaderboard_file.write_text(
        _render_markdown(root_seed, roster, n_matches, cfg, rows, matches, seed_file)
    )
    return Leaderboard(
        root_seed=root_seed,
        roster=roster,
        num_matches_per_pair=n_matches,
        config=cfg,
        matches=matches,
        rows=rows,
        out_dir=out_dir,
        leaderboard_file=leaderboard_file,
        seed_file=seed_file,
    )


def _save_match_plot(
    out_dir: Path,
    plot_path: Path,
    agent_a: Agent,
    agent_b: Agent,
    result: MatchResult,
    title: str,
    cfg: MatchConfig,
) -> None:
    """Plot the match's final board + all shot trajectories, then save."""
    game = _replay_final_game(result, agent_a, agent_b, cfg)
    shots_with_team = [(s.team_id, s.result) for s in result.shots]
    # render.py returns a bare ``object`` to keep matplotlib out of its public
    # API (project convention); we need the Figure here, so cast to Any.
    fig = cast(Any, render.render_match(game, shots_with_team, title=title))
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _replay_final_game(
    result: MatchResult,
    agent_a: Agent,
    agent_b: Agent,
    cfg: MatchConfig,
) -> Game:
    """Reconstruct the match's final board from its seed + the **actually fired**
    expression of every shot (a failed emission was replaced by the safe dud).

    The board is 100% reproducible from the seed; re-firing replays every kill
    so the plotted final state shows the survivors.
    """
    game = Game.create(result.seed, num_teams=2, num_soldiers=cfg.num_soldiers)
    for shot in result.shots:
        game.fire("0*x" if shot.parse_failure else shot.expression)
        game.state.advance_turn()
    return game


def _aggregate(matches: Sequence[MatchResult], roster: Sequence[str]) -> list[AgentLeaderRow]:
    """Roll per-match stats up into one :class:`AgentLeaderRow` per agent, and
    count wins/draws (a draw at the turn cap counts for neither side)."""
    rows = {name: AgentLeaderRow(agent=name) for name in roster}
    for match in matches:
        winner_agent = (
            match.agent_a
            if match.winner == TEAM1
            else match.agent_b
            if match.winner == TEAM2
            else None
        )
        for name, match_stats in match.stats.items():
            rows[name].merge(match_stats)
            if winner_agent == name:
                rows[name].wins += 1
    return [rows[name] for name in roster]


def _write_seed_file(
    path: Path,
    root_seed: int,
    roster: tuple[str, ...],
    n_matches: int,
    cfg: MatchConfig,
    plan: Sequence[PlannedMatch],
) -> None:
    """Persist the exact plan (the seed file alone reproduces the leaderboard)."""
    payload = {
        "root_seed": root_seed,
        "roster": list(roster),
        "num_matches_per_pair": n_matches,
        "num_soldiers": cfg.num_soldiers,
        "max_turns": cfg.max_turns,
        "matches": [asdict(m) for m in plan],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


# --- Leaderboard markdown ----------------------------------------------------


def _render_markdown(
    root_seed: int,
    roster: tuple[str, ...],
    n_matches: int,
    cfg: MatchConfig,
    rows: Sequence[AgentLeaderRow],
    matches: Sequence[MatchResult],
    seed_file: Path,
) -> str:
    lines: list[str] = ["# Graphwar agent leaderboard", ""]
    n_total = len(matches)
    lines.append(
        f"- config: root_seed={root_seed}, matches/pair={n_matches} x 2 sides "
        f"({n_total} matches), num_soldiers={cfg.num_soldiers}, "
        f"max_turns={cfg.max_turns}"
    )
    lines.append(f"- roster: {', '.join(roster)}")
    lines.append(
        f"- seeds file: `{seed_file.name}` — rerun via `python3 -m eval --from-seeds <file>`"
    )
    lines.append(
        "- metrics (M5.1 taxonomy): win rate + hit rate, with every turn classified "
        "HIT | MISS | PASS_UNREACHABLE | SOLVER_FAILED | PARSE_ERROR | TIMEOUT. "
        "PASS_UNREACHABLE turns are passes, not shots, and never count against hit "
        "rate. Draws are `TURN_CAP` or `STALEMATE` (consecutive PASS_UNREACHABLE "
        "from both teams). Duplicate attempts are suppressed into `Repeats`."
    )
    lines.append("")

    lines.append("## Win rate & hit rate")
    lines.append("")
    lines.append(
        "| Agent | Matches | Wins | Win rate | Shots | Hit rate | Kills | "
        "Friendly fire | Passes | SolverFail | Parse | Timeout | Repeats | Retries |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append("| " + " | ".join(row.as_row()) + " |")
    lines.append("")

    draws_matches = [m for m in matches if m.winner is None]
    cap_draws = sum(1 for m in draws_matches if m.draw_reason == "TURN_CAP")
    stall_draws = sum(1 for m in draws_matches if m.draw_reason == "STALEMATE")
    lines.append(f"- draws: {len(draws_matches)} ({cap_draws} turn-cap, {stall_draws} stalemate)")
    lines.append("")

    lines.append("## Head-to-head")
    lines.append("")
    lines.append("| Pair | A wins | B wins | Draws |")
    lines.append("|---|---|---|---|")
    for a, b in combinations(roster, 2):
        a_wins = sum(
            1
            for m in matches
            if {m.agent_a, m.agent_b} == {a, b}
            and ((m.winner == TEAM1 and m.agent_a == a) or (m.winner == TEAM2 and m.agent_b == a))
        )
        b_wins = sum(
            1
            for m in matches
            if {m.agent_a, m.agent_b} == {a, b}
            and ((m.winner == TEAM1 and m.agent_a == b) or (m.winner == TEAM2 and m.agent_b == b))
        )
        draws = sum(1 for m in matches if {m.agent_a, m.agent_b} == {a, b} and m.winner is None)
        lines.append(f"| {a} vs {b} | {a_wins} | {b_wins} | {draws} |")
    lines.append("")

    solver_rungs = _solver_rung_summary(matches)
    if solver_rungs:
        lines.append("## Solver degradation rungs (informational)")
        lines.append("")
        for agent_name, rung_counts in solver_rungs.items():
            desc = ", ".join(f"{k}={v}" for k, v in rung_counts.items())
            lines.append(f"- {agent_name}: {desc}")
        lines.append("")

    lines.append("## Per-match log")
    lines.append("")
    lines.append(
        "| # | Seed | TEAM1 | TEAM2 | Winner | Turns | TEAM1 shots | "
        "TEAM1 hits | TEAM2 shots | TEAM2 hits | Plot |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, m in enumerate(matches):
        t1_stats = m.stats.get(m.agent_a)
        t2_stats = m.stats.get(m.agent_b)
        lines.append(
            f"| {i} | {m.seed} | {m.agent_a} | {m.agent_b} | {_match_label(m)} | "
            f"{m.turns} | {t1_stats.shots if t1_stats else 0} | "
            f"{t1_stats.enemy_hit_shots if t1_stats else 0} | "
            f"{t2_stats.shots if t2_stats else 0} | "
            f"{t2_stats.enemy_hit_shots if t2_stats else 0} | "
            f"[plot](plots/{m.plot_file}) |"
        )
    lines.append("")
    return "\n".join(lines)


def _match_label(m: MatchResult) -> str:
    """Winner cell: team name, or the specific draw reason (M5.1)."""
    if m.winner is not None:
        return team_label(m.winner)
    if m.draw_reason == "STALEMATE":
        return "draw (stalemate)"
    return "draw (turn cap)"


def _solver_rung_summary(
    matches: Sequence[MatchResult],
) -> dict[str, dict[str, int]]:
    """Roll the solver agents' per-shot degradation rungs across matches."""
    totals: dict[str, Counter[str]] = {}
    for m in matches:
        for agent_name, counts in m.rung_counts.items():
            totals.setdefault(agent_name, Counter()).update(counts)
    return {name: dict(counter) for name, counter in totals.items()}


__all__ = [
    "DEFAULT_ROSTER",
    "Leaderboard",
    "MatchConfig",
    "MatchResult",
    "PlannedMatch",
    "ShotRecord",
    "build_plan",
    "make_agent",
    "play_match",
    "run_from_seed_file",
    "run_leaderboard",
]
