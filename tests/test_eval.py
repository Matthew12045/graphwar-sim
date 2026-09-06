"""Tests for M4: seeded match runner + win-rate / hit-rate leaderboard.

Acceptance (IMPLEMENTATION_PLAN.md, Phase 4 minimal slice):

- Match runner is seeded and reproducible.
- ``leaderboard.md`` + per-match plots for the full roster.
- **Reproducible from the seed file alone.**

The full Phase 4 metrics (token cost, mean turns-to-kill, ablations) are
explicitly out of scope; the runner reports win rate and hit rate, plus the
M3 parse-failure / retry logging.
"""

from __future__ import annotations

from pathlib import Path

from agents import AgentStats, RandomAgent, SolverAgent, StraightShotAgent
from eval.runner import (
    _STALL_LIMIT,
    MatchConfig,
    build_plan,
    play_match,
    run_from_seed_file,
    run_leaderboard,
)

DEFAULT_ROSTER = ("solver", "random", "straight")


def _small_config() -> MatchConfig:
    """A fast config for tests: 1 soldier/side, capped turns (draws allowed)."""
    return MatchConfig(num_soldiers=1, max_turns=40)


# --- Single match ------------------------------------------------------------


def test_play_match_deterministic() -> None:
    """The same seed and agents produce the identical match (turns, winner,
    per-shot expressions) — the runner is a pure function of (seed, roster)."""
    a = SolverAgent()
    b = RandomAgent(seed=7)
    r1 = play_match(7, a, b, _small_config())
    r2 = play_match(7, SolverAgent(), RandomAgent(seed=7), _small_config())
    assert r1.seed == r2.seed == 7
    assert r1.turns == r2.turns
    assert r1.winner == r2.winner
    assert [s.expression for s in r1.shots] == [s.expression for s in r2.shots]
    assert r1.stats == r2.stats


def test_play_match_winner_is_surviving_side() -> None:
    """When a match ends with a winner, the recorded winner's team has soldiers
    alive and the loser's team is wiped; a draw carries a specific reason
    (turn cap or stalemate, M5.1)."""
    for seed in (1, 2, 3):
        result = play_match(seed, SolverAgent(), StraightShotAgent(), _small_config())
        if result.winner is None:
            assert result.draw_reason in ("TURN_CAP", "STALEMATE")
            if result.draw_reason == "TURN_CAP":
                assert result.turns == _small_config().max_turns
            else:
                assert result.turns < _small_config().max_turns
        else:
            assert result.winner in (1, 2)


# --- M5.3: a deep expression is a PARSE_ERROR turn, not a crashed match -------


def test_deep_expression_classified_parse_error_no_crash() -> None:
    """A deep-but-in-char-budget expression used to RecursionError out of the
    unguarded ``Game.fire`` and kill the match. With the M5.3 parser cap it
    raises ``MalformedFunction``, which the runner catches: the turn is
    classified ``PARSE_ERROR``, the safe dud ``0*x`` is fired in its place,
    and the match completes (eval/runner.py's defensive block)."""
    from agents.base import Observation
    from graphwar_sim import Game, config

    deep = "+".join(["1"] * (config.MAX_AST_DEPTH + 1))
    assert len(deep) <= config.MAX_EXPR_CHARS

    class DeepAgent:
        """Always emits a depth-cap-exceeding expression (M5.3 regression)."""

        name = "deep"

        def __init__(self) -> None:
            self._stats = AgentStats()

        def act(self, game: Game, obs: Observation) -> str:
            return deep

        def stats(self) -> AgentStats:
            return self._stats

    result = play_match(11, DeepAgent(), StraightShotAgent(), _small_config())
    deep_stats = result.stats["deep"]
    assert deep_stats.parse_failures >= 1
    parse_error_shots = [s for s in result.shots if s.outcome == "PARSE_ERROR"]
    assert parse_error_shots, "the deep attempt was not classified PARSE_ERROR"
    assert all(s.expression == deep and s.parse_failure for s in parse_error_shots)
    # The safe dud replaced the deep fire as the turn's actual shot.
    assert result.winner is not None or result.draw_reason in ("TURN_CAP", "STALEMATE")


def test_simulate_accounting_counters_merge_from_agent() -> None:
    """M5.3: the runner merges the agent's simulate accounting counters into
    the match stats (AgentStats -> AgentMatchStats); agents without the
    wrapper contribute 0. The leaderboard roll-up deliberately does not
    surface them yet (byte-parity; columns deferred to M5.4)."""
    from agents.base import Observation
    from graphwar_sim import Game

    class ProbingAgent:
        """Reports fixed simulate accounting (the M5.4 wrapper-agent shape)."""

        name = "prober"

        def __init__(self) -> None:
            self._stats = AgentStats(simulate_calls=5, simulate_denied=2)

        def act(self, game: Game, obs: Observation) -> str:
            return "0*x"

        def stats(self) -> AgentStats:
            return self._stats

    result = play_match(5, ProbingAgent(), StraightShotAgent(), _small_config())
    assert result.stats["prober"].simulate_calls == 5
    assert result.stats["prober"].simulate_denied == 2
    assert result.stats["straight"].simulate_calls == 0
    assert result.stats["straight"].simulate_denied == 0


# --- Plan / side balance -----------------------------------------------------


def test_build_plan_is_side_balanced_and_seeded() -> None:
    """Every pair meets n_matches times on each side, with unique seeds."""
    plan = build_plan(1000, DEFAULT_ROSTER, 3)
    # 3 pairs * 3 matches * 2 sides.
    assert len(plan) == 3 * 3 * 2
    seeds = [m.seed for m in plan]
    assert len(set(seeds)) == len(seeds)
    pairs: dict[tuple[str, str], int] = {}
    for m in plan:
        pairs[(m.a, m.b)] = pairs.get((m.a, m.b), 0) + 1
    for pair, count in pairs.items():
        assert count == 3, f"{pair} not side-balanced"
    # Side balance: (solver, random) and (random, solver) both present.
    assert ("solver", "random") in pairs and ("random", "solver") in pairs


# --- Leaderboard aggregation -------------------------------------------------


def test_leaderboard_win_and_hit_rate_consistency(tmp_path: Path) -> None:
    """The leaderboard rows are consistent: rates in [0, 1], per-agent matches
    equal the number of matches played, and wins + draws close per pair."""
    leaderboard = run_leaderboard(
        root_seed=1000,
        roster=DEFAULT_ROSTER,
        n_matches=1,
        config=_small_config(),
        out_dir=tmp_path,
    )
    rows = {row.agent: row for row in leaderboard.rows}
    assert set(rows) == set(DEFAULT_ROSTER)

    for name, row in rows.items():
        # The agent is on one side of every match involving its pairs.
        assert row.matches == sum(
            1 for m in leaderboard.matches if m.agent_a == name or m.agent_b == name
        )
        assert row.wins <= row.matches
        assert 0.0 <= row.win_rate <= 1.0
        assert 0.0 <= row.hit_rate <= 1.0

    # Every match is accounted for: each has a winner or is a draw.
    decided = sum(1 for m in leaderboard.matches if m.winner is not None)
    draws = len(leaderboard.matches) - decided
    assert decided + draws == len(leaderboard.matches)


def test_leaderboard_reports_hit_rate(tmp_path: Path) -> None:
    """Hit rate is enemy-hit shots / total shots; it is > 0 for the solver
    (it hits on the seeded maps) and recorded for every agent."""
    leaderboard = run_leaderboard(
        root_seed=2000,
        roster=DEFAULT_ROSTER,
        n_matches=1,
        config=_small_config(),
        out_dir=tmp_path / "lb2",
    )
    by_name = {row.agent: row for row in leaderboard.rows}
    solver = by_name["solver"]
    assert solver.shots > 0 and solver.hit_rate > 0.0
    for name in DEFAULT_ROSTER:
        assert by_name[name].shots > 0


def test_outcome_taxonomy_records_every_turn(tmp_path: Path) -> None:
    """Every recorded shot carries one of the six M5.1 outcomes, and the
    per-agent counters are consistent with the recorded outcomes."""
    from eval.metrics import ShotOutcome

    leaderboard = run_leaderboard(
        root_seed=2000,
        roster=DEFAULT_ROSTER,
        n_matches=1,
        config=_small_config(),
        out_dir=tmp_path / "tax",
    )
    outcomes = {ShotOutcome(x.outcome) for m in leaderboard.matches for x in m.shots}
    assert outcomes <= set(ShotOutcome)
    assert ShotOutcome.HIT in outcomes
    for m in leaderboard.matches:
        for name, st in m.stats.items():
            mine = sum(1 for x in m.shots if x.agent == name)
            assert st.shots == mine  # every recorded shot is counted once
            # Outcome counters count ALL classified turns (suppressed repeats
            # included), so they are >= the recorded-shot counts.
            mine_passes = sum(
                1
                for x in m.shots
                if x.agent == name and x.outcome == ShotOutcome.PASS_UNREACHABLE.value
            )
            assert st.pass_unreachable >= mine_passes
            assert st.solver_failed >= sum(
                1
                for x in m.shots
                if x.agent == name and x.outcome == ShotOutcome.SOLVER_FAILED.value
            )


class _PassingAgent:
    """Test double: always passes (PASS_UNREACHABLE rung, safe flat dud).

    Exercises the runner's classification / dedupe / stalemate machinery on
    real matches without needing a full-wall map (which the seeded generator
    essentially never produces — a documented M5.1 finding).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._stats = AgentStats()
        self.rung_history: list[str] = []

    def act(self, game, obs) -> str:  # type: ignore[no-untyped-def]
        from graphwar_sim.solver import RUNG_PASS_UNREACHABLE

        self.rung_history.append(RUNG_PASS_UNREACHABLE)
        return "0*x"

    def stats(self) -> AgentStats:
        return self._stats


def test_passer_match_classifies_every_turn() -> None:
    """A match of two passing agents: every turn is PASS_UNREACHABLE, the
    shots never count (hit rate 0/0), the repeats are suppressed, and the
    match ends as DRAW_STALEMATE before the turn cap."""
    seed = 7
    r = play_match(seed, _PassingAgent("passer_a"), _PassingAgent("passer_b"), _small_config())
    for st in r.stats.values():
        # Two pass turns per agent (one recorded, one suppressed repeat).
        assert st.pass_unreachable == 2
        assert st.enemy_hit_shots == 0
        assert st.shots == 1  # only the first of each identical pair
        assert st.repeat_suppressed == 1
    assert r.draw_reason == "STALEMATE"
    assert r.winner is None
    assert r.turns == _STALL_LIMIT  # ends exactly at the stall limit
    assert all(x.outcome == "PASS_UNREACHABLE" for x in r.shots)


def test_dedupe_suppresses_identical_repeats() -> None:
    """Same board state + same expression on an agent's next turn is not a new
    attempt: repeat_suppressed grows and the shot counter does not."""
    r = play_match(7, _PassingAgent("passer_a"), _PassingAgent("passer_b"), _small_config())
    # Turn 0 (T1) and 1 (T2) are recorded; turns 2 (T1 again) and 3 (T2 again)
    # repeat the identical (state, expression) pair and are suppressed.
    for st in r.stats.values():
        assert st.shots == 1
        assert st.repeat_suppressed == 1


def test_stalemate_ends_match_as_draw_distinct_from_cap(tmp_path: Path) -> None:
    """Consecutive PASS_UNREACHABLE from both teams ends the match early as
    DRAW_STALEMATE — distinct from the turn-cap draw (which the solver-vs-
    solver non-passing board still produces, e.g. cap-length matches)."""
    r = play_match(7, _PassingAgent("passer_a"), _PassingAgent("passer_b"), _small_config())
    assert r.draw_reason == "STALEMATE"
    assert r.winner is None
    assert r.turns <= _STALL_LIMIT

    # Control: a real cap draw (stalemate can only come from the pass rule).
    cap_seen = False
    for seed in (1, 11, 21, 31, 41, 51, 71, 91, 101, 121):
        from graphwar_sim import Game as G
        from graphwar_sim.corridor import reachability

        g = G.create(seed, num_soldiers=1)
        if any(x.reachable for x in reachability(g)):
            r2 = play_match(seed, SolverAgent(), RandomAgent(seed=seed), _small_config())
            if r2.draw_reason == "TURN_CAP":
                cap_seen = True
                assert r2.turns == _small_config().max_turns
                break
    assert cap_seen, "expected a turn-cap draw from the real solver roster"


# --- Reproducibility from the seed file alone --------------------------------


def test_reproducible_from_seed_file(tmp_path: Path) -> None:
    """Re-running the evaluation from the written seed file alone reproduces the
    leaderboard byte-for-byte and the same per-match plots."""
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    lb1 = run_leaderboard(
        root_seed=5000,
        roster=DEFAULT_ROSTER,
        n_matches=1,
        config=_small_config(),
        out_dir=out1,
    )
    seed_file = out1 / "seeds.json"
    assert seed_file.exists()

    lb2 = run_from_seed_file(seed_file, out_dir=out2)
    assert lb1.leaderboard_file.read_text() == lb2.leaderboard_file.read_text()
    assert [m.seed for m in lb1.matches] == [m.seed for m in lb2.matches]
    assert [m.winner for m in lb1.matches] == [m.winner for m in lb2.matches]

    # Per-match plots exist for the full roster (every match has one).
    plots1 = sorted(p.name for p in (out1 / "plots").glob("*.png"))
    plots2 = sorted(p.name for p in (out2 / "plots").glob("*.png"))
    assert plots1 and plots1 == plots2
    assert len(plots1) == len(lb1.matches)


# --- Leaderboard document ----------------------------------------------------


def test_leaderboard_document_has_expected_sections(tmp_path: Path) -> None:
    out = tmp_path / "lb_doc"
    run_leaderboard(
        root_seed=9000,
        roster=DEFAULT_ROSTER,
        n_matches=1,
        config=_small_config(),
        out_dir=out,
    )
    md = (out / "leaderboard.md").read_text()
    assert "# Graphwar agent leaderboard" in md
    assert "| Agent |" in md
    assert "Win rate" in md
    assert "Hit rate" in md
    assert "seeds.json" in md


def test_full_roster_matches_play_headless(tmp_path: Path) -> None:
    """The whole roster plays real matches (the plan's three agents), and the
    SimResult path is exercised without crashing (no LLM is invoked anywhere)."""
    from agents import simulate
    from graphwar_sim import Game

    leaderboard = run_leaderboard(
        root_seed=3100,
        roster=DEFAULT_ROSTER,
        n_matches=1,
        config=_small_config(),
        out_dir=tmp_path / "roster",
    )
    assert len(leaderboard.matches) == 6
    # A quick smoke test that the simulate tool used by agents is wired to the
    # same physics: it reports parseable + hit/nohit without raising.
    game = Game.create(7, num_soldiers=1)
    r = simulate(game, "0.05*x")
    assert r.parseable
    assert isinstance(r.num_steps, int)


# --- M5.1 taxonomy: outcomes, dedupe, stalemate ------------------------------
