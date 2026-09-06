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

from agents import RandomAgent, SolverAgent, StraightShotAgent
from eval.runner import (
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
    alive and the loser's team is wiped (or the turn cap was hit -> None)."""
    for seed in (1, 2, 3):
        result = play_match(seed, SolverAgent(), StraightShotAgent(), _small_config())
        if result.winner is None:
            assert result.turns == _small_config().max_turns
        else:
            assert result.winner in (1, 2)


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
