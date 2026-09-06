"""M4: evaluation harness (seeded match runner + leaderboard).

Minimal viable slice (IMPLEMENTATION_PLAN.md Phase 4): a seeded, reproducible
match runner that pits agents head-to-head, reports **win rate** and
**hit rate** (plus the M3 parse-failure / retry logging), writes
``leaderboard.md`` + per-match plots to ``eval/results/``, and is reproducible
**from the seed file alone** (``run_from_seed_file``).

Public surface:
    - :class:`MatchConfig`, :class:`MatchResult`, :class:`ShotRecord`
    - :func:`build_plan`, :func:`play_match`, :func:`run_leaderboard`,
      :func:`run_from_seed_file`
    - :class:`AgentMatchStats`, :class:`AgentLeaderRow` (win/hit rate)
"""

from __future__ import annotations

from .metrics import AgentLeaderRow, AgentMatchStats
from .runner import (
    DEFAULT_ROSTER,
    MatchConfig,
    MatchResult,
    ShotRecord,
    build_plan,
    play_match,
    run_from_seed_file,
    run_leaderboard,
)

__all__ = [
    "AgentLeaderRow",
    "AgentMatchStats",
    "DEFAULT_ROSTER",
    "MatchConfig",
    "MatchResult",
    "ShotRecord",
    "build_plan",
    "play_match",
    "run_from_seed_file",
    "run_leaderboard",
]
