"""Match-level and leaderboard-level metrics (Phase 4 / M4).

The minimal slice reports **win rate** and **hit rate** (IMPLEMENTATION_PLAN.md
Phase 4). The per-agent row also carries the raw counters the rates are derived
from, plus the M3 parse-failure / retry logging — all recorded per match by the
runner and rolled up here. Token cost and ablations are explicitly out of scope.

Definitions
-----------
- win rate: matches won / matches played (a draw at the turn cap counts for
  neither agent).
- hit rate: shots that hit at least one enemy soldier / shots fired.
- kills: distinct enemy soldiers struck (each hit kills, multi-kill allowed).
"""

from __future__ import annotations

from dataclasses import dataclass

from graphwar_sim import config


@dataclass
class AgentMatchStats:
    """One agent's per-match counters (built by the runner's turn loop)."""

    agent: str
    shots: int = 0
    enemy_hit_shots: int = 0  # shots that hit >= 1 enemy
    kills: int = 0  # enemy soldiers struck (distinct, each dies)
    friendly_fire_shots: int = 0
    parse_failures: int = 0  # emissions that failed to parse (logged, M3)
    retries: int = 0  # re-samples after a parse failure (logged, M3)


@dataclass
class AgentLeaderRow:
    """Roll-up of :class:`AgentMatchStats` across a leaderboard run."""

    agent: str
    matches: int = 0
    wins: int = 0
    shots: int = 0
    enemy_hit_shots: int = 0
    kills: int = 0
    friendly_fire_shots: int = 0
    parse_failures: int = 0
    retries: int = 0

    @property
    def win_rate(self) -> float:
        """Matches won / matches played; 0.0 for an agent with no matches."""
        return self.wins / self.matches if self.matches else 0.0

    @property
    def hit_rate(self) -> float:
        """Enemy-hit shots / shots fired; 0.0 for an agent that never fired."""
        return self.enemy_hit_shots / self.shots if self.shots else 0.0

    def merge(self, other: AgentMatchStats) -> None:
        """Add one match's stats into the roll-up."""
        self.matches += 1
        self.shots += other.shots
        self.enemy_hit_shots += other.enemy_hit_shots
        self.kills += other.kills
        self.friendly_fire_shots += other.friendly_fire_shots
        self.parse_failures += other.parse_failures
        self.retries += other.retries

    def as_row(self) -> tuple[str, ...]:
        """Markdown-ready table row (win rate + hit rate + raw counters)."""
        return (
            self.agent,
            str(self.matches),
            str(self.wins),
            f"{self.win_rate:.3f}",
            str(self.shots),
            f"{self.hit_rate:.3f}",
            str(self.kills),
            str(self.friendly_fire_shots),
            str(self.parse_failures),
            str(self.retries),
        )


def team_label(team_id: int | None) -> str:
    """Human label for a winner (``TEAM1``/``TEAM2``) or a draw."""
    if team_id == config.TEAM1:
        return "TEAM1"
    if team_id == config.TEAM2:
        return "TEAM2"
    return "draw"


__all__ = ["AgentLeaderRow", "AgentMatchStats", "team_label"]
