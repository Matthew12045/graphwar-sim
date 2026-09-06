"""Match-level and leaderboard-level metrics (M5.1).

The outcome taxonomy replaces the old single ``dud`` bucket. Every turn is
classified into exactly one :class:`ShotOutcome`:

- ``HIT`` — fired and struck at least one enemy soldier.
- ``MISS`` — fired and struck nobody (a teammate graze is still a MISS, and is
  additionally counted as friendly fire).
- ``PASS_UNREACHABLE`` — the M5.1 corridor pre-check proved no clean monotone
  trajectory can reach any enemy; the turn is a pass, NOT a shot, and never
  counts against hit rate.
- ``SOLVER_FAILED`` — the corridor found a reachable target but the
  degradation ladder could not convert it (a fit gap, distinct from
  unreachable).
- ``PARSE_ERROR`` — the emission failed to parse; the safe dud was fired in
  its place.
- ``TIMEOUT`` — the trajectory evaluation exceeded the harness time budget.

Attempt deduplication (M5.1): when the board state (positions, alive flags,
current shooter) is unchanged from the last recorded attempt of the same
agent and the agent emits an identical expression, no new attempt is recorded
— an internal ``repeat_suppressed`` counter is incremented instead.

Definitions
-----------
- win rate: matches won / matches played (a draw counts for neither agent).
- hit rate: shots that hit at least one enemy soldier / shots fired.
  ``PASS_UNREACHABLE`` turns are not shots and do not count against hit rate.
- kills: distinct enemy soldiers struck (each hit kills, multi-kill allowed).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from graphwar_sim import config


class ShotOutcome(StrEnum):
    """Outcome taxonomy for one turn (replaces the M2 ``dud`` bucket)."""

    HIT = "HIT"
    MISS = "MISS"
    PASS_UNREACHABLE = "PASS_UNREACHABLE"
    SOLVER_FAILED = "SOLVER_FAILED"
    PARSE_ERROR = "PARSE_ERROR"
    TIMEOUT = "TIMEOUT"


@dataclass
class AgentMatchStats:
    """One agent's per-match counters (built by the runner's turn loop)."""

    agent: str
    shots: int = 0  # fired attempts (HIT + MISS + SOLVER_FAILED + PARSE_ERROR + TIMEOUT)
    enemy_hit_shots: int = 0  # shots that hit >= 1 enemy
    kills: int = 0  # enemy soldiers struck (distinct, each dies)
    friendly_fire_shots: int = 0
    parse_failures: int = 0  # emissions that failed to parse (logged, M3)
    retries: int = 0  # re-samples after a parse failure (logged, M3)
    # M5.3 simulate-tool accounting (merged from the agent's AgentStats; 0 for
    # agents that never call the oracle — the M3 roster). Surfaced in the
    # leaderboard only from M5.4, when a simulate-consuming agent exists
    # (byte-parity: AgentLeaderRow deliberately does not aggregate these yet).
    simulate_calls: int = 0  # delegated simulate-tool calls (wrapper-using agents)
    simulate_denied: int = 0  # over-budget calls the wrapper refused
    # M5.1 taxonomy counters (turn outcomes / suppression, not attempts).
    pass_unreachable: int = 0  # corridor-proven unreachable turns
    solver_failed: int = 0  # reachable but the ladder could not convert it
    timeouts: int = 0  # trajectory evaluation exceeded the time budget
    repeat_suppressed: int = 0  # duplicate attempts not recorded (M5.1 dedupe)


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
    pass_unreachable: int = 0
    solver_failed: int = 0
    timeouts: int = 0
    repeat_suppressed: int = 0

    @property
    def win_rate(self) -> float:
        """Matches won / matches played; 0.0 for an agent with no matches."""
        return self.wins / self.matches if self.matches else 0.0

    @property
    def hit_rate(self) -> float:
        """Enemy-hit shots / shots fired; 0.0 for an agent that never fired.

        ``PASS_UNREACHABLE`` turns are not shots and do not count here (M5.1).
        """
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
        self.pass_unreachable += other.pass_unreachable
        self.solver_failed += other.solver_failed
        self.timeouts += other.timeouts
        self.repeat_suppressed += other.repeat_suppressed

    def as_row(self) -> tuple[str, ...]:
        """Markdown-ready table row (rates + raw counters + M5.1 taxonomy)."""
        return (
            self.agent,
            str(self.matches),
            str(self.wins),
            f"{self.win_rate:.3f}",
            str(self.shots),
            f"{self.hit_rate:.3f}",
            str(self.kills),
            str(self.friendly_fire_shots),
            str(self.pass_unreachable),
            str(self.solver_failed),
            str(self.parse_failures),
            str(self.timeouts),
            str(self.repeat_suppressed),
            str(self.retries),
        )


def team_label(team_id: int | None) -> str:
    """Human label for a winner (``TEAM1``/``TEAM2``) or a draw."""
    if team_id == config.TEAM1:
        return "TEAM1"
    if team_id == config.TEAM2:
        return "TEAM2"
    return "draw"


__all__ = ["AgentLeaderRow", "AgentMatchStats", "ShotOutcome", "team_label"]
