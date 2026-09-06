"""Agent protocol, observation frame, and shared stats (Phase 3 / M3).

This module defines the seam the match runner plays against:

- :class:`Agent` — the protocol every agent implements. ``act(game, obs)``
  returns a Graphwar ``y = f(x)`` expression string in the **centered world
  frame** described by the :class:`Observation`; ``stats()`` exposes the
  agent's own parse-failure / retry counters (the M3 logging requirement).
- :class:`Observation` — the board as seen in the **centered world frame**
  (``x`` in ``[-25, 25]``, ``y`` up — the space where ``f`` is evaluated),
  mirrored so the current shooter always faces right. This is the frame
  contract: an expression the agent emits is evaluated by the game in exactly
  this frame (see the frame round-trip test in ``tests/test_agents.py``).
- :func:`hit_team_counts` — classify a shot's hits into enemy / teammate.

Design notes
------------
The :class:`Agent` protocol deliberately passes **both** the raw :class:`Game`
(the deterministic :class:`SolverAgent` needs it to call ``solve``) and the
pre-built :class:`Observation` (the baselines read enemy positions from it).
The runner builds one observation per turn and hands it to both, so the frame
the agent sees is exactly the frame the physics integrates. Frame confusion is
the highest-risk M3 integration point — see ``IMPLEMENTATION_PLAN.md`` Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from graphwar_sim import TEAM1, TEAM2, Game
from graphwar_sim.config import PLANE_GAME_LENGTH, PLANE_HEIGHT, PLANE_LENGTH
from graphwar_sim.physics import ShotResult

# The observation frame label. Agents must emit expressions evaluated in this
# frame (centered world, shooter facing right / enemies at positive x).
FRAME_CENTERED_WORLD = "centered_world_shooter_facing"


@dataclass(frozen=True)
class Observation:
    """The board in the centered world frame, prepared for an agent.

    All coordinates are **world** coordinates: ``x`` in ``[-25, 25]``,
    ``y`` positive up, computed with the same transform the physics uses
    (GROUND_TRUTH.md §1.3, Function.java:193-194) and with the TEAM2 mirror
    applied (Function.java:188-191), so the current shooter is always the
    left-most point and enemies lie at positive ``x``. ``f`` is evaluated in
    exactly this frame, and the auto vertical offset forces the fired curve
    through ``shooter`` (Function.java:206).
    """

    frame: str
    turn_index: int
    team_id: int
    # (x, y) of the muzzle in world coords.
    shooter: tuple[float, float]
    # Alive allies incl. the shooter, world coords.
    own_soldiers: tuple[tuple[float, float], ...]
    # Alive enemy soldiers, world coords. All at x > shooter.x.
    enemy_soldiers: tuple[tuple[float, float], ...]
    # Coarse grid of terrain-blocked world cells (agent-friendly terrain shape).
    terrain_blocks: tuple[tuple[float, float], ...]
    # ASCII map of the same frame: 'M' shooter, 'S' ally, 'E' enemy, '#' terrain.
    ascii_board: str


@dataclass
class AgentStats:
    """The agent's own counters, logged per match by the runner (M3 logging).

    The runner separately counts shots / hits (it owns the physics oracle);
    these fields are the agent-internal record of the emission loop. For the
    three M3 agents candidates are generated from a grammar that is
    guaranteed-parseable, so both counters are 0 in practice — their presence
    is the observable record that the (defensive) validation loop ran.
    """

    parse_failures: int = 0
    retries: int = 0


@runtime_checkable
class Agent(Protocol):
    """A Graphwar player. ``act`` returns a legal ``y = f(x)`` expression."""

    name: str

    def act(self, game: Game, obs: Observation) -> str:
        """Emit the expression for the current turn (centered world frame)."""
        ...

    def stats(self) -> AgentStats:
        """The agent's accumulated parse-failure / retry counters."""
        ...


def hit_team_counts(game: Game, result: ShotResult) -> tuple[int, int]:
    """Classify a shot's hits: ``(enemy_hits, teammate_hits)``.

    ``result.hits`` are ``(player_index, soldier_index, pos)`` (physics.py);
    ``player_index`` indexes ``game.state.teams``. A hit is a *teammate* hit
    when the struck soldier belongs to the current shooter's side — the
    current-turn soldier itself is never hit (physics.py:252-262) but a second
    soldier of the same side can be (multi-kill, no break).
    """
    shooter_team = game.state.current_team().team
    enemy = 0
    teammate = 0
    for player_index, _soldier_index, _pos in result.hits:
        if game.state.teams[player_index].team == shooter_team:
            teammate += 1
        else:
            enemy += 1
    return enemy, teammate


def world_coords(
    px: float,
    py: float,
    mirrored: bool,
    length: int = PLANE_LENGTH,
    height: int = PLANE_HEIGHT,
    game_length: int = PLANE_GAME_LENGTH,
) -> tuple[float, float]:
    """Plane/pixel -> centered world, applying the TEAM2 mirror if ``mirrored``.

    Faithful to the muzzle transform in the physics (Function.java:188-194;
    see also ``solver._plane_to_world``). Centralised here so the observation
    and the round-trip tests agree on one transform.
    """
    if mirrored:
        px = length - px
    x = game_length * (px - length / 2.0) / length
    y = game_length * (-py + height / 2.0) / length
    return x, y


def plane_from_world(
    wx: float,
    wy: float,
    mirrored: bool,
    length: int = PLANE_LENGTH,
    height: int = PLANE_HEIGHT,
    game_length: int = PLANE_GAME_LENGTH,
) -> tuple[float, float]:
    """Inverse of :func:`world_coords` (GROUND_TRUTH.md §1.4). Plane y is down."""
    px = length * wx / game_length + length / 2.0
    py = -length * wy / game_length + height / 2.0
    if mirrored:
        px = length - px
    return px, py


def same_side(team_id: int, other: int) -> bool:
    """True when ``team_id`` and ``other`` are the same side."""
    return team_id == other


__all__ = [
    "FRAME_CENTERED_WORLD",
    "Agent",
    "AgentStats",
    "Observation",
    "hit_team_counts",
    "plane_from_world",
    "same_side",
    "world_coords",
    "TEAM1",
    "TEAM2",
]
