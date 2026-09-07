"""The simulate tool: fire a candidate expression through the real physics.

This is the Phase-3 ``simulate(state, expr)`` driver (the sketch
``3._Give_it_a_simulate_tool.py`` used ``sympify`` — banned in the hot path —
so the *structure* is kept but the math is the faithful ported parser and
integrator). It is a **pure oracle**: it does not apply kills (unlike
``Game.fire``), so an agent can probe a candidate before committing to it.

M5.4 miss telemetry: a binary ``hit_enemy`` hides WHY a shot missed — the
live diagnosis (2026-09-07) caught the LLM probing curves that were 0.04-0.09
world units on-target at the enemy's x while dying 20-34 world units short
(plane-band kills, terrain walls at the muzzle). The telemetry fields turn a
blind re-roll into a gradient: ``nearest_miss`` / ``miss_direction`` say how
far and which way, ``stopped_at_x`` / ``stop_reason`` say where and why it
ended. Computed post-hoc from the returned trajectory — the physics module
stays byte-identical.

No ``eval``/``exec``/``sympify``: the expression goes through
:class:`~graphwar_sim.parser.PolishNotationFunction` only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from graphwar_sim import TEAM2, Game, PolishNotationFunction, config
from graphwar_sim.physics import ShotResult, Soldier, process_function_range

from .base import hit_team_counts, world_coords


@dataclass(frozen=True)
class SimResult:
    """What firing ``expr`` on the current turn *would* do."""

    parseable: bool
    hit_enemy: bool
    hit_teammate: bool
    num_hits: int  # distinct soldiers struck (enemy + teammate)
    num_steps: int
    # Python-side diagnostic (the reference exception is message-free; see
    # docs/GROUND_TRUTH.md §3.6).
    error: str | None = None
    # Miss telemetry (None when unparseable or no trajectory points), in the
    # SHOOTER's centered world frame — the frame of agents.observation and
    # the turn message.
    nearest_miss: float | None = None
    miss_direction: str | None = None  # "high" | "low" (relative to the enemy)
    stopped_at_x: float | None = None
    stop_reason: str | None = None  # "hit" | "terrain" | "off_map" | "short" | "passed"


# A trajectory that terminates at a plane edge dies within ~one step of the
# border; the last recorded point sits this many px from it. # TUNABLE —
# not from source (derived from FUNC_MAX_STEP_DISTANCE).
_EDGE_PX: float = 22.0

# Slop (px) for "the last point touches a terrain circle": the physics
# terminates one step BEFORE the colliding pixel, so the last recorded point
# can sit just outside the circle. # TUNABLE — not from source.
_TERRAIN_SLOP_PX: float = 4.0


def _telemetry(
    game: Game,
    shooter: Soldier,
    inverted: bool,
    result: ShotResult,
    hit: bool,
) -> tuple[float | None, str | None, float | None, str | None]:
    """``(nearest_miss, miss_direction, stopped_at_x, stop_reason)`` in the
    shooter's world frame, computed from the trajectory points."""
    points = result.points[: result.num_steps]
    if not points:
        return None, None, None, None
    enemies = [s for s in game.all_soldiers() if s.alive and s.player_index != shooter.player_index]
    enemy_world = [world_coords(s.x, s.y, mirrored=inverted) for s in enemies]

    nearest: float | None = None
    direction: str | None = None
    for px, py in points:
        wx, wy = world_coords(px, py, mirrored=inverted)
        for ex, ey in enemy_world:
            d = math.hypot(wx - ex, wy - ey)
            if nearest is None or d < nearest:
                nearest = d
                # World y is up: a smaller plane py means the curve passed
                # ABOVE the enemy.
                direction = "high" if wy > ey else "low"

    last_px, last_py = points[-1]
    stop_wx, _ = world_coords(last_px, last_py, mirrored=inverted)
    if hit:
        reason = "hit"
    elif last_py <= _EDGE_PX or last_py >= config.PLANE_HEIGHT - _EDGE_PX:
        reason = "off_map"
    elif _touches_terrain(game, last_px, last_py):
        reason = "terrain"
    elif enemy_world:
        reason = "short" if stop_wx < min(ex for ex, _ in enemy_world) else "passed"
    else:
        reason = "passed"
    return nearest, direction, stop_wx, reason


def _touches_terrain(game: Game, px: float, py: float) -> bool:
    """True when the last trajectory point sits on (or just outside) a terrain
    circle — the physics terminates one step before the colliding pixel."""
    return any(math.hypot(px - cx, py - cy) <= r + _TERRAIN_SLOP_PX for cx, cy, r in game.circles)


def simulate(game: Game, expr: str) -> SimResult:
    """Fire ``expr`` through the physics without applying kills.

    The game state alone determines the outcome; the expression is evaluated
    in the centered world frame of the current shooter (see
    :func:`~agents.observation.observe`).
    """
    try:
        f = PolishNotationFunction(expr)
    except Exception as exc:  # noqa: BLE001 - the reference raises no message
        return SimResult(
            parseable=False,
            hit_enemy=False,
            hit_teammate=False,
            num_hits=0,
            num_steps=0,
            error=f"{type(exc).__name__}",
        )

    team = game.state.current_team()
    shooter = team.current_soldier()
    inverted = team.team == TEAM2
    result = process_function_range(f, shooter, game.all_soldiers(), game.terrain, inverted)
    enemy_hits, teammate_hits = hit_team_counts(game, result)
    nearest, direction, stop_x, reason = _telemetry(
        game, shooter, inverted, result, hit=enemy_hits > 0 or teammate_hits > 0
    )
    return SimResult(
        parseable=True,
        hit_enemy=enemy_hits > 0,
        hit_teammate=teammate_hits > 0,
        num_hits=enemy_hits + teammate_hits,
        num_steps=result.num_steps,
        nearest_miss=nearest,
        miss_direction=direction,
        stopped_at_x=stop_x,
        stop_reason=reason,
    )


__all__ = ["SimResult", "simulate"]
