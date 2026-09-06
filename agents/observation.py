"""Observation builder: a Game -> model-agnostic board view (Phase 3 / M3).

Presents the board in the **centered world frame** — the space where the game
evaluates ``y = f(x)`` (``x`` in ``[-25, 25]``, ``y`` up) — with the TEAM2
mirror applied so the current shooter always faces right. This is the frame
contract for every agent expression: the round-trip test
(``tests/test_agents.py``) asserts that an expression derived from this
observation hits exactly the geometry the physics integrates.

The terrain is sampled at a coarse grid into world-space blocked cells
(``# TUNABLE`` step; terrain occlusion is what makes a straight line fail).
The ASCII board renders the same frame with 'M' (shooter), 'S' (ally),
'E' (enemy), '#' (terrain).
"""

from __future__ import annotations

from collections.abc import Sequence

from graphwar_sim import TEAM2, Game, config
from graphwar_sim.physics import Obstacle

from .base import FRAME_CENTERED_WORLD, Observation, world_coords

# Coarse terrain sampling step in plane pixels. # TUNABLE — coarse enough to be
# cheap (52x30 cells), fine enough to show occlusion blobs.
_TERRAIN_STEP: int = 15

# ASCII canvas over world x in [-25, 25] and the reachable y band (the plane is
# only 450px tall => y in [-14.63, +14.63], GROUND_TRUTH.md §1.2).
_ASCII_COLS: int = 70
_ASCII_ROWS: int = 32
_ASCII_X_MIN: float = -25.0
_ASCII_X_MAX: float = 25.0
_ASCII_Y_MIN: float = -15.0
_ASCII_Y_MAX: float = 15.0


def observe(game: Game) -> Observation:
    """Build the :class:`Observation` for the current turn (centered world frame).

    If the current team is TEAM2, the board is mirrored about the vertical
    centre (Function.java:188-191) before the plane->world transform, so the
    shooter is always the left-most entity and enemies are at positive ``x``.
    """
    team = game.state.current_team()
    mirrored = team.team == TEAM2
    shooter = team.current_soldier()
    shooter_world = world_coords(shooter.x, shooter.y, mirrored)

    own: list[tuple[float, float]] = []
    enemies: list[tuple[float, float]] = []
    for t in game.state.teams:
        for s in t.soldiers:
            if not s.alive:
                continue
            w = world_coords(s.x, s.y, mirrored)
            if t.team == team.team:
                own.append(w)
            else:
                enemies.append(w)
    # Nearest-enemy-first is the ergonomic order for baseline aiming.
    enemies.sort(key=lambda p: (p[0], p[1]))

    terrain = _sample_terrain(game.terrain, mirrored)
    return Observation(
        frame=FRAME_CENTERED_WORLD,
        turn_index=game.state.current_turn,
        team_id=team.team,
        shooter=shooter_world,
        own_soldiers=tuple(own),
        enemy_soldiers=tuple(enemies),
        terrain_blocks=terrain,
        ascii_board=_ascii_board(shooter_world, own, enemies, terrain),
    )


def _sample_terrain(
    obstacle: Obstacle,
    mirrored: bool,
) -> tuple[tuple[float, float], ...]:
    """Coarse world-frame cells where terrain blocks (plane pixels sampled on a
    grid, each transformed into the (possibly mirrored) world frame)."""
    blocks: list[tuple[float, float]] = []
    collide = obstacle.collide_point
    for py in range(0, config.PLANE_HEIGHT, _TERRAIN_STEP):
        for px in range(0, config.PLANE_LENGTH, _TERRAIN_STEP):
            if collide(px, py):
                blocks.append(world_coords(px, py, mirrored))
    return tuple(blocks)


def _ascii_board(
    shooter: tuple[float, float],
    own: Sequence[tuple[float, float]],
    enemies: Sequence[tuple[float, float]],
    terrain_blocks: Sequence[tuple[float, float]],
) -> str:
    """Render the world frame as a fixed-width ASCII map (y up, row 0 = top)."""
    grid: list[list[str]] = [
        [" "] * _ASCII_COLS for _ in range(_ASCII_ROWS)
    ]
    for wx, wy in terrain_blocks:
        col, row = _cell(wx, wy)
        if col is not None and row is not None:
            grid[row][col] = "#"
    for wx, wy in enemies:
        col, row = _cell(wx, wy)
        if col is not None and row is not None:
            grid[row][col] = "E"
    for wx, wy in own:
        col, row = _cell(wx, wy)
        if col is not None and row is not None:
            grid[row][col] = "S"
    col, row = _cell(*shooter)
    if col is not None and row is not None:
        grid[row][col] = "M"
    return "\n".join("".join(row_cells) for row_cells in grid)


def _cell(wx: float, wy: float) -> tuple[int | None, int | None]:
    """World coords -> (col, row) on the ASCII grid, or (None, None) if OOB."""
    if not (_ASCII_X_MIN <= wx <= _ASCII_X_MAX and _ASCII_Y_MIN <= wy <= _ASCII_Y_MAX):
        return None, None
    col = int((wx - _ASCII_X_MIN) / (_ASCII_X_MAX - _ASCII_X_MIN) * _ASCII_COLS)
    row = int((_ASCII_Y_MAX - wy) / (_ASCII_Y_MAX - _ASCII_Y_MIN) * _ASCII_ROWS)
    return min(col, _ASCII_COLS - 1), min(row, _ASCII_ROWS - 1)


__all__ = ["observe"]
