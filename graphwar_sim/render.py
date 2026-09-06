"""Headless matplotlib rendering of a Graphwar state.

Minimal for M1: draw the terrain (from an :class:`Obstacle`'s collide grid),
the soldiers (coloured by team), and a shot trajectory. Used for debugging and,
later, the M4 eval harness. No GUI — ``matplotlib`` is used in the non-
interactive ``Agg`` backend so it works without a display.

The reference renders in *plane/pixel* space (770×450, origin top-left, ``y``
down). We render in that same space for fidelity; ``y`` is flipped so "up" is
up on screen.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import config
from .physics import Obstacle, ShotResult, Soldier
from .state import Game, Team


def obstacle_to_image(terrain: Obstacle) -> np.ndarray:
    """Rasterise an :class:`Obstacle` into a boolean ``(height, length)`` grid.

    ``True`` = terrain (collides). The grid is indexed ``[y][x]`` in plane
    space (``y`` down). This is what ``GoldenShot`` dumps for byte-identical
    parity, so the simulator and the reference agree on terrain.
    """
    grid = np.zeros((config.PLANE_HEIGHT, config.PLANE_LENGTH), dtype=bool)
    for y in range(config.PLANE_HEIGHT):
        for x in range(config.PLANE_LENGTH):
            grid[y, x] = terrain.collide_point(x, y)
    return grid


def _team_color(team_id: int) -> str:
    return "#e11d48" if team_id == config.TEAM1 else "#2563eb"


def render_state(
    game: Game,
    shot: ShotResult | None = None,
    title: str = "Graphwar",
) -> object:
    """Render a :class:`Game` (and an optional shot) to a matplotlib ``Figure``.

    Returns the ``Figure`` so callers can ``savefig`` / assert on it. Uses the
    ``Agg`` backend so it is safe headless.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.7, 4.5), dpi=100)

    # Terrain: white background, black terrain.
    image = obstacle_to_image(game.terrain)
    ax.imshow(
        image,
        cmap="gray_r",
        extent=(0, config.PLANE_LENGTH, config.PLANE_HEIGHT, 0),
        interpolation="nearest",
        origin="upper",
    )

    # Soldiers.
    for team in game.state.teams:
        color = _team_color(team.team)
        for s in team.soldiers:
            if not s.alive:
                continue
            ax.plot(
                s.x,
                s.y,
                "o",
                color=color,
                markersize=config.SOLDIER_RADIUS,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )

    # Current-turn highlight.
    if 0 <= game.state.current_turn < len(game.state.teams):
        cur = game.state.current_team().current_soldier()
        ax.plot(
            cur.x,
            cur.y,
            "o",
            fillstyle="none",
            color="yellow",
            markersize=config.SOLDIER_RADIUS * 2,
            markeredgewidth=2,
        )

    # Trajectory.
    if shot is not None and shot.points:
        pts = np.asarray(shot.points, dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], "-", color="orange", linewidth=1.2)

    ax.set_xlim(0, config.PLANE_LENGTH)
    ax.set_ylim(config.PLANE_HEIGHT, 0)  # flip so up is up
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    return fig


def render_match(
    game: Game,
    shots: Sequence[tuple[int, ShotResult]],
    title: str = "Match",
) -> object:
    """Render a full match: final board + every shot's trajectory (M4).

    ``shots`` is the turn-ordered list of ``(team_id, ShotResult)`` the match
    runner recorded. Each trajectory is drawn in its shooter's team color; hit
    points are marked with '+'; eliminated soldiers are shown as faint 'x' so
    the plot tells the match narrative at a glance. The board (alive soldiers)
    is the state after the last recorded shot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.7, 4.5), dpi=100)
    image = obstacle_to_image(game.terrain)
    ax.imshow(
        image,
        cmap="gray_r",
        extent=(0, config.PLANE_LENGTH, config.PLANE_HEIGHT, 0),
        interpolation="nearest",
        origin="upper",
    )

    for team in game.state.teams:
        color = _team_color(team.team)
        for s in team.soldiers:
            if s.alive:
                ax.plot(
                    s.x,
                    s.y,
                    "o",
                    color=color,
                    markersize=config.SOLDIER_RADIUS,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
            else:
                ax.plot(s.x, s.y, "x", color="gray", markersize=5)

    for team_id, shot in shots:
        if not shot.points:
            continue
        color = _team_color(team_id)
        pts = np.asarray(shot.points, dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=1.0, alpha=0.85)
        # Hit markers at the struck soldiers (hit positions index into points).
        for _player, _soldier, pos in shot.hits:
            if 0 <= pos < len(shot.points):
                hx, hy = shot.points[pos]
                ax.plot(hx, hy, "+", color="black", markersize=8)

    ax.set_xlim(0, config.PLANE_LENGTH)
    ax.set_ylim(config.PLANE_HEIGHT, 0)  # flip so up is up
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    return fig


def render_shot(
    terrain: Obstacle,
    shooter: Soldier,
    teams: Sequence[Team],
    shot: ShotResult,
    title: str = "Shot",
) -> object:
    """Render a single shot against terrain + soldiers (no full ``Game``)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.7, 4.5), dpi=100)
    image = obstacle_to_image(terrain)
    ax.imshow(
        image,
        cmap="gray_r",
        extent=(0, config.PLANE_LENGTH, config.PLANE_HEIGHT, 0),
        interpolation="nearest",
        origin="upper",
    )
    for team in teams:
        color = _team_color(team.team)
        for s in team.soldiers:
            if not s.alive:
                continue
            ax.plot(
                s.x,
                s.y,
                "o",
                color=color,
                markersize=config.SOLDIER_RADIUS,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
    pts = np.asarray(shot.points, dtype=float)
    ax.plot(pts[:, 0], pts[:, 1], "-", color="orange", linewidth=1.2)
    ax.set_xlim(0, config.PLANE_LENGTH)
    ax.set_ylim(config.PLANE_HEIGHT, 0)
    ax.set_aspect("equal")
    ax.set_title(title)
    return fig
