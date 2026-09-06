"""Faithful Python port of ``Graphwar.Function.processFunctionRange``.

This is the heart of the simulator: given a parsed ``y = f(x)`` function, the
current shooter, the field of soldiers, and the terrain, it integrates the
trajectory with the reference's adaptive step-halving and reports every soldier
hit.

Reference: ``ref/graphwar/src/Graphwar/Function.java``
- ``getStartAngle``            (Function.java:133-160)
- ``playerAlreadyHit``         (Function.java:162-171)
- ``processFunctionRange``     (Function.java:173-309)

Coordinate frames (see ``docs/GROUND_TRUTH.md`` §1):
- **plane/pixel** space: 770×450, origin top-left, ``y`` positive *down*.
  Soldiers and the terrain live here.
- **centered world** space: ``x`` in ``[-25, 25]``, ``y`` positive *up*.
  The function ``f`` is evaluated here.

The two are related by the transforms in ``processFunctionRange``. Note the
source quirk that the *inverse* transform uses ``PLANE_LENGTH`` for **both**
axes (Function.java:243-244); we reproduce it exactly rather than "fixing" it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from . import config
from .parser import PolishNotationFunction

# Integer divisions in the source are Java ``int`` divisions (Function.java:193,
# 194, 243, 244, 301, 302). Both are exact here (770/2, 450/2) but we keep ``//``
# to stay faithful to the integer-division semantics.
_HALF_LENGTH = config.PLANE_LENGTH // 2  # 385
_HALF_HEIGHT = config.PLANE_HEIGHT // 2  # 225


@dataclass
class Soldier:
    """A single soldier in plane/pixel coordinates (``y`` down).

    ``player_index`` / ``soldier_index`` are the (j, k) coordinates used by the
    reference's hit bookkeeping (Function.java:252-283).
    """

    x: float
    y: float
    alive: bool = True
    player_index: int = 0
    soldier_index: int = 0


@dataclass
class Obstacle:
    """Terrain. ``collide_point`` mirrors ``Obstacle.collidePoint``.

    Empirically confirmed (prior session): returns ``True`` for black pixels and
    out-of-bounds, ``False`` for white pixels.
    """

    collide_point: Callable[[int, int], bool]  # (int x, int y) -> bool

    def __post_init__(self) -> None:
        # Expose the Java name too, for call sites that mirror the source.
        # (Attribute is dynamically added; newer mypy accepts this without an
        # ignore — retained here for older mypy and clarity.)
        self.collidePoint = self.collide_point


@dataclass
class ShotResult:
    """Outcome of one shot.

    - ``points``: the integrated trajectory in plane/pixel coords, ``(x, y)``,
      from the muzzle (index 0) through the last step (index ``num_steps - 1``).
    - ``hits``: ``(player_index, soldier_index, position_i)`` for each distinct
      soldier struck, in the order first hit.
    - ``last_x`` / ``last_y``: the final point in plane coords (Function.java:301-302).
    - ``num_steps``: number of integrated points (Function.java:211, 237-297).
    """

    points: list[tuple[float, float]] = field(default_factory=list)
    hits: list[tuple[int, int, int]] = field(default_factory=list)
    last_x: float = 0.0
    last_y: float = 0.0
    num_steps: int = 0


def _to_plane_x(values_x: float) -> float:
    return config.PLANE_LENGTH * values_x / config.PLANE_GAME_LENGTH + _HALF_LENGTH


def _to_plane_y(values_y: float) -> float:
    # Source quirk: PLANE_LENGTH (not PLANE_HEIGHT) scales the y axis.
    return -config.PLANE_LENGTH * values_y / config.PLANE_GAME_LENGTH + _HALF_HEIGHT


def _get_start_angle(f: PolishNotationFunction, x: float, radius: float) -> float:
    """``getStartAngle`` (Function.java:133-160).

    Fixed-point iteration on the curve's own tangent to find the launch angle at
    the muzzle. Returns ``atan`` of the tangent, refined until the angle change
    drops below ``ANGLE_ERROR`` or ``MAX_ANGLE_LOOPS`` is reached.
    """
    angle = 0.0

    start_tangent = (f.evaluate(x + config.STEP_SIZE) - f.evaluate(x)) / config.STEP_SIZE
    angle = math.atan(start_tangent)

    error = 10000.0
    i = 0
    while error > config.ANGLE_ERROR and i < config.MAX_ANGLE_LOOPS:
        final_x = x + radius * math.cos(angle)
        start_tangent = (
            f.evaluate(final_x + config.STEP_SIZE) - f.evaluate(final_x)
        ) / config.STEP_SIZE
        new_angle = math.atan(start_tangent)
        error = abs(new_angle - angle)
        angle = new_angle
        i += 1

    return angle


def process_function_range(
    f: PolishNotationFunction,
    shooter: Soldier,
    soldiers: Sequence[Soldier],
    obstacle: Obstacle,
    inverted: bool,
) -> ShotResult:
    """Port of ``processFunctionRange`` (Function.java:173-309).

    Parameters
    ----------
    f:
        The parsed firing function ``y = f(x)`` (centered-world coordinates).
    shooter:
        The current-turn soldier (its ``(x, y)`` are the muzzle in plane coords).
        Its ``player_index`` / ``soldier_index`` identify it for the skip rule.
    soldiers:
        Every soldier on the field (plane coords, ``alive`` flag, and
        ``player_index`` / ``soldier_index``). The shooter is included; the
        reference skips only the shooter's *current-turn* soldier, so any other
        soldier of the shooter's own player can still be hit.
    obstacle:
        Terrain, exposing ``collide_point(int x, int y) -> bool``.
    inverted:
        ``True`` for the TEAM2 (right-side) shooter, which is mirrored about the
        vertical centre (Function.java:188-191, 247-250).
    """
    # --- Muzzle: plane -> centered world (Function.java:185-194) -------------
    values_x = shooter.x
    values_y = shooter.y
    if inverted:
        values_x = config.PLANE_LENGTH - values_x
    values_x = (config.PLANE_GAME_LENGTH * (values_x - _HALF_LENGTH)) / config.PLANE_LENGTH
    values_y = (config.PLANE_GAME_LENGTH * (-values_y + _HALF_HEIGHT)) / config.PLANE_LENGTH

    game_coordinate_radius = (
        config.PLANE_GAME_LENGTH * config.SOLDIER_RADIUS
    ) / config.PLANE_LENGTH

    # --- Launch angle from the curve's own tangent (Function.java:198-204) ---
    fire_angle = _get_start_angle(f, values_x, game_coordinate_radius)
    if not (math.isnan(fire_angle) or math.isinf(fire_angle)):
        values_x += game_coordinate_radius * math.cos(fire_angle)
        values_y += game_coordinate_radius * math.sin(fire_angle)

    # --- Auto vertical offset: force the curve through the muzzle ------------
    # offSet = y0 - f(x0)  (Function.java:206)
    off_set = -f.evaluate(values_x) + values_y

    step_size = config.STEP_SIZE
    temp_step_size = config.STEP_SIZE  # noqa: F841 - mirrors source; reset each iter

    num_steps = config.FUNC_MAX_STEPS

    # Trajectory storage (game coords), indexed like the source's arrays.
    xs = [0.0] * config.FUNC_MAX_STEPS
    ys = [0.0] * config.FUNC_MAX_STEPS
    xs[0] = values_x
    ys[0] = values_y

    # Hit bookkeeping (Function.java:175-178).
    players_hit: list[int] = []
    soldiers_hit: list[int] = []
    hit_positions: list[int] = []

    def player_already_hit(player: int, soldier: int) -> bool:
        for i in range(len(players_hit)):
            if players_hit[i] == player and soldiers_hit[i] == soldier:
                return True
        return False

    for i in range(1, config.FUNC_MAX_STEPS):
        temp_step_size = step_size

        xs[i] = xs[i - 1] + temp_step_size
        ys[i] = f.evaluate(xs[i]) + off_set

        end_func = False
        # Adaptive step-halving (Function.java:221-235).
        while (
            math.pow(xs[i] - xs[i - 1], 2) + math.pow(ys[i] - ys[i - 1], 2)
            > config.FUNC_MAX_STEP_DISTANCE_SQUARED
        ):
            if xs[i] - xs[i - 1] > config.FUNC_MIN_X_STEP_DISTANCE:
                temp_step_size /= 2
                xs[i] = xs[i - 1] + temp_step_size
                ys[i] = f.evaluate(xs[i]) + off_set
            else:
                end_func = True
                break

        if end_func:
            num_steps = i
            break

        # Game -> plane (Function.java:243-244), then TEAM2 mirror (247-250).
        x = _to_plane_x(xs[i])
        y = _to_plane_y(ys[i])
        if inverted:
            x = config.PLANE_LENGTH - x

        # Hit test (Function.java:252-284): strict <, multi-kill, dedup.
        for s in soldiers:
            if s.player_index == shooter.player_index and (
                s.soldier_index == shooter.soldier_index
            ):
                continue
            if s.alive:
                dist_x = s.x - x
                dist_y = s.y - y
                dist_squared = math.pow(dist_x, 2) + math.pow(dist_y, 2)
                in_radius = dist_squared < config.SOLDIER_RADIUS * config.SOLDIER_RADIUS
                if in_radius and not player_already_hit(s.player_index, s.soldier_index):
                    players_hit.append(s.player_index)
                    soldiers_hit.append(s.soldier_index)
                    hit_positions.append(i)

        # Terrain / NaN termination (Function.java:287-297).
        if obstacle.collide_point(int(x), int(y)):
            num_steps = i
            break
        if math.isnan(y) or math.isinf(y):
            num_steps = i
            break

    # Final point (Function.java:301-302). NOTE: the source does NOT apply the
    # inverted mirror to lastX/lastY (only to the per-point x at 247-250), so we
    # reproduce that asymmetry exactly.
    last_x = _to_plane_x(xs[num_steps - 1])
    last_y = _to_plane_y(ys[num_steps - 1])

    # Build the plane-coord trajectory (muzzle through last step).
    points: list[tuple[float, float]] = []
    for i in range(num_steps):
        px = _to_plane_x(xs[i])
        py = _to_plane_y(ys[i])
        if inverted:
            px = config.PLANE_LENGTH - px
        points.append((px, py))

    return ShotResult(
        points=points,
        hits=list(zip(players_hit, soldiers_hit, hit_positions, strict=True)),
        last_x=last_x,
        last_y=last_y,
        num_steps=num_steps,
    )
