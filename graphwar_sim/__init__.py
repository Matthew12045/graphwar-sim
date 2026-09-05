"""Graphwar simulator (M1: headless simulator + golden tests).

A clean-room Python reimplementation of the artillery game Graphwar's
NORMAL_FUNC shot mechanics. The trajectory of a shot is a function ``y = f(x)``
typed by the player; this package parses, integrates, and resolves it faithfully
to the GPL-licensed Java reference (cited throughout, not copied).

Public surface:
    - :class:`PolishNotationFunction` / :class:`MalformedFunction` (parser)
    - :class:`Soldier`, :class:`Obstacle`, :class:`ShotResult`,
      :func:`process_function_range` (physics)
    - :class:`Game`, :class:`GameState` (state / turn order / win rule)
    - :mod:`config` (all constants, each cited to the Java source)
"""

from .config import (
    PLANE_LENGTH,
    PLANE_HEIGHT,
    PLANE_GAME_LENGTH,
    SOLDIER_RADIUS,
    STEP_SIZE,
    TEAM1,
    TEAM2,
    NORMAL_FUNC,
)
from .parser import MalformedFunction, PolishNotationFunction
from .physics import Obstacle, ShotResult, Soldier, process_function_range
from .state import Game, GameState, Team

__all__ = [
    "PLANE_LENGTH",
    "PLANE_HEIGHT",
    "PLANE_GAME_LENGTH",
    "SOLDIER_RADIUS",
    "STEP_SIZE",
    "TEAM1",
    "TEAM2",
    "NORMAL_FUNC",
    "MalformedFunction",
    "PolishNotationFunction",
    "Obstacle",
    "ShotResult",
    "Soldier",
    "process_function_range",
    "Game",
    "GameState",
    "Team",
]
