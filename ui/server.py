"""FastAPI server wrapping ``graphwar_sim.state.Game`` for the web UI.

Pure additive layer on the tested core: ``graphwar_sim/`` is imported
unchanged. Endpoints:

- ``POST /api/new_game`` ``{seed?: int, num_soldiers?: int}`` -> board JSON
- ``GET  /api/state``   -> board JSON (lazy-creates a game)
- ``POST /api/fire``    ``{func_str: str}`` -> shot + board + outcome JSON,
  or a distinct 400 error shape on ``MalformedFunction`` (the reference has
  no diagnostic strings either — ``docs/GROUND_TRUTH.md`` §6 correction 2 —
  so the error is clearly labeled as a Python-side parse failure, not a
  fake "source" message).

Angle display: ``graphwar_sim.physics._get_start_angle`` (physics.py:100) is
imported directly here for **display purposes only** (the compass widget,
``docs/UI_GROUND_TRUTH.md`` §3). The private function is used as-is rather
than promoted to a public helper, so the core stays untouched; the muzzle
world-transform below is a read-only duplicate of ``physics.py:154-164``.
It never alters the shot itself — the dial updates on fire, exactly like the
reference (``GameData.java:1113`` sets the soldier's display angle only when
the function is processed).

Not thread-safe by the core: a single match is held behind a module lock.
"""

from __future__ import annotations

import math
import random
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graphwar_sim import config
from graphwar_sim.parser import MalformedFunction, PolishNotationFunction
from graphwar_sim.physics import ShotResult, Soldier, _get_start_angle
from graphwar_sim.render import _team_color
from graphwar_sim.state import Game

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Graphwar (NORMAL_FUNC)", docs_url="/api/docs")

_lock = threading.Lock()
_game: Game | None = None
_seed: int | None = None


# --- request bodies ---------------------------------------------------------


class NewGameBody(BaseModel):
    seed: int | None = None
    num_soldiers: int | None = None


class FireBody(BaseModel):
    func_str: str


# --- serialization helpers --------------------------------------------------


def _new_seed() -> int:
    return random.randrange(2**32)


def _clamp_num_soldiers(num_soldiers: int | None) -> int:
    if num_soldiers is None:
        return config.INITIAL_NUM_SOLDIERS
    return max(1, min(config.MAX_SOLDIERS_PER_PLAYER, num_soldiers))


def _soldier_json(soldier: Soldier) -> dict[str, Any]:
    return {
        "x": soldier.x,
        "y": soldier.y,
        "alive": soldier.alive,
        "player_index": soldier.player_index,
        "soldier_index": soldier.soldier_index,
    }


def _shooter_json(game: Game) -> dict[str, Any]:
    """Identity + display info for the soldier that fires this turn."""
    team = game.state.current_team()
    soldier = team.current_soldier()
    return {
        "player_index": game.state.current_turn,
        "soldier_index": soldier.soldier_index,
        "label": f"Player {game.state.current_turn + 1}",
        "color": _team_color(team.team),
        "inverted": team.team == config.TEAM2,
    }


def _board_json(game: Game) -> dict[str, Any]:
    """Full board state for drawing (``docs/UI_GROUND_TRUTH.md`` §1-§2).

    ``all_soldiers()`` also assigns each soldier its ``(player_index,
    soldier_index)`` hit identity (``graphwar_sim/state.py:213-222``) and is
    idempotent, so calling it here is safe.
    """
    soldiers = game.all_soldiers()
    del soldiers  # only the index-assignment side effect is needed
    return {
        "teams": [
            {
                "name": team.name,
                "team": team.team,
                "label": f"Player {j + 1}",
                "color": _team_color(team.team),
                "current_turn_soldier": team.current_turn_soldier,
                "num_alive": team.num_alive(),
                "soldiers": [_soldier_json(s) for s in team.soldiers],
            }
            for j, team in enumerate(game.state.teams)
        ],
        "current_turn": game.state.current_turn,
        "shooter": _shooter_json(game),
        # Terrain as circles — the frontend draws filled circles directly
        # (do NOT rasterize the obstacle grid like render.py does).
        "circles": [list(circle) for circle in game.circles],
        "finished": game.finished(),
        "winner": game.winner(),
    }


def _finite_or_none(value: float) -> float | None:
    """JSON-safe float: shots that end on NaN/Inf carry non-finite points
    (``physics.py:248-250`` terminates *after* adding such a point). Python's
    json would emit bare ``NaN``/``Infinity`` — invalid JSON — so they travel
    as ``null``; list indices stay stable for the hit bookkeeping."""
    return value if math.isfinite(value) else None


def _shot_json(shot: ShotResult) -> dict[str, Any]:
    return {
        "points": [
            [_finite_or_none(x), _finite_or_none(y)] for x, y in shot.points
        ],
        "hits": [list(hit) for hit in shot.hits],
        "last_x": _finite_or_none(shot.last_x),
        "last_y": _finite_or_none(shot.last_y),
        "num_steps": shot.num_steps,
    }


def _display_start_angle(func_str: str, shooter: Soldier, inverted: bool) -> float | None:
    """The compass wedge angle (radians, game space) for a typed function.

    Display-only duplicate of the muzzle transform in
    ``process_function_range`` (``physics.py:154-164``) feeding the private
    ``_get_start_angle`` (``physics.py:100``). Returns ``None`` when the
    function does not parse or the angle is NaN/Inf — the dial then simply
    keeps its previous value, like the reference's unparsed-input behavior.
    """
    try:
        f = PolishNotationFunction(func_str)
    except MalformedFunction:
        return None

    values_x, values_y = shooter.x, shooter.y
    if inverted:
        values_x = config.PLANE_LENGTH - values_x
    values_x = (config.PLANE_GAME_LENGTH * (values_x - config.PLANE_LENGTH // 2)) / config.PLANE_LENGTH
    values_y = (config.PLANE_GAME_LENGTH * (-values_y + config.PLANE_HEIGHT // 2)) / config.PLANE_LENGTH
    radius = (config.PLANE_GAME_LENGTH * config.SOLDIER_RADIUS) / config.PLANE_LENGTH

    angle = _get_start_angle(f, values_x, radius)
    if math.isnan(angle) or math.isinf(angle):
        return None
    return angle


def _lazy_game() -> Game:
    global _game, _seed
    if _game is None:
        _seed = _new_seed()
        _game = Game.create(seed=_seed, num_teams=2, num_soldiers=config.INITIAL_NUM_SOLDIERS)
    return _game


# --- endpoints --------------------------------------------------------------


@app.post("/api/new_game")
def new_game(body: NewGameBody | None = None) -> dict[str, Any]:
    """Create a fresh match. ``seed`` is optional for reproducibility."""
    global _game, _seed
    seed = body.seed if body is not None and body.seed is not None else _new_seed()
    num_soldiers = _clamp_num_soldiers(body.num_soldiers if body is not None else None)
    with _lock:
        _seed = seed
        _game = Game.create(seed=seed, num_teams=2, num_soldiers=num_soldiers)
        return {"seed": seed, "num_soldiers": num_soldiers, **_board_json(_game)}


@app.get("/api/state")
def state() -> dict[str, Any]:
    with _lock:
        game = _lazy_game()
        return {"seed": _seed, **_board_json(game)}


@app.post("/api/fire")
def fire(body: FireBody) -> JSONResponse:
    """Fire the current-turn soldier and advance the turn.

    Uses ``Game.play_turn`` (fires AND advances; ``state.py:250-254``). A
    malformed function raises inside ``Game.fire`` *before* the turn
    advances, so the board stays consistent; the distinct 400 shape below is
    the frontend's cue to show an inline error instead of a board update.
    """
    with _lock:
        game = _lazy_game()
        if game.finished():
            return JSONResponse(
                status_code=409,
                content={
                    "error": "game_over",
                    "detail": "The match is over — start a new game to keep playing.",
                    "board": _board_json(game),
                },
            )

        team = game.state.current_team()
        shooter = team.current_soldier()
        inverted = team.team == config.TEAM2
        shooter_info = {
            "player_index": game.state.current_turn,
            "soldier_index": shooter.soldier_index,
            "label": f"Player {game.state.current_turn + 1}",
            "color": _team_color(team.team),
            "inverted": inverted,
        }
        start_angle = _display_start_angle(body.func_str, shooter, inverted)

        try:
            shot = game.play_turn(body.func_str)
        except MalformedFunction:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "malformed_function",
                    "detail": (
                        "Python-side parse error: the string is not a well-formed "
                        "y = f(x) expression. The reference client reports no "
                        "diagnostics either."
                    ),
                    "func_str": body.func_str,
                },
            )

        payload: dict[str, Any] = {
            "shot": _shot_json(shot),
            "board": _board_json(game),
            "game_over": game.finished(),
            "winner": game.winner(),
            "shooter": shooter_info,
            "func_str": body.func_str,
            "start_angle": start_angle,
            "seed": _seed,
        }
        return JSONResponse(content=payload)


# Static frontend last, so /api/* routes match first. html=True serves
# ui/static/index.html at "/".
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
