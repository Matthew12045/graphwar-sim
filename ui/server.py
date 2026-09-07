"""FastAPI server wrapping ``graphwar_sim.state.Game`` for the web UI.

Pure additive layer on the tested core: ``graphwar_sim/`` is imported
unchanged. Endpoints:

- ``POST /api/new_game`` ``{seed?, num_soldiers?, team_modes?, max_turns?}``
  -> board JSON (``team_modes`` selects the per-side driver: ``"human"``, an
  eval roster entry, or ``llm:<model>`` — agents are constructed EAGERLY so a
  missing auth token or an unknown mode fails fast as a 400)
- ``GET  /api/state``   -> board JSON (lazy-creates a game)
- ``POST /api/fire``    ``{func_str: str}`` -> shot + board + outcome JSON,
  or a distinct 400 error shape on ``MalformedFunction`` (the reference has
  no diagnostic strings either — ``docs/GROUND_TRUTH.md`` §6 correction 2 —
  so the error is clearly labeled as a Python-side parse failure, not a
  fake "source" message)
- ``POST /api/agent_turn`` (no body) -> one turn played by the current
  side's agent (the spectator driver; never a whole match synchronously).

Angle display: ``graphwar_sim.physics._get_start_angle`` (physics.py:100) is
imported directly here for **display purposes only** (the compass widget,
``docs/UI_GROUND_TRUTH.md`` §3). The private function is used as-is rather
than promoted to a public helper, so the core stays untouched; the muzzle
world-transform below is a read-only duplicate of ``physics.py:154-164``.
It never alters the shot itself — the dial updates on fire, exactly like the
reference (``GameData.java:1113`` sets the soldier's display angle only when
the function is processed).

Not thread-safe by the core: a single match is held behind a module lock.
NOTE: ``/api/agent_turn`` calls ``agent.act`` INSIDE that lock, so an LLM
side blocks other requests for the length of one API round-trip loop —
acceptable for this single-user UI (M5.4 risk note), documented here.
"""

from __future__ import annotations

import math
import random
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents import Agent, hit_team_counts, observe
from eval.runner import _classify, _peek_solver_rung, make_agent
from graphwar_sim import TEAM1, TEAM2, config
from graphwar_sim.parser import MalformedFunction, PolishNotationFunction
from graphwar_sim.physics import ShotResult, Soldier, _get_start_angle
from graphwar_sim.render import _team_color
from graphwar_sim.state import Game

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Graphwar (NORMAL_FUNC)", docs_url="/api/docs")

_lock = threading.Lock()
_game: Game | None = None
_seed: int | None = None
# M5.4 spectator state: the per-side driver mode (roster entry or "human"),
# the eagerly-constructed agent instances, and the match turn budget.
_team_modes: dict[int, str] = {}
_team_agents: dict[int, Agent] = {}
_turns_played: int = 0
_max_turns: int | None = None


# --- request bodies ---------------------------------------------------------


class _TeamModesBody(BaseModel):
    team1: str
    team2: str


class NewGameBody(BaseModel):
    seed: int | None = None
    num_soldiers: int | None = None
    team_modes: _TeamModesBody | None = None
    max_turns: int | None = None


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
    idempotent, so calling it here is safe. M5.4: the turn budget travels on
    the board so the frontend can resync after any response.
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
        "turns_played": _turns_played,
        "max_turns": _max_turns,
    }


def _finite_or_none(value: float) -> float | None:
    """JSON-safe float: shots that end on NaN/Inf carry non-finite points
    (``physics.py:248-250`` terminates *after* adding such a point). Python's
    json would emit bare ``NaN``/``Infinity`` — invalid JSON — so they travel
    as ``null``; list indices stay stable for the hit bookkeeping."""
    return value if math.isfinite(value) else None


def _shot_json(shot: ShotResult) -> dict[str, Any]:
    return {
        "points": [[_finite_or_none(x), _finite_or_none(y)] for x, y in shot.points],
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
    values_x = (
        config.PLANE_GAME_LENGTH * (values_x - config.PLANE_LENGTH // 2)
    ) / config.PLANE_LENGTH
    values_y = (
        config.PLANE_GAME_LENGTH * (-values_y + config.PLANE_HEIGHT // 2)
    ) / config.PLANE_LENGTH
    radius = (config.PLANE_GAME_LENGTH * config.SOLDIER_RADIUS) / config.PLANE_LENGTH

    angle = _get_start_angle(f, values_x, radius)
    if math.isnan(angle) or math.isinf(angle):
        return None
    return angle


def _team_modes_wire() -> dict[str, str]:
    """The per-side modes in wire form (``team1``/``team2`` keys), defaulting
    to ``"human"`` so a lazy/legacy match reads unchanged."""
    return {
        "team1": _team_modes.get(TEAM1, "human"),
        "team2": _team_modes.get(TEAM2, "human"),
    }


def _cap_reached() -> bool:
    """True when the match's turn budget is spent (None = unlimited)."""
    return _max_turns is not None and _turns_played >= _max_turns


def _agent_for_mode(mode: str) -> Agent:
    """Construct one side's agent from a roster-style mode string.

    Reuses :func:`eval.runner.make_agent` directly (the roster mapping is
    NOT duplicated here): unknown names raise ``ValueError``; an
    ``llm:<model>`` entry raises ``RuntimeError`` at construction when no
    auth env var is set. Both surface as a clean 400 from ``new_game``.
    """
    if mode == "llm:":
        raise ValueError("llm: mode needs a model name (llm:<model>)")
    return make_agent(mode, seed=0)


def _shooter_info(game: Game) -> dict[str, Any]:
    """Pre-turn shooter identity + display info (shared by fire/agent_turn)."""
    team = game.state.current_team()
    shooter = team.current_soldier()
    return {
        "player_index": game.state.current_turn,
        "soldier_index": shooter.soldier_index,
        "label": f"Player {game.state.current_turn + 1}",
        "color": _team_color(team.team),
        "inverted": team.team == config.TEAM2,
    }


def _lazy_game() -> Game:
    global _game, _seed, _team_modes, _team_agents, _turns_played, _max_turns
    if _game is None:
        _seed = _new_seed()
        _game = Game.create(seed=_seed, num_teams=2, num_soldiers=config.INITIAL_NUM_SOLDIERS)
        _team_modes = {}
        _team_agents = {}
        _turns_played = 0
        _max_turns = None
    return _game


# --- endpoints --------------------------------------------------------------


@app.post("/api/new_game", response_model=None)
def new_game(body: NewGameBody | None = None) -> dict[str, Any] | JSONResponse:
    """Create a fresh match. ``seed`` is optional for reproducibility.

    ``team_modes`` (``{"team1": ..., "team2": ...}``) selects each side's
    driver — defaults to ``"human"``/``"human"`` when omitted (the M5.3
    behavior). Agent sides are constructed EAGERLY under the lock, so a
    missing auth token or an unknown mode fails here as a clean 400 instead
    of exploding mid-match inside ``/api/agent_turn``.
    """
    global _game, _seed, _team_modes, _team_agents, _turns_played, _max_turns
    seed = body.seed if body is not None and body.seed is not None else _new_seed()
    num_soldiers = _clamp_num_soldiers(body.num_soldiers if body is not None else None)
    mode_team1 = body.team_modes.team1 if body is not None and body.team_modes else "human"
    mode_team2 = body.team_modes.team2 if body is not None and body.team_modes else "human"
    max_turns = body.max_turns if body is not None else None
    if max_turns is not None and max_turns < 1:
        return JSONResponse(
            status_code=400,
            content={"error": "bad_max_turns", "detail": "max_turns must be >= 1"},
        )
    with _lock:
        try:
            agents_by_team: dict[int, Agent] = {}
            for team_id, mode in ((TEAM1, mode_team1), (TEAM2, mode_team2)):
                if mode != "human":
                    agents_by_team[team_id] = _agent_for_mode(mode)
        except (ValueError, RuntimeError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": "bad_team_modes", "detail": str(exc)},
            )
        _seed = seed
        _game = Game.create(seed=seed, num_teams=2, num_soldiers=num_soldiers)
        _team_modes = {TEAM1: mode_team1, TEAM2: mode_team2}
        _team_agents = agents_by_team
        _turns_played = 0
        _max_turns = max_turns
        return {
            "seed": seed,
            "num_soldiers": num_soldiers,
            "team_modes": _team_modes_wire(),
            "max_turns": _max_turns,
            **_board_json(_game),
        }


@app.get("/api/state")
def state() -> dict[str, Any]:
    with _lock:
        game = _lazy_game()
        return {"seed": _seed, "team_modes": _team_modes_wire(), **_board_json(game)}


@app.post("/api/fire")
def fire(body: FireBody) -> JSONResponse:
    """Fire the current-turn soldier and advance the turn.

    Uses ``Game.play_turn`` (fires AND advances; ``state.py:250-254``). A
    malformed function raises inside ``Game.fire`` *before* the turn
    advances, so the board stays consistent; the distinct 400 shape below is
    the frontend's cue to show an inline error instead of a board update.

    M5.4 turn cap: when the budget is spent after this shot (match not
    otherwise finished), the response reports ``game_over: true`` with
    ``draw_reason: "TURN_CAP"``; requests arriving after the cap (no shot
    left to fire) get the same shape with no shot.
    """
    global _turns_played
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
        if _cap_reached():
            return JSONResponse(
                content={
                    "game_over": True,
                    "winner": None,
                    "draw_reason": "TURN_CAP",
                    "board": _board_json(game),
                    "seed": _seed,
                }
            )

        shooter_info = _shooter_info(game)
        team = game.state.current_team()
        shooter = team.current_soldier()
        inverted = team.team == config.TEAM2
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
        _turns_played += 1

        cap_draw = not game.finished() and _cap_reached()
        payload: dict[str, Any] = {
            "shot": _shot_json(shot),
            "board": _board_json(game),
            "game_over": game.finished() or cap_draw,
            "winner": game.winner(),
            "draw_reason": "TURN_CAP" if cap_draw else None,
            "shooter": shooter_info,
            "func_str": body.func_str,
            "start_angle": start_angle,
            "seed": _seed,
        }
        return JSONResponse(content=payload)


@app.post("/api/agent_turn", response_model=None)
def agent_turn() -> dict[str, Any] | JSONResponse:
    """Play exactly ONE turn with the current side's agent (M5.4 spectator).

    The endpoint is the frontend autoplay/step driver's single unit of work —
    never a whole match synchronously. Classification reuses the eval
    runner's own logic (``hit_team_counts`` + ``_classify`` via
    ``_peek_solver_rung`` — imported, not reimplemented), and the defensive
    malformed-emission branch mirrors ``eval/runner.py:223-228`` (safe dud
    ``"0*x"`` fired in place). The response is byte-identical to
    ``/api/fire``'s payload shape plus the ``agent``/``outcome``/``solver_rung``
    fields.

    LLM latency happens INSIDE the module lock (one API round-trip loop per
    call) — acceptable for this single-user UI, see the module docstring.
    """
    global _turns_played
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
        if _cap_reached():
            return JSONResponse(
                content={
                    "game_over": True,
                    "winner": None,
                    "draw_reason": "TURN_CAP",
                    "board": _board_json(game),
                    "seed": _seed,
                }
            )
        team = game.state.current_team()
        agent = _team_agents.get(team.team)
        if agent is None:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "human_turn",
                    "detail": "The current side is human-driven; use /api/fire.",
                },
            )

        shooter_info = _shooter_info(game)
        shooter = team.current_soldier()
        inverted = team.team == config.TEAM2

        obs = observe(game)
        solver_rung = _peek_solver_rung(agent)
        expr = agent.act(game, obs)

        t0 = time.perf_counter()
        try:
            shot = game.play_turn(expr)
            parse_failure = False
        except MalformedFunction:
            # Defensive: a buggy agent must never crash a match — the safe
            # dud replaces the emission (mirrors eval/runner.py:223-228).
            shot = game.fire("0*x")
            game.state.advance_turn()
            parse_failure = True
        elapsed = time.perf_counter() - t0
        _turns_played += 1

        enemy_hits, teammate_hits = hit_team_counts(game, shot)
        outcome = _classify(solver_rung, parse_failure, enemy_hits, elapsed)

        cap_draw = not game.finished() and _cap_reached()
        payload: dict[str, Any] = {
            "shot": _shot_json(shot),
            "board": _board_json(game),
            "game_over": game.finished() or cap_draw,
            "winner": game.winner(),
            "draw_reason": "TURN_CAP" if cap_draw else None,
            "shooter": shooter_info,
            "func_str": expr,
            "start_angle": _display_start_angle(expr, shooter, inverted),
            "seed": _seed,
            "agent": agent.name,
            "outcome": outcome.value,
            "solver_rung": solver_rung,
        }
        return JSONResponse(content=payload)


# Static frontend last, so /api/* routes match first. html=True serves
# ui/static/index.html at "/".
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
