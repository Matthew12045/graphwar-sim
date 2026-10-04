"""Torus arena (non-reference, reel mode): shots wrap at the plane edges.

The classic arena must stay byte-identical to the faithful port; the torus
only changes what happens once a curve would have left the plane.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from agents.simulate_tool import simulate
from graphwar_sim import Game, config


def _fire(arena: str, expr: str):  # type: ignore[no-untyped-def]
    game = Game.create(5, num_soldiers=2)
    game.arena = arena
    return game.fire(expr)


def test_classic_is_the_default_and_dies_out_of_bounds() -> None:
    game = Game.create(5, num_soldiers=2)
    assert game.arena == "classic" and not game.wraps()
    result = game.fire("20x")
    assert result.stop == "oob"


def test_torus_wraps_and_keeps_flying() -> None:
    classic = _fire("classic", "20x")
    torus = _fire("torus", "20x")
    assert torus.num_steps > classic.num_steps
    # Identical while the classic shot is still on the plane (it can stray
    # under a pixel past the edge before Java's (int) cast notices).
    on_plane = [
        i
        for i, (px, py) in enumerate(classic.points)
        if 0 <= px < config.PLANE_LENGTH and 0 <= py < config.PLANE_HEIGHT
    ]
    assert len(on_plane) > 100
    for i in on_plane:
        assert torus.points[i] == classic.points[i]
    for px, py in torus.points:
        assert 0 <= px < config.PLANE_LENGTH
        assert 0 <= py < config.PLANE_HEIGHT


def test_simulate_follows_the_arena() -> None:
    game = Game.create(5, num_soldiers=2)
    flat = simulate(game, "20x").num_steps
    game.arena = "torus"
    assert simulate(game, "20x").num_steps > flat


def test_new_game_arena_param() -> None:
    from ui.server import app

    client = TestClient(app)
    body = client.post("/api/new_game", json={"seed": 5, "arena": "torus"}).json()
    assert body["arena"] == "torus"
    bad = client.post("/api/new_game", json={"seed": 5, "arena": "mirror"})
    assert bad.status_code == 400
    assert client.post("/api/new_game", json={"seed": 5}).json()["arena"] == "classic"
