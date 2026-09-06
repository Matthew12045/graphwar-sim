"""Tests for the UI server (``ui/server.py``) using FastAPI's TestClient.

These exercise the *additive* layer only: every request is served from the
unchanged ``graphwar_sim`` core. Acceptance for visual fidelity is manual
(side-by-side against the reference screenshots, per implement.md); here we
pin the API contract:

- new game -> fire a known-safe function -> 200 + expected JSON shape
- fire a malformed function -> the distinct 400 error path, never a 500
- a full game to completion doesn't crash (driven by the repo's own
  deterministic solver, mirrored locally since server and test run the same
  ``Game.create(seed)`` + ``play_turn`` sequence)
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from agents.observation import observe
from agents.solver_agent import SolverAgent
from graphwar_sim.state import Game
from ui.server import app

# Seed 3: the solver ends this map on the very first shot (verified against
# the real core); seeds like 1/2/4/7 stall with dud shots, so they are
# avoided for completion tests.
_COMPLETION_SEED = 3


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        response = test_client.post("/api/new_game", json={"seed": 21})
        assert response.status_code == 200
        yield test_client


def _fire(client: TestClient, func_str: str) -> dict:
    response = client.post("/api/fire", json={"func_str": func_str})
    assert response.status_code == 200, response.text
    return response.json()


def test_new_game_shape(client: TestClient) -> None:
    data = client.post("/api/new_game", json={"seed": 7}).json()

    assert data["seed"] == 7
    assert len(data["teams"]) == 2
    for j, team in enumerate(data["teams"]):
        assert team["label"] == f"Player {j + 1}"
        assert team["team"] in (1, 2)
        assert len(team["soldiers"]) == 2  # INITIAL_NUM_SOLDIERS
        for k, soldier in enumerate(team["soldiers"]):
            assert soldier["player_index"] == j
            assert soldier["soldier_index"] == k
            assert soldier["alive"] is True
            assert 0 <= soldier["x"] < 770
            assert 0 <= soldier["y"] < 450
    assert data["current_turn"] == 0
    assert data["finished"] is False
    assert data["winner"] is None
    assert data["shooter"]["player_index"] == 0
    assert data["shooter"]["label"] == "Player 1"
    assert data["shooter"]["inverted"] is False
    assert isinstance(data["circles"], list)
    for circle in data["circles"]:
        assert len(circle) == 3

    # Determinism: the same seed regenerates the same map.
    again = client.post("/api/new_game", json={"seed": 7}).json()
    assert again["circles"] == data["circles"]
    assert again["teams"] == data["teams"]


def test_num_soldiers_clamped(client: TestClient) -> None:
    data = client.post("/api/new_game", json={"seed": 1, "num_soldiers": 99}).json()
    assert all(len(team["soldiers"]) == 4 for team in data["teams"])
    data = client.post("/api/new_game", json={"seed": 1, "num_soldiers": 0}).json()
    assert all(len(team["soldiers"]) == 1 for team in data["teams"])


def test_fire_shape_and_turn_flow(client: TestClient) -> None:
    data = _fire(client, "x/2")

    # Shot contract (ShotResult as JSON; non-finite points travel as null).
    shot = data["shot"]
    assert shot["num_steps"] == len(shot["points"])
    assert shot["num_steps"] >= 1
    for point in shot["points"]:
        assert len(point) == 2
    assert shot["last_x"] is None or isinstance(shot["last_x"], float)
    for hit in shot["hits"]:
        player_index, soldier_index, position = hit
        assert 0 <= position < shot["num_steps"]
        assert 0 <= player_index < 2
        assert 0 <= soldier_index < 2

    # The shooter is the pre-turn player; the board already advanced.
    assert data["shooter"]["label"] == "Player 1"
    assert data["board"]["current_turn"] == 1
    assert data["func_str"] == "x/2"
    assert data["game_over"] is data["board"]["finished"]
    assert data["winner"] == data["board"]["winner"]

    # Display-only start angle: atan(0.5), the curve's own tangent.
    assert data["start_angle"] == pytest.approx(0.4636, abs=1e-2)

    # Malformed input is a distinct 400 shape and does NOT advance the turn.
    current = client.get("/api/state").json()
    response = client.post("/api/fire", json={"func_str": ")"})
    assert response.status_code == 400
    err = response.json()
    assert err["error"] == "malformed_function"
    assert "Python-side" in err["detail"]
    assert client.get("/api/state").json() == current

    # A second safe shot: TEAM2 now (mirrored), same constant tangent.
    data = _fire(client, "x/2")
    assert data["shooter"]["label"] == "Player 2"
    assert data["shooter"]["inverted"] is True
    assert data["board"]["current_turn"] == 0
    assert data["start_angle"] == pytest.approx(0.4636, abs=1e-2)
    # No kill was possible with x/2 (verified against the core), so the
    # game is still live and nobody died.
    assert data["game_over"] is False
    assert all(s["alive"] for t in data["board"]["teams"] for s in t["soldiers"])


def test_fire_after_game_over_is_409(client: TestClient) -> None:
    client.post("/api/new_game", json={"seed": _COMPLETION_SEED})
    mirror = Game.create(seed=_COMPLETION_SEED)
    expr = SolverAgent().act(mirror, observe(mirror))
    data = _fire(client, expr)
    assert data["game_over"] is True
    assert data["winner"] in (1, 2)
    mirror.play_turn(expr)

    response = client.post("/api/fire", json={"func_str": "x"})
    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "game_over"
    assert body["board"]["finished"] is True


def test_full_game_to_completion_does_not_crash(client: TestClient) -> None:
    client.post("/api/new_game", json={"seed": _COMPLETION_SEED})
    mirror = Game.create(seed=_COMPLETION_SEED)
    agent = SolverAgent()

    for _turn in range(30):
        expr = agent.act(mirror, observe(mirror))
        response = client.post("/api/fire", json={"func_str": expr})
        assert response.status_code == 200, response.text
        data = response.json()

        # Invariant: exactly the hit soldiers died on the returned board.
        for player_index, soldier_index, _pos in data["shot"]["hits"]:
            soldier = data["board"]["teams"][player_index]["soldiers"][soldier_index]
            assert soldier["alive"] is False

        mirror.play_turn(expr)
        assert data["board"]["finished"] == mirror.finished()

        if data["game_over"]:
            winner = data["winner"]
            assert winner in (1, 2)
            losing_team = 2 if winner == 1 else 1
            alive = [
                s["alive"]
                for team in data["board"]["teams"]
                if team["team"] == losing_team
                for s in team["soldiers"]
            ]
            assert not any(alive)
            break
    else:
        pytest.fail("seed-3 game did not finish within 30 solver turns")


def test_static_frontend_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Graphwar" in response.text
    assert response.headers["content-type"].startswith("text/html")
    for asset in ("/style.css", "/app.js"):
        assert client.get(asset).status_code == 200


# --- M5.4: team modes, agent turns, turn cap ---------------------------------


def test_team_modes_default_human(client: TestClient) -> None:
    """Omitted team_modes defaults both sides to human (old flow pinned)."""
    data = client.post("/api/new_game", json={"seed": 21}).json()
    assert data["team_modes"] == {"team1": "human", "team2": "human"}
    assert data["max_turns"] is None
    assert data["turns_played"] == 0
    state = client.get("/api/state").json()
    assert state["team_modes"] == {"team1": "human", "team2": "human"}


def test_agent_turn_plays_deterministic_solver(client: TestClient) -> None:
    from eval.metrics import ShotOutcome

    data = client.post(
        "/api/new_game",
        json={
            "seed": 21,
            "team_modes": {"team1": "solver", "team2": "random"},
            "max_turns": 30,
        },
    ).json()
    assert data["team_modes"] == {"team1": "solver", "team2": "random"}
    assert data["max_turns"] == 30

    response = client.post("/api/agent_turn")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["agent"] == "solver"
    assert body["outcome"] in set(ShotOutcome)
    assert body["func_str"] and body["shot"]["num_steps"] >= 1
    assert body["shooter"]["label"] == "Player 1"
    assert body["board"]["current_turn"] == 1  # the turn advanced
    assert body["board"]["turns_played"] == 1
    assert body["game_over"] is False
    assert body["draw_reason"] is None

    # The next side (random) plays too — one turn per request.
    response = client.post("/api/agent_turn")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["agent"] == "random"
    assert body["board"]["turns_played"] == 2

    # /api/fire still drives the human sides (team1 is up again, human).
    fire = client.post("/api/fire", json={"func_str": "x/2"})
    assert fire.status_code == 200, fire.text


def test_agent_turn_on_human_side_is_409(client: TestClient) -> None:
    client.post("/api/new_game", json={"seed": 21})
    response = client.post("/api/agent_turn")
    assert response.status_code == 409
    assert response.json()["error"] == "human_turn"


def test_turn_cap_draw_reports_turn_cap(client: TestClient) -> None:
    client.post(
        "/api/new_game",
        json={
            "seed": 21,
            "team_modes": {"team1": "solver", "team2": "solver"},
            "max_turns": 1,
        },
    )
    first = client.post("/api/agent_turn")
    assert first.status_code == 200, first.text
    # The shot that crosses the cap reports the draw already (post-turn check).
    assert first.json()["game_over"] is True
    assert first.json()["draw_reason"] == "TURN_CAP"

    second = client.post("/api/agent_turn")
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["game_over"] is True
    assert body["winner"] is None
    assert body["draw_reason"] == "TURN_CAP"
    assert "shot" not in body  # no shot is fired at the cap
    assert body["board"]["turns_played"] == 1
    assert body["board"]["max_turns"] == 1

    # /api/fire reports the same cap shape (no shot fired at the cap).
    fire = client.post("/api/fire", json={"func_str": "x"})
    assert fire.status_code == 200
    assert fire.json()["draw_reason"] == "TURN_CAP"


def test_llm_mode_without_auth_env_is_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = client.post(
        "/api/new_game",
        json={"team_modes": {"team1": "llm:qwen3.8-27b-fp8", "team2": "human"}},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "bad_team_modes"
    assert "ANTHROPIC_AUTH_TOKEN" in body["detail"]
    # The lazy match is untouched: state still serves the previous game.
    assert client.get("/api/state").json()["team_modes"] == {"team1": "human", "team2": "human"}


def test_unknown_mode_is_400(client: TestClient) -> None:
    response = client.post(
        "/api/new_game",
        json={"team_modes": {"team1": "nope", "team2": "human"}},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "bad_team_modes"
    assert "unknown agent in roster: nope" in body["detail"]


def test_bad_max_turns_is_400(client: TestClient) -> None:
    response = client.post("/api/new_game", json={"max_turns": 0})
    assert response.status_code == 400
    assert response.json()["error"] == "bad_max_turns"
