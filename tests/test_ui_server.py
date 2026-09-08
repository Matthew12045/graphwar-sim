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

import json
from collections.abc import Iterator

import pytest
from fastapi.responses import JSONResponse
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


def test_board_carries_carves() -> None:
    """Destructible terrain: fresh board has empty carves; one fire appends one."""
    from ui.server import _board_json

    game = Game.create(seed=21)
    board = _board_json(game)
    assert board["carves"] == []
    game.fire("x/2")
    board = _board_json(game)
    assert len(board["carves"]) == 1
    assert len(board["carves"][0]) == 3


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

    # TEAM1 (solver) is up again — /api/fire is gated for agent sides (G2).
    fire = client.post("/api/fire", json={"func_str": "x/2"})
    assert fire.status_code == 409
    assert fire.json()["error"] == "agent_turn"


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


# --- Slice E: persona / hybrid mode passthrough -------------------------------


def test_persona_mode_round_trips_through_new_game(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`llm:<model>@<persona>` constructs eagerly (fail fast) and echoes the
    full wire string back so the UI can round-trip it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-token")
    response = client.post(
        "/api/new_game",
        json={"seed": 21, "team_modes": {"team1": "llm:some-model@sniper", "team2": "human"}},
    )
    assert response.status_code == 200
    assert response.json()["team_modes"] == {
        "team1": "llm:some-model@sniper",
        "team2": "human",
    }
    state = client.get("/api/state").json()
    assert state["team_modes"]["team1"] == "llm:some-model@sniper"


def test_unknown_persona_is_a_400(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-token")
    response = client.post(
        "/api/new_game",
        json={"team_modes": {"team1": "llm:some-model@gremlin", "team2": "human"}},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "bad_team_modes"
    assert "unknown persona" in body["detail"]


def test_bare_hybrid_mode_is_a_400(client: TestClient) -> None:
    response = client.post(
        "/api/new_game",
        json={"team_modes": {"team1": "hybrid:", "team2": "human"}},
    )
    assert response.status_code == 400
    assert "hybrid: mode needs a model name" in response.json()["detail"]


# --- Slice C: cancellation ---------------------------------------------------
#
# A blocked fake agent holds the module lock inside /api/agent_turn;
# /api/new_game sets the cancel event BEFORE waiting on the lock, the agent
# unwinds (TurnCancelled -> 409 aborted), and the fresh match builds. These
# call the endpoint functions directly: a real cross-request hang test needs
# two concurrent callers, which TestClient's portal does not model.


def test_new_game_cancels_a_blocked_agent_turn() -> None:
    import threading
    import time

    import ui.server as server_module
    from agents import AgentStats, TurnCancelled
    from ui.server import NewGameBody, agent_turn, new_game

    created = new_game(NewGameBody(seed=21))
    assert isinstance(created, dict)

    entered = threading.Event()

    class _BlockedAgent:
        name = "blocked"

        def act(self, game: Game, obs: object) -> str:
            entered.set()
            deadline = time.monotonic() + 5.0
            while not server_module._cancel_requested.is_set():
                if time.monotonic() > deadline:
                    raise AssertionError("cancel event never set")
                time.sleep(0.005)
            raise TurnCancelled()

        def stats(self) -> AgentStats:
            return AgentStats()

    # In-place swap: the agent turn reads _team_agents under the lock, and
    # the cancelling new_game below replaces the dict wholesale afterwards.
    server_module._team_agents[1] = _BlockedAgent()  # TEAM1

    results: dict[str, object] = {}
    thread = threading.Thread(target=lambda: results.update(turn=agent_turn()))
    thread.start()
    assert entered.wait(timeout=5.0), "the fake agent never entered act()"

    response = new_game(NewGameBody())  # sets the event pre-lock, then builds
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "agent_turn never unwound"

    turn_response = results["turn"]
    assert isinstance(turn_response, JSONResponse)
    assert turn_response.status_code == 409
    assert json.loads(turn_response.body)["error"] == "aborted"

    assert isinstance(response, dict)  # the fresh match built
    assert response["team_modes"] == {"team1": "human", "team2": "human"}
    assert response["turns_played"] == 0  # the aborted turn fired nothing
    assert not server_module._cancel_requested.is_set()  # flag consumed, not stuck


def test_failed_agent_construction_does_not_stick_the_cancel_flag(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handover's race: new_game sets the event pre-lock; a 400 (bad team
    modes) must still leave it CLEAR for future turns."""
    import ui.server as server_module

    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = client.post(
        "/api/new_game",
        json={"team_modes": {"team1": "llm:some-model", "team2": "human"}},
    )
    assert response.status_code == 400
    assert not server_module._cancel_requested.is_set()


# --- Slice B: live activity feed -------------------------------------------------


def test_activity_serves_events_while_the_game_lock_is_held() -> None:
    """The feed endpoint answers from its OWN lock while a blocked fake agent
    holds the game lock inside agent_turn — the polling feed never stalls."""
    import threading
    import time

    import ui.server as server_module
    from agents import AgentStats, TurnCancelled
    from ui.server import NewGameBody, activity, agent_turn, new_game

    new_game(NewGameBody(seed=21))
    assert activity(since=0)["events"] == []  # the ring reset for the match

    entered = threading.Event()

    class _FeedingBlockedAgent:
        name = "feeder"

        def set_event_sink(self, sink: object) -> None:
            self._sink = sink

        def act(self, game: Game, obs: object) -> str:
            self._sink("tool_call", {"expr": "0*x"})  # type: ignore[operator]
            entered.set()
            deadline = time.monotonic() + 5.0
            while not server_module._cancel_requested.is_set():
                if time.monotonic() > deadline:
                    raise AssertionError("cancel event never set")
                time.sleep(0.005)
            raise TurnCancelled()

        def stats(self) -> AgentStats:
            return AgentStats()

    server_module._team_agents[1] = _FeedingBlockedAgent()  # TEAM1

    results: dict[str, object] = {}
    thread = threading.Thread(target=lambda: results.update(turn=agent_turn()))
    thread.start()
    assert entered.wait(timeout=5.0), "the fake agent never entered act()"

    # The game lock is HELD by agent_turn here; /api/activity still answers.
    payload = activity(since=0)
    assert len(payload["events"]) == 1
    event = payload["events"][0]
    assert event["kind"] == "tool_call"
    assert event["agent"] == "feeder"
    assert event["expr"] == "0*x"
    assert payload["next"] == event["id"]
    since_cursor = payload["next"]

    new_game(NewGameBody())  # cancels the blocked turn and builds a fresh match
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    # The `since` cursor skips old events; the ring reset on the fresh match
    # keeps ids monotonic.
    after = activity(since=since_cursor)
    assert after["events"] == []
    assert after["next"] == since_cursor


def test_activity_endpoint_via_testclient(client: TestClient) -> None:
    """The HTTP surface: empty for a fresh match, shape {"events", "next"}."""
    response = client.get("/api/activity", params={"since": 0})
    assert response.status_code == 200
    body = response.json()
    assert body == {"events": [], "next": 0}


# --- Demo driver swap + thinking bubble plan (G1/G2/G4/G6) ----------------------


def test_set_modes_swaps_mid_match_preserving_board(client: TestClient) -> None:
    """G1: 200-swap mid-match leaves board/turns intact and drivers live."""
    client.post(
        "/api/new_game",
        json={"seed": 21, "team_modes": {"team1": "human", "team2": "human"}},
    )
    before_fire = client.post("/api/fire", json={"func_str": "x/2"}).json()
    assert before_fire["board"]["turns_played"] == 1
    assert before_fire["board"]["current_turn"] == 1
    circles_before = before_fire["board"]["circles"]

    response = client.post("/api/set_modes", json={"team1": "solver"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["team_modes"] == {"team1": "solver", "team2": "human"}
    assert body["board"]["circles"] == circles_before
    assert body["board"]["turns_played"] == 1
    assert body["board"]["current_turn"] == 1
    assert body["board"]["max_turns"] is None

    # Missing keys leave that side unchanged; the swapped driver is live:
    # TEAM1 is solver-driven but it is TEAM2 (human) up, so agent_turn 409s
    # as human_turn; after a human shot TEAM1 is up and fire is gated.
    state = client.get("/api/state").json()
    assert state["team_modes"] == {"team1": "solver", "team2": "human"}
    human_fire = client.post("/api/fire", json={"func_str": "x/2"})
    assert human_fire.status_code == 200, human_fire.text
    assert human_fire.json()["board"]["current_turn"] == 0
    gated = client.post("/api/fire", json={"func_str": "x/2"})
    assert gated.status_code == 409
    assert gated.json()["error"] == "agent_turn"


def test_set_modes_400_leaves_drivers_intact(client: TestClient) -> None:
    """G1: 400 revert semantics — a bad swap changes nothing."""
    client.post("/api/new_game", json={"seed": 21})
    response = client.post("/api/set_modes", json={"team1": "nope"})
    assert response.status_code == 400
    assert response.json()["error"] == "bad_team_modes"
    assert client.get("/api/state").json()["team_modes"] == {
        "team1": "human",
        "team2": "human",
    }

    bare = client.post("/api/set_modes", json={"team1": "hybrid:"})
    assert bare.status_code == 400
    assert "hybrid: mode needs a model name" in bare.json()["detail"]
    assert client.get("/api/state").json()["team_modes"] == {
        "team1": "human",
        "team2": "human",
    }


def test_fire_during_agent_turn_is_409(client: TestClient) -> None:
    """G2: /api/fire on an agent-driven side is a 409, turn untouched."""
    client.post(
        "/api/new_game",
        json={"seed": 21, "team_modes": {"team1": "solver", "team2": "human"}},
    )
    before = client.get("/api/state").json()
    assert before["current_turn"] == 0
    response = client.post("/api/fire", json={"func_str": "x/2"})
    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "agent_turn"
    assert "Step/Play" in body["detail"]
    assert body["board"]["current_turn"] == 0
    assert body["board"]["turns_played"] == 0
    after = client.get("/api/state").json()
    assert after["current_turn"] == 0
    assert after["turns_played"] == 0

    # The human side still fires normally when it is up.
    client.post(
        "/api/new_game",
        json={"seed": 21, "team_modes": {"team1": "human", "team2": "solver"}},
    )
    ok_fire = client.post("/api/fire", json={"func_str": "x/2"})
    assert ok_fire.status_code == 200, ok_fire.text


def test_state_includes_max_turns(client: TestClient) -> None:
    """G6: /api/state carries the turn budget (additive top-level + board)."""
    client.post("/api/new_game", json={"seed": 21, "max_turns": 30})
    state = client.get("/api/state").json()
    assert state["max_turns"] == 30
    assert state["turns_played"] == 0
    # The board payload travels flat (same shape as new_game's spread board).
    assert state["teams"] and state["current_turn"] == 0


def test_agent_for_mode_carries_cancel_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G4: llm:/hybrid: agents built through _agent_for_mode carry the server
    cancel callback; baselines carry none. White-box on privates matches repo
    test style."""
    import ui.server as server_module
    from ui.server import _agent_for_mode

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-token")
    server_module._cancel_requested.clear()
    llm_agent = _agent_for_mode("llm:some-model")
    assert getattr(llm_agent, "_cancel_requested", None) is not None
    assert llm_agent._cancel_requested() is False  # type: ignore[operator]
    hybrid_agent = _agent_for_mode("hybrid:some-model")
    assert getattr(hybrid_agent, "_cancel_requested", None) is not None

    baseline = _agent_for_mode("solver")
    assert getattr(baseline, "_cancel_requested", None) is None

    server_module._cancel_requested.set()
    try:
        assert llm_agent._cancel_requested() is True  # type: ignore[operator]
    finally:
        server_module._cancel_requested.clear()


def test_server_cancel_event_fires_a_scripted_llm_turn() -> None:
    """G4: a set server event unwinds a scripted LLM turn (TurnCancelled)
    before any API round-trip — single-shot turns check the flag up front."""
    import ui.server as server_module
    from agents import TurnCancelled
    from agents.llm_agent import LLMAgent
    from agents.observation import observe
    from graphwar_sim.state import Game

    class _TextBlock:
        def __init__(self, text: str) -> None:
            self.type = "text"
            self.text = text

    class _ScriptedResponse:
        def __init__(self) -> None:
            self.stop_reason = "end_turn"
            self.content = [_TextBlock("0*x")]

    class _FlippingMessages:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: object) -> _ScriptedResponse:
            self.calls += 1
            return _ScriptedResponse()

    class _FlippingClient:
        def __init__(self) -> None:
            self.messages = _FlippingMessages()

    server_module._cancel_requested.clear()
    try:
        agent = LLMAgent(
            model="fake-model",
            client=_FlippingClient(),  # type: ignore[arg-type]
            cancel_requested=server_module._cancel_requested.is_set,
        )
        server_module._cancel_requested.set()  # cancel arrives before the turn
        game = Game.create(21)
        with pytest.raises(TurnCancelled):
            agent.act(game, observe(game))
        assert agent._client.messages.calls == 0  # no API round-trip happened
    finally:
        server_module._cancel_requested.clear()
