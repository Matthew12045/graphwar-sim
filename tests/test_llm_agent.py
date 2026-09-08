"""Tests for :class:`agents.llm_agent.LLMAgent` (single-shot: one API call,
no tools, first output fires).

Zero network: every test injects a fake client that mirrors ONLY the minimal
response surface the agent relies on — ``messages.create(...)`` returning an
object with ``.stop_reason`` and ``.content`` blocks (``.type == "text"``
with ``.text``).
"""

from __future__ import annotations

from typing import Any

import pytest

from agents import StraightShotAgent
from agents.llm_agent import (
    _SYSTEM_PROMPT,
    SAFE_DUD,
    LLMAgent,
)
from agents.observation import observe
from eval.runner import MatchConfig, play_match
from graphwar_sim import Game

# --- fake client (minimal Anthropic surface) ---------------------------------


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, stop_reason: str, content: list[Any]) -> None:
        self.stop_reason = stop_reason
        self.content = content


class _FakeMessages:
    def __init__(self, responses: list[_Response]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        # Snapshot the conversation: the agent mutates one shared list across
        # rounds, so recording the live reference would alias every call.
        snapshot = [
            {
                "role": message["role"],
                "content": (
                    list(message["content"])
                    if isinstance(message["content"], list)
                    else message["content"]
                ),
            }
            for message in kwargs["messages"]
        ]
        self.calls.append({**kwargs, "messages": snapshot})
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]  # repeat the last script forever (long matches)


class _FakeClient:
    def __init__(self, responses: list[_Response]) -> None:
        self.messages = _FakeMessages(responses)


class _DyingAfterMessages:
    """Serves the scripted responses, then raises a retryable transport error
    on every further call (the gateway-killed-every-regeneration case)."""

    def __init__(self, responses: list[_Response]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs: Any) -> _Response:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        raise RemoteProtocolError("peer closed connection")


class RemoteProtocolError(Exception):
    """Duck-typed name the agent's retry set knows (no SDK import)."""


class _DyingAfterClient:
    def __init__(self, responses: list[_Response]) -> None:
        self.messages = _DyingAfterMessages(responses)


class _FakeStream:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.opened = False

    def __enter__(self) -> _FakeStream:
        self.opened = True
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def get_final_message(self) -> _Response:
        return self._response


class _StreamingFakeMessages:
    """A client exposing ``messages.stream`` (the real SDK's surface) — the
    agent must prefer it over ``messages.create`` (Cloudflare 120s proxy
    timeout on the gateway makes streaming mandatory for long thinking)."""

    def __init__(self, response: _Response) -> None:
        self._response = response
        self.stream_kwargs: dict[str, Any] | None = None

    def stream(self, **kwargs: Any) -> _FakeStream:
        self.stream_kwargs = kwargs
        return _FakeStream(self._response)

    def create(self, **kwargs: Any) -> _Response:  # pragma: no cover - must not run
        raise AssertionError("create() called although stream() is available")


class _StreamingFakeClient:
    def __init__(self, response: _Response) -> None:
        self.messages = _StreamingFakeMessages(response)


def _text(text: str) -> _Response:
    return _Response("end_turn", [_TextBlock(text)])


def _agent(responses: list[_Response], **kwargs: Any) -> LLMAgent:
    return LLMAgent(model="fake-model", client=_FakeClient(responses), **kwargs)


def _game_and_obs() -> tuple[Game, Any]:
    game = Game.create(5, num_soldiers=1)
    return game, observe(game)


def _seed21_game_and_obs() -> tuple[Game, Any]:
    """The live-diagnosis board: muzzle rock just right of the shooter, an
    ally the scripted teammate-hit commit curve strikes."""
    game = Game.create(21, num_soldiers=2)
    return game, observe(game)


# --- 1. well-formed first response -------------------------------------------


def test_well_formed_first_response_verbatim_zero_counters() -> None:
    agent = _agent([_text("x^2/4")])
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "x^2/4"
    stats = agent.stats()
    assert stats.parse_failures == 0
    assert stats.retries == 0
    # Single-shot, no oracle: zero calls, zero overrides, exactly one round.
    assert stats.simulate_calls == 0
    assert stats.simulate_denied == 0
    assert stats.guardrail_overrides == 0
    assert len(agent._client.messages.calls) == 1


def test_single_call_offers_no_tools() -> None:
    """The turn is one text-only round: no tools key, no tool_choice — the
    model MUST answer in text and whatever it outputs first is fired."""
    agent = _agent([_text("0.05*x")])
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    calls = agent._client.messages.calls
    assert len(calls) == 1
    assert "tools" not in calls[0]
    assert "tool_choice" not in calls[0]


def test_turn_message_carries_clear_lanes_not_a_budget() -> None:
    """The accuracy strategy is context, not iteration: the turn message
    carries deterministic CLEAR LANE intervals (same world frame as the
    emission) and no simulate-budget line."""
    agent = _agent([_text("0*x")])
    game, obs = _game_and_obs()
    agent.act(game, obs)
    first_message = agent._client.messages.calls[0]["messages"][0]["content"]
    assert "shooter (you):" in first_message
    assert "enemies (nearest first):" in first_message
    assert "CLEAR LANES" in first_message
    assert "simulate" not in first_message
    # Lane lines run from the muzzle toward the enemies in world x.
    lane_lines = [line for line in first_message.splitlines() if line.startswith("x=")]
    assert len(lane_lines) >= 2
    assert all(".." in line or "BLOCKED" in line for line in lane_lines)


# --- 2. malformed first output -> safe dud (no second attempt) ----------------


def test_malformed_first_output_fires_the_safe_dud() -> None:
    """Single-shot means single-shot: a malformed emission counts
    parse_failures/retries once and fires the dud — no correction round."""
    agent = _agent([_text("((")])
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == SAFE_DUD
    stats = agent.stats()
    assert stats.parse_failures == 1
    assert stats.retries == 1
    assert stats.simulate_calls == 0
    assert len(agent._client.messages.calls) == 1  # no re-call


def test_empty_output_fires_the_safe_dud() -> None:
    """A max_tokens stop with no text (hidden-thinking burn) fires the dud."""
    agent = _agent([_Response("max_tokens", [])])
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == SAFE_DUD
    stats = agent.stats()
    assert stats.parse_failures == 1
    assert stats.retries == 1


def test_y_equals_prefix_and_fences_are_stripped() -> None:
    agent = _agent([_text("```\ny = 0.05*x\n```")])
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"


def test_unrecoverable_api_failure_degrades_to_safe_dud() -> None:
    """After the API retries are exhausted the turn degrades to the safe dud
    instead of crashing the match (user-locked live-validation decision)."""

    class RemoteProtocolError(Exception):  # duck-typed name the retry set knows
        pass

    class _DyingMessages:
        def __init__(self) -> None:
            self.attempts = 0

        def create(self, **kwargs: Any) -> _Response:
            self.attempts += 1
            raise RemoteProtocolError("peer closed connection")

    class _DyingClient:
        def __init__(self) -> None:
            self.messages = _DyingMessages()

    agent = LLMAgent(model="fake-model", client=_DyingClient())
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == SAFE_DUD
    stats = agent.stats()
    assert stats.parse_failures == 1
    assert stats.retries == 1
    assert agent._client.messages.attempts == 3  # 1 call + _MAX_API_RETRIES


# --- 4. constructor fails fast without auth env vars ---------------------------


def test_streaming_client_is_preferred_when_available() -> None:
    response = _Response("end_turn", [_TextBlock("0.05*x")])
    agent = LLMAgent(model="fake-model", client=_StreamingFakeClient(response))
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    kwargs = agent._client.messages.stream_kwargs
    assert kwargs is not None
    assert kwargs["max_tokens"] == 128000  # the model's full context budget
    assert kwargs["system"] == _SYSTEM_PROMPT
    assert "tools" not in kwargs  # single-shot: no tools offered


def test_reasoning_effort_auto_with_gateway_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ANTHROPIC_BASE_URL set (the 9arm gateway case) the reasoning
    effort (the measured fix for the budget-burn wall) rides every call."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.9arm.co")
    agent = _agent([_text("0*x")])
    game, obs = _game_and_obs()
    agent.act(game, obs)
    kwargs = agent._client.messages.calls[0]
    assert kwargs["extra_body"] == {"reasoning_effort": "medium"}


def test_reasoning_effort_absent_for_real_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a gateway base_url (real Anthropic) no extra_body is sent —
    the real API rejects unknown body fields. An explicit override wins."""
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    agent = _agent([_text("0*x")])
    game, obs = _game_and_obs()
    agent.act(game, obs)
    assert "extra_body" not in agent._client.messages.calls[0]

    forced = LLMAgent(
        model="fake-model",
        client=_FakeClient([_text("0*x")]),
        reasoning_effort="low",
    )
    forced.act(game, obs)
    assert forced._client.messages.calls[0]["extra_body"] == {"reasoning_effort": "low"}


def test_missing_auth_env_vars_raise_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        LLMAgent(model="fake-model")
    message = str(excinfo.value)
    assert "ANTHROPIC_AUTH_TOKEN" in message
    assert "ANTHROPIC_API_KEY" in message


def test_agent_name_is_model_scoped() -> None:
    agent = _agent([_text("0*x")])
    assert agent.name == "llm:fake-model"


# --- 7. system prompt quotes the exact parser whitelist ------------------------


def test_system_prompt_contains_exact_whitelist() -> None:
    for token in ("sqrt", "log", "ln", "abs", "sin", "cos", "tan", "x ONLY", "e pi"):
        assert token in _SYSTEM_PROMPT, token
    assert 'no "y ="' in _SYSTEM_PROMPT  # the bare-expression output contract


# --- 6. the pure oracle (used by persona verifiers, not by the agent) --------

# The agent itself makes zero oracle calls; the persona machine verifiers
# still verify through the pure agents.simulate_tool.simulate. These pin
# that oracle's contract.


def test_unparseable_simulate_has_null_telemetry() -> None:
    from agents.simulate_tool import simulate as oracle

    game, _obs = _game_and_obs()
    result = oracle(game, "((")
    assert result.parseable is False
    assert result.nearest_miss is None
    assert result.stop_reason is None


def test_telemetry_reports_termination_reason() -> None:
    """Deterministic seed: the committed probe on seed 21 carries a full
    miss report (distance, direction, stop x, reason)."""
    from agents.simulate_tool import simulate as oracle

    game = Game.create(21, num_soldiers=2)
    result = oracle(game, "0*x")  # horizontal line at the shooter's y
    assert result.parseable is True
    assert result.stop_reason in {"hit", "terrain", "off_map", "short", "passed"}
    assert result.nearest_miss is not None and result.nearest_miss >= 0
    assert result.miss_direction in {"high", "low"}
    assert result.stopped_at_x is not None


# --- 7. first output fires verbatim — even a teammate hit (no guardrail) ----
#
# Single-shot means what it says: the model polices itself with the CLEAR
# LANES. Board fact (deterministic seed 21, num_soldiers=2):
# "-1.4117(x+18.117)" is a line through the muzzle that strikes the ALLY.


def test_first_output_fires_verbatim_without_oracle() -> None:
    """No probes, no guardrail, no override: the first output fires as-is
    and the runner records the outcome (a teammate hit here — the model's
    problem, not the harness's)."""
    agent = _agent([_text("-1.4117(x+18.117)")])
    game, obs = _seed21_game_and_obs()
    assert agent.act(game, obs) == "-1.4117(x+18.117)"
    stats = agent.stats()
    assert stats.guardrail_overrides == 0
    assert stats.simulate_calls == 0
    assert len(agent._client.messages.calls) == 1


def test_first_output_flows_through_play_match() -> None:
    """End-to-end: the single-shot emission fires in a real match."""
    agent = _agent([_text("0.05*x")])
    result = play_match(21, agent, StraightShotAgent(), MatchConfig(num_soldiers=2, max_turns=2))
    assert result.stats["llm:fake-model"].guardrail_overrides == 0


# --- 8. muzzle-wall warning in the turn message (M5.4 Fix 3b) -----------------


def _obs_with_terrain(blocks: tuple[tuple[float, float], ...]) -> Any:
    from agents.base import FRAME_CENTERED_WORLD, Observation

    return Observation(
        frame=FRAME_CENTERED_WORLD,
        turn_index=0,
        team_id=1,
        shooter=(-18.1, 7.5),
        own_soldiers=((-18.1, 7.5),),
        enemy_soldiers=((16.9, 14.1),),
        terrain_blocks=blocks,
        ascii_board="",
    )


def test_turn_message_warns_about_a_muzzle_wall() -> None:
    """A terrain block just RIGHT of the muzzle (within ~2.5 x, ~2 y) gets a
    deterministic descending-launch warning — the seed-21 live failure (every
    early probe ascended into the muzzle rock)."""
    from agents.llm_agent import _turn_message
    from graphwar_sim import Game as _Game

    game = _Game.create(5, num_soldiers=1)
    obs = _obs_with_terrain(((-17.2, 7.8), (5.0, -5.0)))
    message = _turn_message(game, obs)
    assert "terrain wall at (x~-17.2, y~7.8) just right of your muzzle" in message
    assert "launch DESCENDING" in message


def test_turn_message_no_warning_for_far_or_left_terrain() -> None:
    """Terrain left of the muzzle, or beyond the scan radii, warns nothing."""
    from agents.llm_agent import _turn_message
    from graphwar_sim import Game as _Game

    game = _Game.create(5, num_soldiers=1)
    for blocks in (
        ((-20.0, 7.5), (5.0, -5.0)),  # left of the muzzle
        ((-14.0, 7.5), (5.0, -5.0)),  # >2.5 right
        ((-17.5, 12.0), (5.0, -5.0)),  # >2 above
    ):
        message = _turn_message(game, _obs_with_terrain(blocks))
        assert "terrain wall" not in message, blocks


def test_seed21_turn_message_carries_the_muzzle_warning() -> None:
    """The live-diagnosis board triggers the warning (integration of the
    construction with the real observation)."""
    from agents.llm_agent import _turn_message

    game, obs = _seed21_game_and_obs()
    message = _turn_message(game, obs)
    assert "just right of your muzzle" in message


def test_clear_lanes_agree_with_the_live_grid() -> None:
    """Spot-check the deterministic hint: at the muzzle's own x the lane
    holds the muzzle, and a fully walled sample reads BLOCKED."""
    from agents.llm_agent import _clear_lane_lines

    game, obs = _game_and_obs()
    lines = _clear_lane_lines(game, obs)
    assert lines, "expected lane samples toward the enemy"
    first = lines[0]
    assert first.startswith(f"x={obs.shooter[0]:.1f}: ")
    assert "BLOCKED" not in first  # the muzzle column holds the muzzle
    assert all(
        line.startswith("x=") and (", " in line or ".." in line or "BLOCKED" in line)
        for line in lines
    )


# --- 10. live activity feed (Slice B) -------------------------------------------
#
# Events: ("round"|"text"|"persona"|"commit"|"delta", payload) — the UI
# server routes them into its ring.


def test_event_sequence_for_a_scripted_turn() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    agent = _agent(
        [_text("0.05*x")],
        on_event=lambda kind, payload: events.append((kind, dict(payload))),
    )
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    assert [kind for kind, _ in events] == ["round", "text", "commit"]
    assert events[0] == ("round", {"n": 1})
    assert events[1] == ("text", {"text": "0.05*x"})
    assert events[2] == ("commit", {"expr": "0.05*x"})


def test_persona_event_carries_the_verdict() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    agent = _agent(
        [_text("0.05*x")],
        persona="sniper",
        on_event=lambda kind, payload: events.append((kind, dict(payload))),
    )
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    assert [kind for kind, _ in events] == ["round", "text", "persona", "commit"]
    assert events[2] == (
        "persona",
        {"verdict": "PASS", "reason": "", "constraint": "simplest_rung"},
    )


def test_broken_sink_never_crashes_the_turn() -> None:
    def bad_sink(kind: str, payload: dict[str, Any]) -> None:
        raise RuntimeError("sink exploded")

    agent = _agent([_text("0.05*x")], on_event=bad_sink)
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"


def test_set_event_sink_detaches() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    agent = _agent([_text("0.05*x")], on_event=lambda kind, payload: events.append((kind, payload)))
    agent.set_event_sink(None)
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    assert events == []


# --- 13. streaming deltas (Slice B) ---------------------------------------------


class _StreamDelta:
    def __init__(self, type: str, **fields: Any) -> None:  # noqa: A002 - wire field
        self.type = type
        for key, value in fields.items():
            setattr(self, key, value)


class _StreamEvent:
    def __init__(self, type: str, delta: _StreamDelta | None = None) -> None:  # noqa: A002
        self.type = type
        self.delta = delta


class _IterableStream:
    """A fake SDK stream that IS iterable (the real one is) and also closes
    via the context-manager protocol."""

    def __init__(self, events: list[_StreamEvent], response: _Response) -> None:
        self._events = list(events)
        self._response = response

    def __iter__(self) -> Any:
        return iter(self._events)

    def __enter__(self) -> _IterableStream:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def get_final_message(self) -> _Response:
        return self._response


class _IterableStreamingMessages:
    def __init__(self, events: list[_StreamEvent], response: _Response) -> None:
        self._events = events
        self._response = response
        self.stream_kwargs: dict[str, Any] | None = None

    def stream(self, **kwargs: Any) -> _IterableStream:
        self.stream_kwargs = kwargs
        return _IterableStream(self._events, self._response)


class _IterableStreamingClient:
    def __init__(self, events: list[_StreamEvent], response: _Response) -> None:
        self.messages = _IterableStreamingMessages(events, response)


def test_stream_deltas_forward_as_events() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    stream_events = [
        _StreamEvent("message_start"),
        _StreamEvent("content_block_start"),
        _StreamEvent(
            "content_block_delta", _StreamDelta("thinking_delta", thinking="planning the arc")
        ),
        _StreamEvent("content_block_delta", _StreamDelta("text_delta", text="0.05*")),
        _StreamEvent("content_block_delta", _StreamDelta("text_delta", text="x")),
        _StreamEvent("content_block_stop"),
    ]
    response = _Response("end_turn", [_TextBlock("0.05*x")])
    agent = LLMAgent(
        model="fake-model",
        client=_IterableStreamingClient(stream_events, response),
        on_event=lambda kind, payload: events.append((kind, dict(payload))),
    )
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    deltas = [payload for kind, payload in events if kind == "delta"]
    assert deltas == [
        {"kind": "thinking", "text": "planning the arc"},
        {"kind": "text", "text": "0.05*"},
        {"kind": "text", "text": "x"},
    ]
    final_text = [payload for kind, payload in events if kind == "text"]
    assert final_text == [{"text": "0.05*x"}]


def test_stream_delta_thinking_field_shape() -> None:
    """The real SDK's thinking delta carries .thinking (not .text)."""
    events: list[tuple[str, dict[str, Any]]] = []
    stream_events = [
        _StreamEvent("content_block_delta", _StreamDelta("thinking_delta", thinking="pondering")),
    ]
    response = _Response("end_turn", [_TextBlock("0.05*x")])
    agent = LLMAgent(
        model="fake-model",
        client=_IterableStreamingClient(stream_events, response),
        on_event=lambda kind, payload: events.append((kind, dict(payload))),
    )
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    deltas = [payload for kind, payload in events if kind == "delta"]
    assert deltas == [{"kind": "thinking", "text": "pondering"}]


def test_non_iterable_stream_fake_gets_no_deltas() -> None:
    """The existing _FakeStream (get_final_message only) is a documented
    no-op for deltas: no delta events, response still served."""
    events: list[tuple[str, dict[str, Any]]] = []
    agent = LLMAgent(
        model="fake-model",
        client=_StreamingFakeClient(_Response("end_turn", [_TextBlock("0.05*x")])),
        on_event=lambda kind, payload: events.append((kind, dict(payload))),
    )
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    assert [kind for kind, _ in events if kind == "delta"] == []


def test_mid_stream_cancel_raises_turn_cancelled() -> None:
    """The per-delta-batch cancel check (Slice C mid-round granularity): the
    flag flips inside the iteration and the stream raises before the final
    message lands."""
    from agents.llm_agent import TurnCancelled

    flag = {"set": False}

    class _CancellingIterableMessages:
        def __init__(self) -> None:
            self.stream_calls = 0

        def stream(self, **kwargs: Any) -> _IterableStream:
            self.stream_calls += 1

            def event_gen() -> Any:
                yield _StreamEvent("content_block_delta", _StreamDelta("text_delta", text="0.05"))
                flag["set"] = True  # cancel arrives mid-stream
                yield _StreamEvent("content_block_delta", _StreamDelta("text_delta", text="*x"))

            return _IterableStream(list(event_gen()), _Response("end_turn", [_TextBlock("0.05*x")]))

    class _CancellingIterableClient:
        def __init__(self) -> None:
            self.messages = _CancellingIterableMessages()

    agent = LLMAgent(
        model="fake-model",
        client=_CancellingIterableClient(),
        cancel_requested=lambda: flag["set"],
    )
    game, obs = _game_and_obs()
    with pytest.raises(TurnCancelled):
        agent.act(game, obs)


# --- 14. cancellation (Slice C) -------------------------------------------------
#
# TurnCancelled derives from BaseException so act's catch-all `except
# Exception` cannot convert a cancel into a dud shot. The UI server passes a
# module-level threading.Event.is_set as the callback; eval passes nothing.


def test_turn_cancelled_is_a_base_exception() -> None:
    from agents.llm_agent import TurnCancelled

    assert issubclass(TurnCancelled, BaseException)
    assert not issubclass(TurnCancelled, Exception)


def test_cancel_set_before_the_turn_raises_immediately() -> None:
    from agents.llm_agent import TurnCancelled

    agent = _agent([_text("0.05*x")], cancel_requested=lambda: True)
    game, obs = _game_and_obs()
    with pytest.raises(TurnCancelled):
        agent.act(game, obs)
    # Board untouched: no shot fired, no soldier died, no turn advanced.
    fresh = Game.create(5, num_soldiers=1)
    assert game.state.current_turn == fresh.state.current_turn
    assert [s.alive for s in game.all_soldiers()] == [s.alive for s in fresh.all_soldiers()]
    assert agent.stats().simulate_calls == 0
    assert agent._client.messages.calls == []  # no API round-trip at all


def test_cancel_beats_an_api_retry() -> None:
    """A transport death plus a set flag: the between-retries check fires
    before the retry regenerates."""
    from agents.llm_agent import TurnCancelled

    flag = {"set": False}

    class _DyingThenFlagMessages:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> _Response:
            self.calls += 1
            flag["set"] = True
            raise RemoteProtocolError("peer closed connection")

    class _DyingThenFlagClient:
        def __init__(self) -> None:
            self.messages = _DyingThenFlagMessages()

    agent = LLMAgent(
        model="fake-model",
        client=_DyingThenFlagClient(),
        cancel_requested=lambda: flag["set"],
    )
    game, obs = _game_and_obs()
    with pytest.raises(TurnCancelled):
        agent.act(game, obs)
    assert agent._client.messages.calls == 1  # the retry never happened


def test_eval_runs_without_a_cancel_callback_and_cannot_swallow_one() -> None:
    """Eval passes no cancel callback (the default never fires) — and the
    runner's defensive branches cannot swallow a TurnCancelled anyway: it is
    a BaseException and propagates out of play_match."""
    from agents.llm_agent import TurnCancelled

    class _CancellingAgent:
        name = "canceller"

        def act(self, game: Game, obs: Any) -> str:
            raise TurnCancelled()

        def stats(self) -> Any:
            from agents import AgentStats

            return AgentStats()

    with pytest.raises(TurnCancelled):
        play_match(
            21, _CancellingAgent(), StraightShotAgent(), MatchConfig(num_soldiers=2, max_turns=2)
        )
