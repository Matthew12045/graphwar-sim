"""Tests for M5.4 Slice 1: :class:`agents.llm_agent.LLMAgent`.

Zero network: every test injects a fake client that mirrors ONLY the minimal
response surface the agent relies on — ``messages.create(...)`` returning an
object with ``.stop_reason`` and ``.content`` blocks (``.type == "text"`` with
``.text``; ``.type == "tool_use"`` with ``.id``, ``.name``, ``.input``). The
simulate probes run against the REAL budgeted oracle on a REAL seeded game.
"""

from __future__ import annotations

from typing import Any

import pytest

from agents.llm_agent import (
    _BUDGET_EXHAUSTED_TEXT,
    _SYSTEM_PROMPT,
    SAFE_DUD,
    LLMAgent,
)
from agents.observation import observe
from graphwar_sim import Game

# --- fake client (minimal Anthropic surface) ---------------------------------


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    def __init__(self, id: str, expr: str) -> None:  # noqa: A002 - mirrors the wire field
        self.type = "tool_use"
        self.id = id
        self.name = "simulate"
        self.input = {"expr": expr}


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


def _text(text: str) -> _Response:
    return _Response("end_turn", [_TextBlock(text)])


def _tool_use(id: str, expr: str) -> _Response:  # noqa: A002
    return _Response("tool_use", [_ToolUseBlock(id, expr)])


def _agent(responses: list[_Response], **kwargs: Any) -> LLMAgent:
    return LLMAgent(model="fake-model", client=_FakeClient(responses), **kwargs)


def _game_and_obs() -> tuple[Game, Any]:
    game = Game.create(5, num_soldiers=1)
    return game, observe(game)


# --- 1. well-formed first response -------------------------------------------


def test_well_formed_first_response_verbatim_zero_counters() -> None:
    agent = _agent([_text("x^2/4")])
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "x^2/4"
    stats = agent.stats()
    assert stats.parse_failures == 0
    assert stats.retries == 0
    assert stats.simulate_calls == 0
    assert stats.simulate_denied == 0


def test_turn_message_states_live_simulate_budget() -> None:
    agent = _agent([_text("0*x")])
    game, obs = _game_and_obs()
    agent.act(game, obs)
    first_message = agent._client.messages.calls[0]["messages"][0]["content"]
    assert "simulate calls remaining: 3" in first_message  # DEFAULT_SIMULATE_BUDGET
    assert "shooter (you):" in first_message
    assert "enemies (nearest first):" in first_message


# --- 2. malformed then corrected ---------------------------------------------


def test_malformed_then_corrected_counts_and_correction_message() -> None:
    agent = _agent([_text("(("), _text("0.05*x")])
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    stats = agent.stats()
    assert stats.parse_failures == 1
    assert stats.retries == 1
    assert stats.simulate_calls == 0

    # The correction message quoted the reference-style (message-free)
    # exception name.
    second_call_messages = agent._client.messages.calls[1]["messages"]
    correction = second_call_messages[-1]["content"]
    assert "MalformedFunction" in correction


def test_y_equals_prefix_and_fences_are_stripped() -> None:
    agent = _agent([_text("```\ny = 0.05*x\n```")])
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"


# --- 3. exhaustion -> safe dud ------------------------------------------------


def test_all_attempts_malformed_returns_safe_dud() -> None:
    agent = _agent([_text("(("), _text("))")], max_attempts=2)
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == SAFE_DUD
    stats = agent.stats()
    assert stats.parse_failures == 2  # one per attempt
    assert stats.retries == 2
    assert stats.simulate_calls == 0


# --- 4. tool-use round routed through BudgetedSimulator -----------------------


def test_tool_use_round_returns_simresult_fields() -> None:
    agent = _agent([_tool_use("sim-1", "0.05*x"), _text("0.05*x")])
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    stats = agent.stats()
    assert stats.simulate_calls == 1
    assert stats.simulate_denied == 0

    second_call_messages = agent._client.messages.calls[1]["messages"]
    assistant_content = second_call_messages[-2]["content"]
    assert assistant_content[0].type == "tool_use"
    assert assistant_content[0].input == {"expr": "0.05*x"}
    tool_result = second_call_messages[-1]["content"][0]
    assert tool_result["tool_use_id"] == "sim-1"
    assert tool_result.get("is_error") is not True  # absent on success
    import json

    payload = json.loads(tool_result["content"])
    assert set(payload) == {"parseable", "hit_enemy", "hit_teammate", "num_hits", "error"}
    assert payload["parseable"] is True


# --- 5. budget denial: error result, no delegation -----------------------------


def test_budget_denial_returns_error_and_does_not_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agents.simulate_budget
    from agents.simulate_tool import simulate as real_simulate

    delegated: list[str] = []

    def recording_simulate(game: Game, expr: str) -> Any:
        delegated.append(expr)
        return real_simulate(game, expr)

    monkeypatch.setattr(agents.simulate_budget, "simulate", recording_simulate)

    denied_expr = "denied-expr"
    agent = _agent(
        [
            _tool_use("sim-1", "0.05*x"),
            _tool_use("sim-2", "0.1*x"),
            _tool_use("sim-3", "0.15*x"),
            _tool_use("sim-4", denied_expr),
            _text("0.05*x"),
        ]
    )
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    stats = agent.stats()
    assert stats.simulate_calls == 3  # DEFAULT_SIMULATE_BUDGET
    assert stats.simulate_denied == 1
    # The denied call never reached the pure oracle.
    assert delegated == ["0.05*x", "0.1*x", "0.15*x"]

    fifth_call_messages = agent._client.messages.calls[4]["messages"]
    tool_result = fifth_call_messages[-1]["content"][0]
    assert tool_result["tool_use_id"] == "sim-4"
    assert tool_result["is_error"] is True
    assert tool_result["content"] == _BUDGET_EXHAUSTED_TEXT


# --- 6. constructor fails fast without auth env vars ---------------------------


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
