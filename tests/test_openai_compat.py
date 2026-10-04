"""Tests for :mod:`agents.openai_compat` — the OpenAI-compatible backend
(local vLLM / Ollama / LM Studio serving Qwen, or a hosted provider).

Zero network: a fake ``openai.OpenAI`` exposing only
``chat.completions.create`` (plain and ``stream=True``) is injected under
the adapter, and the adapter is injected into the real agents.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agents.hybrid_agent import HybridAgent
from agents.llm_agent import LLMAgent, _build_client
from agents.observation import observe
from agents.openai_compat import (
    OpenAICompatClient,
    openai_backend_selected,
    split_thinking,
)
from graphwar_sim import Game

_ENV = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "GRAPHWAR_LLM_PROVIDER",
    "GRAPHWAR_REASONING_EFFORT",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def _completion(content: str, reasoning: str | None = None, finish: str = "stop") -> Any:
    msg = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)])


def _chunk(
    content: str | None = None, reasoning: str | None = None, finish: str | None = None
) -> Any:
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


class _FakeCompletions:
    def __init__(self, completion: Any = None, chunks: list[Any] | None = None) -> None:
        self._completion = completion
        self._chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            assert self._chunks is not None
            return iter(self._chunks)
        return self._completion


def _sdk(completions: _FakeCompletions) -> Any:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class _CreateOnly:
    """Hides ``stream`` so the agent takes the non-streaming path."""

    def __init__(self, inner: OpenAICompatClient) -> None:
        self.messages = SimpleNamespace(create=inner.messages.create)


def _game_and_obs() -> tuple[Game, Any]:
    game = Game.create(5, num_soldiers=1)
    return game, observe(game)


def test_split_thinking_inline_tags() -> None:
    assert split_thinking("<think>aim low</think>\nx/4") == ("aim low", "x/4")
    assert split_thinking("x/4") == ("", "x/4")
    # Cut mid-thought: everything is thinking, no answer.
    assert split_thinking("<think>still going") == ("still going", "")


def test_create_translates_request_and_response() -> None:
    completions = _FakeCompletions(_completion("x/4", reasoning="the lane is open"))
    client = OpenAICompatClient(_sdk(completions), max_tokens=1000)
    msg = client.messages.create(
        model="Qwen/Qwen3.8-27B",
        max_tokens=128000,
        system="SYS",
        messages=[{"role": "user", "content": "TURN"}],
        extra_body={"reasoning_effort": "medium"},
    )
    call = completions.calls[0]
    assert call["model"] == "Qwen/Qwen3.8-27B"
    assert call["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "TURN"},
    ]
    assert call["max_tokens"] == 1000  # clamped to the backend cap
    assert call["extra_body"] == {"reasoning_effort": "medium"}
    assert msg.stop_reason == "end_turn"
    assert [b.type for b in msg.content] == ["thinking", "text"]
    assert msg.content[1].text == "x/4"


def test_llm_agent_fires_answer_not_reasoning_non_streaming() -> None:
    completions = _FakeCompletions(_completion("<think>y=0 hits the wall\nso 0*x</think>\nx^2/40"))
    agent = LLMAgent(
        model="Qwen/Qwen3.8-27B", client=_CreateOnly(OpenAICompatClient(_sdk(completions)))
    )
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "x^2/40"
    assert agent.stats().parse_failures == 0


def test_llm_agent_streams_thinking_deltas() -> None:
    chunks = [
        _chunk(reasoning="reading the lanes"),
        _chunk(content="<thi"),  # tag split across chunks: answer must still be clean
        _chunk(content="nk>more</think>"),
        _chunk(content="x/5"),
        _chunk(finish="stop"),
    ]
    completions = _FakeCompletions(chunks=chunks)
    events: list[tuple[str, dict[str, Any]]] = []
    agent = LLMAgent(
        model="Qwen/Qwen3.8-27B",
        client=OpenAICompatClient(_sdk(completions)),
        on_event=lambda kind, payload: events.append((kind, payload)),
    )
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "x/5"
    assert completions.calls[0]["stream"] is True
    thinking = [p["text"] for k, p in events if k == "delta" and p["kind"] == "thinking"]
    assert "reading the lanes" in thinking


def test_hybrid_agent_runs_over_the_adapter() -> None:
    completions = _FakeCompletions(_completion("not json at all"))
    agent = HybridAgent(model="Qwen/Qwen3.8-27B", client=OpenAICompatClient(_sdk(completions)))
    game, obs = _game_and_obs()
    expr = agent.act(game, obs)  # a bad plan degrades, never crashes
    assert isinstance(expr, str) and expr
    assert completions.calls


def test_backend_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not openai_backend_selected()
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8001/v1")
    assert openai_backend_selected()
    # Explicit Anthropic credentials win unless the provider is forced.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert not openai_backend_selected()
    monkeypatch.setenv("GRAPHWAR_LLM_PROVIDER", "openai")
    assert openai_backend_selected()


def test_build_client_uses_adapter_and_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8001/v1")
    assert isinstance(_build_client(), OpenAICompatClient)
    agent = LLMAgent(model="Qwen/Qwen3.8-27B")
    assert agent._reasoning_effort == "medium"
    monkeypatch.setenv("GRAPHWAR_REASONING_EFFORT", "off")
    assert LLMAgent(model="Qwen/Qwen3.8-27B")._reasoning_effort is None


def test_forced_provider_without_url_fails_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPHWAR_LLM_PROVIDER", "openai")
    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL"):
        LLMAgent(model="Qwen/Qwen3.8-27B")
