"""OpenAI-compatible backend for :class:`~agents.llm_agent.LLMAgent`.

Lets the LLM agents talk to any server that speaks the OpenAI Chat
Completions API — a local vLLM (``vllm serve Qwen/Qwen3.8-27B``), Ollama,
LM Studio, llama.cpp's server, or a hosted provider (OpenRouter, DeepInfra,
Together, ...). The agents were written against the Anthropic Messages
surface, so this module is a thin ADAPTER, not a second agent: it exposes
exactly the duck-typed surface ``LLMAgent`` relies on —
``client.messages.create(**kwargs)`` / ``client.messages.stream(**kwargs)``
returning a message with ``.stop_reason`` and ``.content`` text blocks, and
a stream that yields ``content_block_delta`` events (``text_delta`` /
``thinking_delta``) — and translates both ways. ``LLMAgent`` and
``HybridAgent`` are otherwise untouched.

Configuration (environment; read by :func:`build_openai_client`):

- ``OPENAI_BASE_URL`` — the server's ``/v1`` root, e.g.
  ``http://localhost:8001/v1`` (vLLM), ``http://localhost:11434/v1``
  (Ollama), ``http://localhost:1234/v1`` (LM Studio).
- ``OPENAI_API_KEY`` — optional for local servers (defaults to ``"EMPTY"``,
  the vLLM convention).
- ``GRAPHWAR_LLM_MAX_TOKENS`` — output cap per call (default
  :data:`DEFAULT_MAX_TOKENS`); local servers reject budgets beyond their
  context window, so the agent's 128k Anthropic-gateway budget is clamped
  to this.

Thinking models: Qwen3.x reasons before answering. vLLM's ``--reasoning-parser
qwen3`` returns that reasoning in a separate ``reasoning_content`` (newer
builds: ``reasoning``) field; servers without a parser leave it inline as
``<think>...</think>``. Either way the reasoning is surfaced as THINKING
(the UI's thought bubble) and never as answer text, so the agent's
"last line is the expression" extraction only ever sees the answer.

The ``openai`` SDK is an optional dependency (``pip install
'graphwar-sim[llm]'``), imported lazily inside :func:`build_openai_client`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

# Output cap per call for OpenAI-compatible servers. Qwen3.8-27B thinks for a
# few thousand tokens at medium effort; 32k leaves headroom while fitting
# the default context of common local servers. # TUNABLE — not from source.
DEFAULT_MAX_TOKENS: int = 32768

# Placeholder key for local servers that ignore auth (vLLM's convention).
_NO_KEY: str = "EMPTY"

_THINK_BLOCK: re.Pattern[str] = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL)

# OpenAI finish_reason -> Anthropic stop_reason (the agent only reads it
# for diagnostics; unknown values pass through).
_STOP_REASONS: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ThinkingBlock:
    thinking: str
    type: str = "thinking"


@dataclass
class Message:
    """The Anthropic-shaped response the agents read."""

    content: list[Any]
    stop_reason: str | None
    model: str = ""


@dataclass
class _Delta:
    type: str
    text: str = ""
    thinking: str = ""


@dataclass
class _DeltaEvent:
    delta: _Delta
    type: str = "content_block_delta"


def split_thinking(content: str) -> tuple[str, str]:
    """Split inline ``<think>...</think>`` reasoning out of ``content``.

    Returns ``(thinking, answer)``. An unterminated ``<think>`` (the output
    was cut mid-thought) counts entirely as thinking."""
    thoughts = [m.group(1).strip() for m in _THINK_BLOCK.finditer(content)]
    answer = _THINK_BLOCK.sub("", content).replace("</think>", "").strip()
    return "\n".join(t for t in thoughts if t), answer


def _reasoning_of(obj: Any) -> str:
    """The separate reasoning field of a message/delta, if the server sends
    one (vLLM: ``reasoning_content``; newer vLLM / OpenRouter:
    ``reasoning``). Read through ``model_extra`` too — the SDK keeps unknown
    fields there."""
    for name in ("reasoning_content", "reasoning"):
        value = getattr(obj, name, None)
        if value is None:
            extra = getattr(obj, "model_extra", None) or {}
            value = extra.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


def _to_openai_messages(system: str | None, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic ``system`` + messages -> OpenAI chat messages. Content may
    be a plain string or a list of Anthropic text blocks (joined)."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(block.get("text", ""))
                if isinstance(block, dict)
                else str(getattr(block, "text", ""))
                for block in content
            )
        out.append({"role": msg["role"], "content": content})
    return out


def _message_from(thinking: str, content: str, finish_reason: str | None, model: str) -> Message:
    inline_thinking, answer = split_thinking(content)
    thinking = "\n".join(t for t in (thinking, inline_thinking) if t)
    blocks: list[Any] = []
    if thinking:
        blocks.append(ThinkingBlock(thinking))
    if answer:
        blocks.append(TextBlock(answer))
    stop = _STOP_REASONS.get(finish_reason or "", finish_reason)
    return Message(content=blocks, stop_reason=stop, model=model)


@dataclass
class _Stream:
    """Context manager mirroring the Anthropic SDK's ``MessageStream``:
    iterate for ``content_block_delta`` events, then ``get_final_message``.
    Inline ``<think>`` text is routed to thinking deltas as it streams."""

    _chunks: Any
    _model: str
    _thinking: list[str] = field(default_factory=list)
    _content: list[str] = field(default_factory=list)
    _finish_reason: str | None = None
    _in_think: bool = False
    _drained: bool = False

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *exc: Any) -> None:
        close = getattr(self._chunks, "close", None)
        if callable(close):
            close()

    def __iter__(self) -> Iterator[_DeltaEvent]:
        for chunk in self._chunks:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            if getattr(choice, "finish_reason", None):
                self._finish_reason = choice.finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            reasoning = _reasoning_of(delta)
            if reasoning:
                self._thinking.append(reasoning)
                yield _DeltaEvent(_Delta("thinking_delta", thinking=reasoning))
            text = getattr(delta, "content", None) or ""
            if text:
                self._content.append(text)
                yield from self._route_inline(text)
        self._drained = True

    def _route_inline(self, text: str) -> Iterator[_DeltaEvent]:
        """Best-effort live routing of inline ``<think>`` tags (the final
        message re-splits the full text, so a tag split across chunks only
        affects the live feed, never the answer)."""
        while text:
            tag = "</think>" if self._in_think else "<think>"
            head, sep, text = text.partition(tag)
            if head:
                kind = "thinking_delta" if self._in_think else "text_delta"
                yield _DeltaEvent(
                    _Delta(kind, thinking=head) if self._in_think else _Delta(kind, text=head)
                )
            if sep:
                self._in_think = not self._in_think

    def get_final_message(self) -> Message:
        if not self._drained:
            for _ in self:
                pass
        return _message_from(
            "".join(self._thinking), "".join(self._content), self._finish_reason, self._model
        )


class _Messages:
    def __init__(self, sdk: Any, max_tokens: int) -> None:
        self._sdk = sdk
        self._max_tokens = max_tokens

    def _params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": kwargs["model"],
            "messages": _to_openai_messages(kwargs.get("system"), kwargs["messages"]),
            "max_tokens": min(int(kwargs.get("max_tokens", self._max_tokens)), self._max_tokens),
        }
        # reasoning_effort rides extra_body exactly as on the gateway path;
        # vLLM, Ollama and OpenRouter read it as a top-level field.
        extra = dict(kwargs.get("extra_body") or {})
        if extra:
            params["extra_body"] = extra
        return params

    def create(self, **kwargs: Any) -> Message:
        params = self._params(kwargs)
        resp = self._sdk.chat.completions.create(**params)
        choice = resp.choices[0]
        msg = choice.message
        return _message_from(
            _reasoning_of(msg), msg.content or "", choice.finish_reason, params["model"]
        )

    def stream(self, **kwargs: Any) -> _Stream:
        params = self._params(kwargs)
        chunks = self._sdk.chat.completions.create(**params, stream=True)
        return _Stream(chunks, params["model"])


class OpenAICompatClient:
    """Anthropic-shaped facade over an ``openai.OpenAI`` client."""

    def __init__(self, sdk: Any, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self.messages = _Messages(sdk, max_tokens)


def openai_backend_selected() -> bool:
    """True when the environment points the agents at an OpenAI-compatible
    server: ``GRAPHWAR_LLM_PROVIDER=openai``, or ``OPENAI_BASE_URL`` set
    with no Anthropic credentials (an explicit Anthropic setup wins)."""
    provider = os.environ.get("GRAPHWAR_LLM_PROVIDER", "").strip().lower()
    if provider:
        return provider == "openai"
    has_anthropic = bool(
        os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    )
    return bool(os.environ.get("OPENAI_BASE_URL")) and not has_anthropic


def build_openai_client() -> OpenAICompatClient:
    """Construct the adapter from the environment (see the module doc).

    Raises BEFORE any network call when no base URL is configured — a
    misconfigured agent must fail at construction, never mid-match."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "the OpenAI-compatible backend needs OPENAI_BASE_URL (e.g. "
            "http://localhost:8001/v1 for vLLM, http://localhost:11434/v1 for Ollama)"
        )
    import openai  # optional dependency — imported only when a client is built

    max_tokens = int(os.environ.get("GRAPHWAR_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    sdk = openai.OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY") or _NO_KEY)
    return OpenAICompatClient(sdk, max_tokens=max_tokens)


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "OpenAICompatClient",
    "build_openai_client",
    "openai_backend_selected",
    "split_thinking",
]
