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

from agents import StraightShotAgent
from agents.llm_agent import (
    _BUDGET_EXHAUSTED_TEXT,
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


def _tool_use(id: str, expr: str) -> _Response:  # noqa: A002
    return _Response("tool_use", [_ToolUseBlock(id, expr)])


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
    # No probe ran, so the commit guardrail cannot compare and the commit
    # check is skipped entirely: zero oracle calls, zero overrides.
    assert stats.simulate_calls == 0
    assert stats.simulate_denied == 0
    assert stats.guardrail_overrides == 0


def test_turn_message_states_live_simulate_budget() -> None:
    agent = _agent([_text("0*x")])
    game, obs = _game_and_obs()
    agent.act(game, obs)
    first_message = agent._client.messages.calls[0]["messages"][0]["content"]
    # User-locked live default: unlimited simulate calls.
    assert "simulate calls remaining: unlimited" in first_message
    assert "shooter (you):" in first_message
    assert "enemies (nearest first):" in first_message

    # An explicit int budget formats as the M5.5.6 count instead.
    budgeted = _agent([_text("0*x")], simulate_budget=3)
    budgeted.act(game, obs)
    assert (
        "simulate calls remaining: 3"
        in budgeted._client.messages.calls[0]["messages"][0]["content"]
    )


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


def test_budget_burn_does_not_consume_an_attempt() -> None:
    """A max_tokens stop with no text (hidden-thinking burn) is counted like
    the plan's parse-failure path but the attempt keeps going — the next
    round can still commit."""

    agent = _agent([_Response("max_tokens", []), _text("0.05*x")])
    assert agent.act(*_game_and_obs()) == "0.05*x"  # type: ignore[arg-type]
    stats = agent.stats()
    assert stats.parse_failures == 1  # the burn, not the commit
    assert stats.retries == 1


def test_last_rounds_force_text_only_commit() -> None:
    """The last _COMMIT_ROUNDS rounds set tool_choice "none" so a stochastic
    thinker must emit the bare expression instead of another tool call."""
    agent = _agent([_Response("max_tokens", [])] * 6 + [_text("0.05*x")])
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    calls = agent._client.messages.calls
    assert len(calls) == 7
    assert "tool_choice" not in calls[0]
    assert calls[6]["tool_choice"] == {"type": "none"}
    stats = agent.stats()
    assert stats.parse_failures == 6  # six burns, honest accounting


def test_commit_warning_rides_the_tool_results() -> None:
    """The commit-now signal (the denial text's message) rides the probing
    round that enters the commit window — the model commits on that signal,
    never on its own."""
    agent = _agent(
        [_tool_use("sim-1", "0.05*x"), _tool_use("sim-2", "0.1*x"), _text("0.1*x")],
        tool_rounds=3,
    )
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.1*x"
    calls = agent._client.messages.calls
    r1_user = calls[1]["messages"][-1]["content"]
    assert isinstance(r1_user[0], dict) and r1_user[0]["tool_use_id"] == "sim-1"
    assert "commit your best expression NOW" in r1_user[-1]["text"]  # warning rides r1
    r2_user = calls[2]["messages"][-1]["content"]
    assert "commit your best expression NOW" in r2_user[-1]["text"]


def test_round_cap_falls_back_to_the_last_probed_expression() -> None:
    """When the round cap hits without a text commit, the model's LAST probed
    expression is fired instead of the safe dud (the gateway ignores
    tool_choice "none"; the probing must still pay off)."""
    agent = _agent(
        [_tool_use("sim-1", "0.05*x"), _tool_use("sim-2", "0.1*x")],
        tool_rounds=2,
    )
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.1*x"  # the last probe, not SAFE_DUD
    stats = agent.stats()
    assert stats.simulate_calls == 2
    assert stats.parse_failures == 0

    # Without any probe the safe dud stands.
    burns = _agent([_Response("max_tokens", [])] * 6 + [_text("(("), _text("((")])
    assert burns.act(game, obs) == SAFE_DUD


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


# --- 4. tool-use round routed through BudgetedSimulator -----------------------


def test_unlimited_simulate_budget_never_denies() -> None:
    """The M5.4 user-locked live default: every simulate call delegates, no
    denial stop-signal fires, and the counters still record the calls."""
    agent = _agent([_tool_use(f"sim-{i}", "0.05*x") for i in range(1, 6)] + [_text("0.05*x")])
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    stats = agent.stats()
    # 5 probes + the commit guardrail's check of the final candidate (M5.4:
    # simulate_calls counts ALL oracle calls).
    assert stats.simulate_calls == 6
    assert stats.simulate_denied == 0
    for call in agent._client.messages.calls[1:6]:
        tool_result = call["messages"][-1]["content"][0]
        assert tool_result.get("is_error") is not True


def test_tool_use_round_returns_simresult_fields() -> None:
    agent = _agent([_tool_use("sim-1", "0.05*x"), _text("0.05*x")])
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    stats = agent.stats()
    # 1 probe + the commit guardrail's check (M5.4 semantics).
    assert stats.simulate_calls == 2
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
    assert set(payload) == {
        "parseable",
        "hit_enemy",
        "hit_teammate",
        "num_hits",
        "error",
        "nearest_miss",
        "miss_direction",
        "stopped_at_x",
        "stop_reason",
    }
    assert payload["parseable"] is True
    # Miss telemetry: a real gradient, not a binary coin (live diagnosis).
    assert isinstance(payload["nearest_miss"], float)
    assert payload["miss_direction"] in {"high", "low"}
    assert isinstance(payload["stopped_at_x"], float)
    assert payload["stop_reason"] in {"hit", "terrain", "off_map", "short", "passed"}


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
        ],
        simulate_budget=3,  # the M5.3-style cap; the live default is unlimited
    )
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    stats = agent.stats()
    assert stats.simulate_calls == 3  # the explicit simulate_budget cap
    # 2 denials: the model's 4th probe AND the commit guardrail's check (the
    # budget was already spent — the commit fires as-is, M5.4 degradation).
    assert stats.simulate_denied == 2
    # The denied calls never reached the pure oracle.
    assert delegated == ["0.05*x", "0.1*x", "0.15*x"]

    fifth_call_messages = agent._client.messages.calls[4]["messages"]
    tool_result = fifth_call_messages[-1]["content"][0]
    assert tool_result["tool_use_id"] == "sim-4"
    assert tool_result["is_error"] is True
    assert tool_result["content"] == _BUDGET_EXHAUSTED_TEXT


# --- 6. constructor fails fast without auth env vars ---------------------------


def test_streaming_client_is_preferred_when_available() -> None:
    response = _Response("end_turn", [_TextBlock("0.05*x")])
    agent = LLMAgent(model="fake-model", client=_StreamingFakeClient(response))
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0.05*x"
    kwargs = agent._client.messages.stream_kwargs
    assert kwargs is not None
    assert kwargs["max_tokens"] == 128000  # the model's full context budget
    assert kwargs["system"] == _SYSTEM_PROMPT
    assert kwargs["tools"][0]["name"] == "simulate"


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


# --- 8. miss telemetry (the model's gradient) ---------------------------------


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


# --- 9. commit guardrail (M5.4 Fix 1) ------------------------------------------
#
# The harness oracle-checks the committed expression through the SAME
# BudgetedSimulator and fires the turn's best probed expression instead when
# the commit is STRICTLY worse by _probe_score (ties go to the commit).
# Board facts used below (deterministic seed 21, num_soldiers=2):
# "0*x" dies on the muzzle rock: terrain, nearest_miss 34.725;
# "1(x+18.117)" dies on the same rock, slightly closer: 34.698;
# "-1.4117(x+18.117)" is a line through the muzzle that strikes the ALLY
# (hit_teammate, the critical-failure rank).


def test_probe_score_ordering() -> None:
    """Lower = better: a teammate hit is NEVER preferred over a clean no-hit
    (whatever its nearest_miss), reach ranks over distance, and unparseable
    probes (all-None telemetry) rank worst."""
    from agents.llm_agent import _probe_score
    from agents.simulate_tool import SimResult

    def res(**kwargs: Any) -> SimResult:
        base: dict[str, Any] = {
            "parseable": True,
            "hit_enemy": False,
            "hit_teammate": False,
            "num_hits": 0,
            "num_steps": 10,
        }
        base.update(kwargs)
        return SimResult(**base)

    clean_hit = res(hit_enemy=True, num_hits=1, nearest_miss=0.1, stop_reason="hit")
    teammate_hit = res(hit_teammate=True, num_hits=1, nearest_miss=0.05, stop_reason="hit")
    no_hit_far = res(nearest_miss=100.0, stop_reason="short")
    no_hit_passed = res(nearest_miss=2.0, stop_reason="passed")
    no_hit_short = res(nearest_miss=2.0, stop_reason="short")
    unparseable = res(parseable=False, num_steps=0)

    assert _probe_score(teammate_hit) > _probe_score(no_hit_far)
    assert _probe_score(no_hit_passed) < _probe_score(no_hit_short)
    assert _probe_score(clean_hit) < _probe_score(no_hit_far)
    assert _probe_score(unparseable) == (1, 1, float("inf"))
    assert _probe_score(no_hit_far) < _probe_score(unparseable)


def test_guardrail_fires_a_strictly_better_probe_over_a_dead_commit() -> None:
    """Probes ran (dead on the muzzle rock), the commit strikes a TEAMMATE —
    the guardrail fires the probe instead and counts one override."""
    agent = _agent([_tool_use("sim-1", "0*x"), _text("-1.4117(x+18.117)")])
    game, obs = _seed21_game_and_obs()
    assert agent.act(game, obs) == "0*x"
    stats = agent.stats()
    assert stats.guardrail_overrides == 1
    assert stats.simulate_calls == 2  # probe + commit check
    assert stats.simulate_denied == 0


def test_guardrail_keeps_a_commit_at_least_as_good_as_the_best_probe() -> None:
    """Ties and close calls go to the commit: the model's own word wins when
    its candidate is not strictly worse than the best probe."""
    agent = _agent([_tool_use("sim-1", "0*x"), _text("1(x+18.117)")])
    game, obs = _seed21_game_and_obs()
    assert agent.act(game, obs) == "1(x+18.117)"  # 34.698 beats the probe's 34.725
    stats = agent.stats()
    assert stats.guardrail_overrides == 0
    assert stats.simulate_calls == 2


def test_guardrail_commit_check_denied_on_budget_returns_commit() -> None:
    """The commit check is only deniable on a finite budget: with the turn's
    budget spent, the check raises and the commit fires as-is (documented
    degradation), no crash, no override."""
    agent = _agent([_tool_use("sim-1", "0*x"), _text("1(x+18.117)")], simulate_budget=1)
    game, obs = _seed21_game_and_obs()
    assert agent.act(game, obs) == "1(x+18.117)"
    stats = agent.stats()
    assert stats.guardrail_overrides == 0
    assert stats.simulate_calls == 1  # the probe; the check was denied
    assert stats.simulate_denied == 1  # the denied commit check


def test_unrecoverable_failure_falls_back_to_the_best_probe() -> None:
    """Probes ran, then the gateway killed every regeneration: the turn fires
    the BEST probed expression (the guardrail's preference), not merely the
    last one."""
    agent = LLMAgent(
        model="fake-model",
        client=_DyingAfterClient([_tool_use("sim-1", "0*x"), _tool_use("sim-2", "1(x+18.117)")]),
    )
    game, obs = _seed21_game_and_obs()
    assert agent.act(game, obs) == "1(x+18.117)"  # best (34.698), not 0*x (34.725)
    stats = agent.stats()
    assert stats.parse_failures == 1
    assert stats.retries == 1
    assert agent._client.messages.calls == 5  # 2 rounds + 3 attempts on the death


def test_guardrail_override_flows_through_play_match() -> None:
    """End-to-end: the runner merges the new counter into the match stats
    (mirrors the simulate_calls merge test in test_eval.py)."""
    agent = _agent([_tool_use("sim-1", "0*x"), _text("-1.4117(x+18.117)")])
    result = play_match(21, agent, StraightShotAgent(), MatchConfig(num_soldiers=2, max_turns=2))
    assert result.stats["llm:fake-model"].guardrail_overrides == 1


# --- 10. telemetry echo in the commit nudge (M5.4 Fix 2) -----------------------


class _UnknownToolBlock:
    def __init__(self, id: str) -> None:  # noqa: A002 - mirrors the wire field
        self.type = "tool_use"
        self.id = id
        self.name = "other_tool"
        self.input = {}


def test_commit_warning_echoes_the_last_probe_telemetry() -> None:
    """When the commit nudge fires, it carries the last probe's identity and
    telemetry so the model can commit its best probe or fix exactly its
    failure (the live diagnosis: the commit never re-read old results)."""
    agent = _agent([_tool_use("sim-1", "0*x"), _text("1(x+18.117)")], tool_rounds=3)
    game, obs = _seed21_game_and_obs()
    assert agent.act(game, obs) == "1(x+18.117)"
    warning = agent._client.messages.calls[1]["messages"][-1]["content"][-1]["text"]
    assert warning.startswith("That was your last probe of the turn")
    assert "Your last probe '0*x'" in warning
    assert "nearest_miss=34.725" in warning
    assert "miss_direction=low" in warning
    assert "stop_reason=terrain" in warning
    assert "commit THAT expression" in warning


def test_commit_warning_has_no_echo_without_a_probe() -> None:
    """No oracle-reaching probe this turn -> the plain nudge only (the echo
    cannot invent telemetry)."""
    agent = _agent([_Response("tool_use", [_UnknownToolBlock("u-1")]), _text("0*x")], tool_rounds=2)
    game, obs = _game_and_obs()
    assert agent.act(game, obs) == "0*x"
    warning = agent._client.messages.calls[1]["messages"][-1]["content"][-1]["text"]
    assert "commit your best expression NOW" in warning
    assert "Your last probe" not in warning


# --- 11. muzzle-wall warning in the turn message (M5.4 Fix 3b) -----------------


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

    obs = _obs_with_terrain(((-17.2, 7.8), (5.0, -5.0)))
    message = _turn_message(obs, "unlimited")
    assert "terrain wall at (x~-17.2, y~7.8) just right of your muzzle" in message
    assert "launch DESCENDING" in message


def test_turn_message_no_warning_for_far_or_left_terrain() -> None:
    """Terrain left of the muzzle, or beyond the scan radii, warns nothing."""
    from agents.llm_agent import _turn_message

    for blocks in (
        ((-20.0, 7.5), (5.0, -5.0)),  # left of the muzzle
        ((-14.0, 7.5), (5.0, -5.0)),  # >2.5 right
        ((-17.5, 12.0), (5.0, -5.0)),  # >2 above
    ):
        message = _turn_message(_obs_with_terrain(blocks), "unlimited")
        assert "terrain wall" not in message, blocks


def test_seed21_turn_message_carries_the_muzzle_warning() -> None:
    """The live-diagnosis board triggers the warning (integration of the
    construction with the real observation)."""
    from agents.llm_agent import _turn_message

    _, obs = _seed21_game_and_obs()
    message = _turn_message(obs, "unlimited")
    assert "just right of your muzzle" in message
