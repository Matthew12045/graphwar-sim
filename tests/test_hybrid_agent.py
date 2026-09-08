"""Tests for M5.5 Slice D3: the HybridAgent.

- the isolation guarantee (M5.5.1/M5.5.8): the LLM's text NEVER reaches the
  parser and is NEVER emitted — the fired expression is the CCF solver's;
- the schema loop (M5.5.3): corrections are cheap, ``schema_errors`` is its
  own counter;
- the relaxation ladder (M5.5.5): lowest-priority dropped first, ties by the
  most-binding (tightest tol); ``UNREACHABLE`` stops the ladder with an
  explicit report;
- the M5.5.6 feedback: specific u/sigma/binding + remaining budget, NEVER
  the expression;
- the corridor summary (M5.5.7): teammate bands pre-applied;
- the roster grammar + stats.

Zero network: the same fake-client pattern as tests/test_llm_agent.py; the
CCF solver runs for real on seeded boards.
"""

from __future__ import annotations

from typing import Any

import pytest

from agents.hybrid_agent import HybridAgent, _extract_json_object
from agents.llm_agent import SAFE_DUD
from agents.observation import observe
from graphwar_sim import Game
from graphwar_sim.ccf import (
    CCFCandidate,
    CCFCertificate,
    CCFOutcome,
    CCFSolution,
    solve_target,
)

# seed 1, 2 soldiers: bare CCF for target 0 is CERTIFIED; seed 4 is
# BASIS_INFEASIBLE at zero waypoints (probed and pinned).


def _game_and_obs(seed: int = 1) -> tuple[Game, Any]:
    game = Game.create(seed, num_soldiers=2)
    return game, observe(game)


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Text:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    def __init__(self, tool_id: str, name: str, arguments: dict[str, Any]) -> None:
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        self.input = arguments


class _Response:
    def __init__(self, stop_reason: str, content: list) -> None:
        self.stop_reason = stop_reason
        self.content = content


class _ScriptedMessages:
    def __init__(self, responses: list[_Response]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(dict(kwargs))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class _ScriptedClient:
    def __init__(self, responses: list[_Response]) -> None:
        self.messages = _ScriptedMessages(responses)


def _plan_text(plan: str) -> _Response:
    return _Response("end_turn", [_TextBlock(plan)])


def _hybrid_agent(responses: list[_Response], **kwargs: Any) -> HybridAgent:
    return HybridAgent(model="fake-model", client=_ScriptedClient(responses), **kwargs)


EMPTY_PLAN = '{"target_id": "enemy_0", "waypoints": [], "rationale": "let the solver decide"}'


# --- identity / grammar -------------------------------------------------------


def test_name_grammar_and_unknown_persona() -> None:
    agent = _hybrid_agent([_plan_text(EMPTY_PLAN)])
    assert agent.name == "hybrid:fake-model"
    with pytest.raises(ValueError, match="unknown persona"):
        HybridAgent(
            model="fake-model", client=_ScriptedClient([_plan_text(EMPTY_PLAN)]), persona="gremlin"
        )


def test_persona_rides_the_name() -> None:
    agent = _hybrid_agent([_plan_text(EMPTY_PLAN)], persona="howitzer")
    assert agent.name == "hybrid:fake-model@howitzer"


def test_system_prompt_is_the_hybrid_prompt() -> None:
    from agents.hybrid_agent import _hybrid_system_prompt

    agent = _hybrid_agent([_plan_text(EMPTY_PLAN)])
    assert agent._system_prompt == _hybrid_system_prompt()
    assert "You do NOT write mathematical expressions" in agent._system_prompt


# --- happy path ----------------------------------------------------------------


def test_empty_waypoint_plan_fires_the_bare_ccf_expression() -> None:
    game, obs = _game_and_obs()
    agent = _hybrid_agent([_plan_text(EMPTY_PLAN)])
    expr = agent.act(game, obs)
    # seed 1: the bare solve CERTIFIES — the fired shot is the CCF
    # expression, not the safe dud and not the LLM's text.
    from graphwar_sim.corridor import shooter_frame

    fr = shooter_frame(game)
    bare = solve_target(fr.mx, fr.my, fr.targets, fr.teammates, fr.circles, fr.inverted, 0)
    assert expr == bare.candidates[0].expression
    stats = agent.stats()
    assert stats.ccf_certified == 1
    assert stats.ccf_infeasible == 0
    assert stats.waypoints_applied == 0
    assert stats.waypoints_dropped == 0
    assert stats.schema_errors == 0
    assert stats.simulate_calls == 0


def test_feedback_never_contains_the_expression() -> None:
    game, obs = _game_and_obs()
    agent = _hybrid_agent([_plan_text(EMPTY_PLAN)])
    agent.act(game, obs)
    feedback = agent._last_feedback
    assert feedback is not None
    from graphwar_sim.corridor import shooter_frame

    fr = shooter_frame(game)
    bare = solve_target(fr.mx, fr.my, fr.targets, fr.teammates, fr.circles, fr.inverted, 0)
    expression = bare.candidates[0].expression
    assert expression not in feedback  # M5.5.6: NEVER the expression
    assert "CERTIFIED" in feedback
    assert "binding:" in feedback
    assert "sigma=" in feedback
    assert "simulate_tool calls remaining" in feedback


def test_feedback_rides_the_next_turn_message() -> None:
    game, obs = _game_and_obs()
    client = _ScriptedClient([_plan_text(EMPTY_PLAN), _plan_text(EMPTY_PLAN)])
    agent = HybridAgent(model="fake-model", client=client)
    agent.act(game, obs)
    # Second turn: the previous feedback must appear in the new turn message.
    agent.act(game, obs)
    first_user_message = agent._client.messages.calls[-1]["messages"][0]["content"]
    assert "previous attempt:" in first_user_message
    assert "ATTEMPT" in first_user_message


# --- schema loop (M5.5.3) --------------------------------------------------------


def test_prose_then_valid_plan_corrects_cheaply() -> None:
    game, obs = _game_and_obs()
    agent = _hybrid_agent([_Response("end_turn", [_Text("no plan here")]), _plan_text(EMPTY_PLAN)])
    expr = agent.act(game, obs)
    assert expr != SAFE_DUD  # the second round's plan was accepted
    stats = agent.stats()
    assert stats.schema_errors == 1  # its OWN counter
    assert stats.parse_failures == 0  # never mixed into solver failures
    assert stats.retries == 0
    # The correction message reached the model.
    last_messages = agent._client.messages.calls[-1]["messages"]
    assert any("no JSON plan found" in m["content"] for m in last_messages)


def test_schema_invalid_plan_is_corrected_without_a_solver_attempt() -> None:
    bad = '{"target_id": "enemy_9", "waypoints": []}'
    game, obs = _game_and_obs()
    agent = _hybrid_agent([_plan_text(bad), _plan_text(EMPTY_PLAN)])
    expr = agent.act(game, obs)
    assert expr != SAFE_DUD
    stats = agent.stats()
    assert stats.schema_errors == 1
    last_messages = agent._client.messages.calls[-1]["messages"]
    assert any("dead target_id" in m["content"] for m in last_messages[1:])


def test_round_cap_without_a_plan_fires_the_safe_dud() -> None:
    game, obs = _game_and_obs()
    agent = _hybrid_agent(
        [_Response("end_turn", [_Text("no json")]) for _ in range(8)], tool_rounds=3
    )
    expr = agent.act(game, obs)
    assert expr == SAFE_DUD
    assert agent.stats().schema_errors == 3


# --- isolation (M5.5.8) -----------------------------------------------------------


def test_expression_in_rationale_is_never_emitted() -> None:
    """The hostile model hides a valid expression in 'rationale': the fired
    shot is still the CCF solver's expression."""
    game, obs = _game_and_obs()
    hostile_plan = (
        '{"target_id": "enemy_0", "waypoints": [], "rationale": "just fire 0.5*x^2"'
        ', "branch_hint": "any"}'
    )
    from graphwar_sim.corridor import shooter_frame

    fr = shooter_frame(game)
    bare = solve_target(fr.mx, fr.my, fr.targets, fr.teammates, fr.circles, fr.inverted, 0)
    expected = bare.candidates[0].expression
    agent = _hybrid_agent([_plan_text(hostile_plan)])
    expr = agent.act(game, obs)
    assert expr == expected
    assert expr != "0.5*x^2"
    assert "0.5*x^2" not in expr


# --- relaxation ladder (M5.5.5, mechanism via a scripted solver) -------------------


def _cert(outcome: CCFOutcome) -> CCFCertificate:
    return CCFCertificate(outcome=outcome, target_index=0)


def _cand(expression: str, certified: bool) -> CCFCandidate:
    return CCFCandidate(
        expression=expression,
        target_index=0,
        sigma=2.0,
        branch_index=0,
        branch_kind="over",
        certified=certified,
        slack_total=0.0,
        m_bound=1.0,
        mode="allowance",
        nonzero_weights=1,
    )


def _sol(outcome: CCFOutcome, expression: str | None) -> CCFSolution:
    candidates = (
        []
        if outcome not in (CCFOutcome.CERTIFIED, CCFOutcome.UNCERTIFIED)
        else [_cand(expression, outcome is CCFOutcome.CERTIFIED)]
    )
    return CCFSolution(0, candidates, CCFCertificate(outcome=outcome, target_index=0))


def test_ladder_drops_the_lowest_priority_waypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """First solve (both waypoints) infeasible -> the LOWEST-priority one is
    dropped and the re-solve certifies."""
    calls: list[int] = []

    def fake_solve_target(*args: Any, waypoints: Any = (), **kwargs: Any) -> CCFSolution:
        calls.append(len(waypoints))
        if len(waypoints) == 2:
            return _sol(CCFOutcome.BASIS_INFEASIBLE, "")
        return _sol(CCFOutcome.CERTIFIED, "CERT_EXPR")

    monkeypatch.setattr("agents.hybrid_agent.solve_target", fake_solve_target)
    plan_text = (
        '{"target_id": "enemy_0", "waypoints": ['
        '{"u": 4.0, "y": 0.0, "tol": 2.0, "priority": 1},'
        '{"u": 9.0, "y": 0.0, "tol": 2.0, "priority": 9}], "rationale": ""}'
    )
    agent = _hybrid_agent([_plan_text(plan_text)])
    game, obs = _game_and_obs()
    expr = agent.act(game, obs)
    assert expr == "CERT_EXPR"
    assert calls == [2, 1]  # full set first, then the lowest priority dropped
    stats = agent.stats()
    assert stats.waypoints_dropped == 1
    assert stats.waypoints_applied == 1
    assert stats.ccf_certified == 1
    feedback = agent._last_feedback or ""
    assert "dropped" in feedback and "u=4.00" in feedback
    assert "CERT_EXPR" not in feedback  # the feedback never quotes the shot


def test_ladder_tie_break_drops_the_tightest_waypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two priority-1 waypoints with different tol: the most-binding
    (tightest tol) is dropped first (the documented approximation)."""
    dropped_order: list[float] = []

    def fake_solve_target(*args: Any, waypoints: Any = (), **kwargs: Any) -> CCFSolution:
        if len(waypoints) >= 2:
            return _sol(CCFOutcome.BASIS_INFEASIBLE, "")
        dropped_order.extend(round(w.u, 1) for w in waypoints)
        return _sol(CCFOutcome.CERTIFIED, "CERT_EXPR")

    monkeypatch.setattr("agents.hybrid_agent.solve_target", fake_solve_target)
    plan_text = (
        '{"target_id": "enemy_0", "waypoints": ['
        '{"u": 4.0, "y": 0.0, "tol": 3.0, "priority": 1},'
        '{"u": 8.0, "y": 0.0, "tol": 0.5, "priority": 1}], "rationale": ""}'
    )
    agent = _hybrid_agent([_plan_text(plan_text)])
    game, obs = _game_and_obs()
    expr = agent.act(game, obs)
    assert expr == "CERT_EXPR"
    stats = agent.stats()
    assert stats.waypoints_dropped == 1
    feedback = agent._last_feedback or ""
    assert "u=8.00" in feedback  # the TIGHTER one (tol 0.5) went first


def test_unreachable_stops_the_ladder_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """UNREACHABLE is never the waypoints' fault: no waypoint is dropped and
    the report says so explicitly (M5.5.5). The middle rung still fires the
    M2 best-effort shot instead of the dud."""

    def fake_solve_target(*args: Any, waypoints: Any = (), **kwargs: Any) -> CCFSolution:
        return _sol(CCFOutcome.UNREACHABLE, "")

    monkeypatch.setattr("agents.hybrid_agent.solve_target", fake_solve_target)
    plan_text = (
        '{"target_id": "enemy_0", "waypoints": ['
        '{"u": 4.0, "y": 0.0, "tol": 2.0, "priority": 5}], "rationale": ""}'
    )
    agent = _hybrid_agent([_plan_text(plan_text)])
    game, obs = _game_and_obs()
    expr = agent.act(game, obs)
    stats = agent.stats()
    assert stats.waypoints_dropped == 0
    assert stats.ccf_unreachable == 1
    assert stats.m2_fallbacks == 1
    from agents.simulate_tool import simulate

    assert simulate(game, expr).parseable
    feedback = agent._last_feedback or ""
    assert "UNREACHABLE" in feedback
    assert "Your plan was NOT the problem" in feedback


def test_zero_waypoint_infeasible_reported_explicitly() -> None:
    """seed 4: the bare solve is infeasible with ZERO waypoints — the plan
    was never the problem (M5.5.5). The report still says so, but the turn
    fires the M2 best-effort rung (``arc`` here) instead of the dud."""
    game, obs = _game_and_obs(4)
    agent = _hybrid_agent([_plan_text(EMPTY_PLAN)])
    expr = agent.act(game, obs)
    assert expr != SAFE_DUD
    stats = agent.stats()
    assert stats.ccf_infeasible == 1
    assert stats.m2_fallbacks == 1
    feedback = agent._last_feedback or ""
    assert "BASIS_INFEASIBLE" in feedback
    assert "Your plan was NOT the problem" in feedback


def test_magician_miss_board_fires_m2_double_kill() -> None:
    """Regression for the live-demo Master Magician miss (seed 1605663942):
    after the solver's opening turn the corridor to the tight enemy cluster
    admits ZERO CCF branches — the plan was never the problem — so the turn
    must fire the M2 best-effort rung (a simulated double kill), not the dud.
    """
    from agents.simulate_tool import simulate
    from agents.solver_agent import SolverAgent

    game = Game.create(1605663942, num_soldiers=2)
    opener = SolverAgent()
    game.play_turn(opener.act(game, observe(game)))
    obs = observe(game)
    magician_plan = (
        '{"target_id": "enemy_0", "secondary_targets": ["enemy_1"], '
        '"branch_hint": "over", "waypoints": ['
        '{"u": 15.0, "y": 2.5, "tol": 3.0, "priority": 1}, '
        '{"u": 25.0, "y": 4.0, "tol": 3.0, "priority": 2}], '
        '"style": "master_magician", "rationale": "tight cluster, over-arc"}'
    )
    agent = _hybrid_agent([_plan_text(magician_plan)], persona="master_magician")
    expr = agent.act(game, obs)
    assert expr != SAFE_DUD
    stats = agent.stats()
    assert stats.ccf_infeasible == 1
    assert stats.waypoints_dropped == 2
    assert stats.m2_fallbacks == 1
    sim = simulate(game, expr)
    assert sim.hit_enemy and not sim.hit_teammate
    assert sim.num_hits == 2


# --- corridor summary (M5.5.7: bands pre-applied) ------------------------------------


def test_corridor_summary_removes_teammate_bands() -> None:
    from agents.hybrid_agent import _corridor_summary

    # Shooter at (0, 0); one teammate directly at u=5 (band removed), no
    # terrain; the corridor at u=0 must be SPLIT by the teammate's band.
    lines, _band = _corridor_summary(
        mx=0.0,
        my=0.0,
        targets=((20.0, 0.0),),
        teammates=((0.0, 0.0),),
        circles=(),
        inverted=False,
    )
    first = lines[0]
    assert "BLOCKED" not in first
    # The band around y=0 is gone: no interval may contain 0.0.
    assert not _interval_contains_zero(first)


def _interval_contains_zero(line: str) -> bool:
    for token in line.split(":")[1].split("]"):
        token = token.strip("[] ")
        if not token:
            continue
        lo, hi = (float(v) for v in token.replace("..", " ").split())
        if lo < 0.0 < hi:
            return True
    return False


# --- events (Slice B surface, hybrid kinds) -----------------------------------------


def test_event_sequence_for_a_scripted_plan_turn() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    agent = _hybrid_agent(
        [_plan_text(EMPTY_PLAN)],
        on_event=lambda kind, payload: events.append((kind, dict(payload))),
    )
    game, obs = _game_and_obs()
    agent.act(game, obs)
    assert [kind for kind, _ in events] == ["round", "text", "plan", "solve", "commit"]
    assert events[2][1] == {"target": "enemy_0", "n_waypoints": 0, "branch_hint": "any"}
    assert events[3][1]["outcome"] == "CERTIFIED"
    assert events[3][1]["applied"] == 0
    commit = events[-1][1]
    assert commit["expr"].startswith("((")  # the CCF expression, not the plan


# --- persona overlays (M5.5.7) --------------------------------------------------------


def test_master_magician_overlay_covers_secondary_targets() -> None:
    from agents.hybrid_agent import _PERSONA_OVERLAYS

    overlay = _PERSONA_OVERLAYS["master_magician"]
    assert "secondary_targets" in overlay
    assert "cluster" in overlay.lower()  # tooth merging is a PROMPT method


def test_hybrid_persona_does_not_attach_the_m5_4_verifier() -> None:
    """The hybrid's persona is a shape prior (M5.5.7), not the M5.4
    constraint+verifier game: no rung_history, no constraint counters."""
    agent = _hybrid_agent([_plan_text(EMPTY_PLAN)], persona="sniper")
    assert not hasattr(agent, "rung_history")
    stats = agent.stats()
    assert stats.constraint_checks == 0


# --- stats merge through play_match -----------------------------------------------------


class _HybridStubAgent:
    name = "stub"

    def __init__(self) -> None:
        from agents import AgentStats

        self._stats = AgentStats(
            schema_errors=2, waypoints_applied=3, waypoints_dropped=1, ccf_certified=1
        )

    def act(self, game: Game, obs: Any) -> str:
        return "0*x"

    def stats(self) -> Any:

        return self._stats


def test_hybrid_counters_merge_through_play_match_and_markdown() -> None:
    import json
    import tempfile
    from pathlib import Path

    from eval.metrics import AgentLeaderRow
    from eval.runner import MatchConfig, StraightShotAgent, _render_markdown, play_match

    result = play_match(
        21, _HybridStubAgent(), StraightShotAgent(), MatchConfig(num_soldiers=2, max_turns=2)
    )
    stats = result.stats["stub"]
    assert stats.schema_errors == 2
    assert stats.waypoints_applied == 3
    assert stats.waypoints_dropped == 1

    row = AgentLeaderRow(agent="stub")
    row.merge(stats)
    with tempfile.TemporaryDirectory() as tmp:
        seed_file = Path(tmp) / "seeds.json"
        seed_file.write_text(json.dumps([]))
        markdown = _render_markdown(
            21, ("stub", "straight"), 1, MatchConfig(num_soldiers=2), [row], [result], seed_file
        )
    assert "## Hybrid planner telemetry" in markdown
    assert "| stub | 3 | 1 | 1 | 0 | 0 | 0 |" in markdown


# --- JSON extraction ------------------------------------------------------------------


def test_extract_json_object_tolerates_fences_and_prose() -> None:
    assert _extract_json_object('{"target_id": "enemy_0"}') == {"target_id": "enemy_0"}
    assert _extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json_object('prose {"a": "has { and } inside"} tail') == {
        "a": "has { and } inside"
    }
    assert _extract_json_object("no json at all") is None
    assert _extract_json_object("") is None


# --- roster grammar ---------------------------------------------------------------------


def test_make_agent_hybrid_grammar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-token")
    from eval.runner import make_agent

    agent = make_agent("hybrid:fake-model@howitzer", seed=0)
    assert isinstance(agent, HybridAgent)
    assert agent.name == "hybrid:fake-model@howitzer"

    plain = make_agent("hybrid:fake-model", seed=0)
    assert isinstance(plain, HybridAgent)
    assert plain.name == "hybrid:fake-model"

    with pytest.raises(ValueError, match="unknown persona"):
        make_agent("hybrid:fake-model@gremlin", seed=0)
