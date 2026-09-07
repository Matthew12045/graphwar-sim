"""Tests for M5.4 Slice A: the persona harness.

- manifest enumeration (bridge M5.4.0: never a literal count),
- prompt composition (persona appended, core byte-identical),
- one verifier unit test per persona on real seeded boards,
- the x-monotone trajectory invariant the magician's num_hits inference
  rests on (physics.py:216-228),
- the make_agent roster grammar (``llm:<model>[@<persona]``),
- AgentStats merge through play_match.

Zero network: LLM tests inject the same fake-client pattern as
tests/test_llm_agent.py; verifier oracle calls hit the REAL free oracle
(``agents.simulate_tool.simulate``) on REAL seeded games.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from agents import AgentStats, StraightShotAgent
from agents.llm_agent import _SYSTEM_PROMPT, LLMAgent
from agents.observation import observe
from agents.personas import PERSONAS, resolve_verifier
from agents.personas import texts as persona_texts
from agents.personas.verifiers import (
    VERIFIERS,
    VerdictKind,
    VerifierContext,
    _sniper_candidates,
    _terrain_circles_world,
    verify_bodyguard,
    verify_howitzer,
    verify_master_magician,
    verify_professor,
    verify_serpent,
    verify_sniper,
)
from eval.metrics import AgentLeaderRow, AgentMatchStats
from eval.runner import MatchConfig, _split_persona, make_agent, play_match
from graphwar_sim import TEAM2, Game, PolishNotationFunction
from graphwar_sim.physics import process_function_range

# --- fixtures (deterministic boards, probed once and pinned) ------------------
#
# seed 9, 1 soldier: the fitted line hits the lone enemy.
# seed 3, 2 soldiers: the fitted line hits exactly one of two enemies.
# seed 21, 2 soldiers: "-1.4117(x+18.117)" strikes the ALLY; one teammate at
#   (-9.805, -3.636); enemy at (16.948, 14.091) and (22.597, 3.117).
# seed 5, 1 soldier: corridor terrain top ~4.09; column x=0 is terrain-free.
# seed 1, 1 soldier: a terrain circle centered near x=-10.26 right of the
#   muzzle at (-11.30, 0.78).


def _game_and_obs(seed: int, num_soldiers: int) -> tuple[Game, object]:
    game = Game.create(seed, num_soldiers=num_soldiers)
    return game, observe(game)


def _ctx(text: str = "") -> VerifierContext:
    return VerifierContext(assistant_text=text)


# --- 1. manifest enumeration (M5.4.0) ------------------------------------------


def _discovered_style_ids() -> set[str]:
    """The ids the texts module actually carries — the enumeration the
    manifest must match (a literal count is forbidden)."""
    return {
        name[: -len("_STYLE")].lower()
        for name in vars(persona_texts)
        if name.isupper() and name.endswith("_STYLE")
    }


def test_manifest_enumerates_the_style_texts() -> None:
    assert set(PERSONAS) == _discovered_style_ids()


def test_manifest_metadata_is_complete_and_resolvable() -> None:
    assert set(PERSONAS) == {
        "sniper",
        "howitzer",
        "serpent",
        "bodyguard",
        "professor",
        "master_magician",
    }
    for spec in PERSONAS.values():
        assert spec.id in PERSONAS
        assert spec.style_text
        assert spec.verifier in VERIFIERS
        assert resolve_verifier(spec) is VERIFIERS[spec.verifier]
        assert spec.degrades_to in {kind.value for kind in VerdictKind}


def test_gremlin_is_excluded() -> None:
    # Designer call: file.txt (Gremlin) has no verifier and a cross-turn
    # rule the fresh-conversation agent cannot honor.
    assert "gremlin" not in PERSONAS


def test_style_texts_are_adapted_to_the_real_frame() -> None:
    for spec in PERSONAS.values():
        text = spec.style_text
        assert "exp(" not in text  # the parser has no exp — e^(...) only
        assert "MAX_LENGTH" not in text
        assert "BLAST_RADIUS" not in text
        assert "min_dy" not in text
        assert "simulate" in text


def test_curve_personas_quote_parser_legal_syntax() -> None:
    # The sigmoid step is parser-legal (uses e^, /, parentheses).
    PolishNotationFunction(
        "c/(1+e^(-b*(x - k)))".replace("c", "1").replace("b", "1").replace("k", "2")
    )
    PolishNotationFunction("0.5*sin(3x)*e^(-0.05x)")
    PolishNotationFunction("1*e^(-9*(x-2)^2)")
    for pid in ("sniper", "serpent", "master_magician"):
        assert "e^(" in PERSONAS[pid].style_text, pid


# --- 2. prompt composition + identity (plan A2) --------------------------------


class _FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        raise AssertionError("tests below never need a real round")


class _FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def test_persona_appends_style_after_the_byte_identical_core() -> None:
    client = _FakeClient()
    agent = LLMAgent(model="fake-model", client=client, persona="sniper")
    # The composed prompt is only referenced through _create's kwargs; read
    # the attribute directly (composition is the contract under test).
    assert agent._system_prompt.startswith(_SYSTEM_PROMPT)
    assert agent._system_prompt[len(_SYSTEM_PROMPT) :] == ("\n\n" + PERSONAS["sniper"].style_text)


def test_no_persona_keeps_the_core_prompt_byte_identical() -> None:
    agent = LLMAgent(model="fake-model", client=_FakeClient())
    assert agent._system_prompt == _SYSTEM_PROMPT
    assert agent.name == "llm:fake-model"
    assert not hasattr(agent, "rung_history")  # rung_counts output unchanged


def test_persona_rides_the_name_and_decollides_mirror_matches() -> None:
    plain = LLMAgent(model="fake-model", client=_FakeClient())
    sniper = LLMAgent(model="fake-model", client=_FakeClient(), persona="sniper")
    assert plain.name == "llm:fake-model"
    assert sniper.name == "llm:fake-model@sniper"
    assert plain.name != sniper.name


def test_unknown_persona_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="unknown persona"):
        LLMAgent(model="fake-model", client=_FakeClient(), persona="gremlin")


def test_persona_agent_keeps_a_rung_history() -> None:
    agent = LLMAgent(model="fake-model", client=_FakeClient(), persona="howitzer")
    assert agent.rung_history == []


# --- 3. sniper verifier ---------------------------------------------------------


def test_sniper_passes_a_bottom_rung_line_without_oracle_calls() -> None:
    game, obs = _game_and_obs(9, 1)
    verdict = verify_sniper("0.05*x", game, obs, _ctx())
    assert verdict.kind is VerdictKind.PASS


def test_sniper_flags_an_arc_when_a_fitted_line_hits() -> None:
    game, obs = _game_and_obs(9, 1)
    lines = _sniper_candidates(0, obs)
    assert lines, "seed-9 fixture must admit a line candidate"
    assert any(simulate_hit(game, c) for c in lines), "fixture precondition"
    verdict = verify_sniper("-0.01*x*(x-10)", game, obs, _ctx())
    assert verdict.kind is VerdictKind.VIOLATION
    assert "m*x" in verdict.reason


def test_sniper_passes_an_unclassifiable_emission() -> None:
    game, obs = _game_and_obs(9, 1)
    assert verify_sniper("sin(x)*x", game, obs, _ctx()).kind is VerdictKind.PASS


def simulate_hit(game: Game, expr: str) -> bool:
    from agents.simulate_tool import simulate

    return simulate(game, expr).hit_enemy


# --- 4. howitzer verifier --------------------------------------------------------


def _corridor_terrain_top(game: Game, obs: object) -> float:
    mirrored = obs.team_id == TEAM2  # type: ignore[attr-defined]
    x0 = obs.shooter[0]  # type: ignore[attr-defined]
    x1 = max(ex for ex, _ in obs.enemy_soldiers)  # type: ignore[attr-defined]
    circles = [
        (cx, cy, r)
        for cx, cy, r in _terrain_circles_world(game, mirrored)
        if cx + r >= x0 and cx - r <= x1
    ]
    return max((cy + r for _, cy, r in circles), default=float("-inf"))


def test_howitzer_passes_a_peak_well_above_the_corridor_terrain() -> None:
    game, obs = _game_and_obs(5, 1)
    terrain_top = _corridor_terrain_top(game, obs)
    peak = terrain_top + 100.0
    a = 4.0 * peak / 100.0  # L=10 -> vertex at x=5, inside seed-5 corridor
    verdict = verify_howitzer(f"-{a:.6f}*x*(x-10)", game, obs, _ctx())
    assert verdict.kind is VerdictKind.PASS


def test_howitzer_flags_a_low_curve() -> None:
    game, obs = _game_and_obs(5, 1)
    assert _corridor_terrain_top(game, obs) > -8.0  # fixture precondition
    verdict = verify_howitzer("0*x", game, obs, _ctx())
    assert verdict.kind is VerdictKind.VIOLATION


# --- 5. serpent verifier ---------------------------------------------------------


def _first_block_center_right_of_muzzle(game: Game, obs: object) -> float:
    mirrored = obs.team_id == TEAM2  # type: ignore[attr-defined]
    circles = _terrain_circles_world(game, mirrored)
    return min(cx for cx, _, _ in circles if cx > obs.shooter[0])  # type: ignore[attr-defined]


def test_serpent_passes_crossings_in_terrain_free_columns() -> None:
    # seed 5, w=0.05: the only corridor crossing (x=0) sits in a free column.
    game, obs = _game_and_obs(5, 1)
    verdict = verify_serpent("sin(0.05*x)*e^(-0.05*x)", game, obs, _ctx())
    assert verdict.kind is VerdictKind.PASS


def test_serpent_flags_a_crossing_inside_a_terrain_column() -> None:
    game, obs = _game_and_obs(1, 1)
    cx = _first_block_center_right_of_muzzle(game, obs)
    # A crossing exactly at the circle's center: w = -pi/x_c puts x = -pi/w
    # (and multiples) on the blocked column.
    w = -math.pi / cx
    verdict = verify_serpent(f"sin({w:.6f}*x)*e^(-0.05*x)", game, obs, _ctx())
    assert verdict.kind is VerdictKind.VIOLATION
    assert "zero crossing" in verdict.reason


# --- 6. bodyguard verifier --------------------------------------------------------


def test_bodyguard_passes_a_curve_clear_of_the_teammate_band() -> None:
    game, obs = _game_and_obs(21, 2)
    verdict = verify_bodyguard("5", game, obs, _ctx())
    assert verdict.kind is VerdictKind.PASS


def test_bodyguard_hard_fails_inside_the_band() -> None:
    game, obs = _game_and_obs(21, 2)
    verdict = verify_bodyguard("-3.6", game, obs, _ctx())
    assert verdict.kind is VerdictKind.VIOLATION
    assert "forbidden band" in verdict.reason


# --- 7. professor verifier (structural, the weakest) ------------------------------


def test_professor_passes_when_every_constraint_is_stated() -> None:
    game, obs = _game_and_obs(21, 2)
    text = "g(16.9) = 14.1\ng(22.6) = 3.1\n|g(-9.8) - (-3.6)| > 0.45\n0.05*x"
    verdict = verify_professor("0.05*x", game, obs, _ctx(text))
    assert verdict.kind is VerdictKind.PASS


def test_professor_flags_a_missing_teammate_constraint() -> None:
    game, obs = _game_and_obs(21, 2)
    text = "g(16.9) = 14.1\ng(22.6) = 3.1"
    verdict = verify_professor("0.05*x", game, obs, _ctx(text))
    assert verdict.kind is VerdictKind.VIOLATION
    assert "teammate" in verdict.reason


def test_professor_flags_a_missing_enemy_constraint() -> None:
    game, obs = _game_and_obs(21, 2)
    text = "|g(-9.8) - (-3.6)| > 0.45"
    verdict = verify_professor("0.05*x", game, obs, _ctx(text))
    assert verdict.kind is VerdictKind.VIOLATION
    assert "enemy" in verdict.reason


def test_professor_flags_an_empty_text() -> None:
    game, obs = _game_and_obs(21, 2)
    assert verify_professor("0.05*x", game, obs, _ctx("")).kind is VerdictKind.VIOLATION


def test_professor_ignores_numbers_glued_into_larger_ones() -> None:
    game, obs = _game_and_obs(21, 2)
    # "116.9" must NOT satisfy the "16.9" enemy constraint.
    text = "g(116.9) = 141.1\ng(22.6) = 3.1\n|g(-9.8) - (-3.6)| > 0.45"
    verdict = verify_professor("0.05*x", game, obs, _ctx(text))
    assert verdict.kind is VerdictKind.VIOLATION


# --- 8. master magician verifier (M5.4.2) ------------------------------------------


def test_magician_full_on_a_single_enemy_board() -> None:
    game, obs = _game_and_obs(9, 1)
    line = _sniper_candidates(0, obs)[0]
    verdict = verify_master_magician(line, game, obs, _ctx())
    assert verdict.kind is VerdictKind.MAGICIAN_FULL
    assert verdict.hits == 1 and verdict.living == 1
    assert verdict.rung == "MAGICIAN_FULL"


def test_magician_partial_is_first_class() -> None:
    game, obs = _game_and_obs(3, 2)
    from agents.simulate_tool import simulate

    line = next(c for c in _sniper_candidates(0, obs) if simulate(game, c).hit_enemy)
    verdict = verify_master_magician(line, game, obs, _ctx())
    assert verdict.kind is VerdictKind.MAGICIAN_PARTIAL
    assert (verdict.hits, verdict.living) == (1, 2)
    assert verdict.rung == "MAGICIAN_PARTIAL(1,2)"


def test_magician_subtracts_teammate_hits_from_num_hits() -> None:
    # Seed 21: the line strikes the ALLY — num_hits=1 with hit_teammate must
    # be ZERO enemy hits (VIOLATION), not a phantom partial.
    game, obs = _game_and_obs(21, 2)
    verdict = verify_master_magician("-1.4117(x+18.117)", game, obs, _ctx())
    assert verdict.kind is VerdictKind.VIOLATION
    assert (verdict.hits, verdict.living) == (0, 2)


def test_magician_flags_a_clean_miss() -> None:
    game, obs = _game_and_obs(21, 2)
    verdict = verify_master_magician("0*x", game, obs, _ctx())
    assert verdict.kind is VerdictKind.VIOLATION


def test_x_monotone_trajectory_invariant() -> None:
    """The magician's num_hits inference rests on the trajectory's x being
    strictly increasing (physics.py:216-228) — pin the invariant directly:
    each enemy is then reached at most once."""
    for seed, expr in ((9, "0.44*x"), (21, "-1.4117(x+18.117)"), (5, "0.05*x*x")):
        game = Game.create(seed, num_soldiers=2)
        f = PolishNotationFunction(expr)
        shooter = game.state.current_team().current_soldier()
        inverted = game.state.current_team().team == TEAM2
        result = process_function_range(f, shooter, game.all_soldiers(), game.terrain, inverted)
        points = result.points[: result.num_steps]
        assert len(points) >= 2, (seed, expr)
        xs = [p[0] for p in points]
        assert all(b > a for a, b in zip(xs, xs[1:], strict=False)), (seed, expr)


def test_verifiers_never_touch_the_budget_wrapper() -> None:
    """Bridge M5.4.2: the verifiers' oracle is the FREE simulate tool — the
    module must never IMPORT the agent's budget wrapper (docstring mentions
    of the mandate are fine; import statements are not)."""
    import ast

    tree = ast.parse(Path("agents/personas/verifiers.py").read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("simulate_budget" in name for name in imported)
    assert not any("BudgetedSimulator" in name for name in imported)


# --- 9. LLMAgent verdict flow (fake client, real oracle) ----------------------------
#
# The fake-client pattern mirrors tests/test_llm_agent.py.


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, stop_reason: str, content: list) -> None:
        self.stop_reason = stop_reason
        self.content = content


class _ScriptedMessages:
    def __init__(self, responses: list[_Response]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class _ScriptedClient:
    def __init__(self, responses: list[_Response]) -> None:
        self.messages = _ScriptedMessages(responses)


def test_persona_verdict_flows_into_rung_history_and_stats() -> None:
    agent = LLMAgent(
        model="fake-model",
        client=_ScriptedClient([_Response("end_turn", [_TextBlock("0.05*x")])]),
        persona="sniper",
    )
    game, obs = _game_and_obs(5, 1)
    assert agent.act(game, obs) == "0.05*x"
    assert agent.rung_history == ["PASS"]
    stats = agent.stats()
    assert stats.constraint_checks == 1
    assert stats.constraint_violations == 0
    assert stats.magician_partials == 0
    # The verifier stayed out of the agent's budgeted oracle accounting.
    assert stats.simulate_calls == 0


def test_persona_violation_flows_into_rung_history_and_stats() -> None:
    agent = LLMAgent(
        model="fake-model",
        client=_ScriptedClient([_Response("end_turn", [_TextBlock("-0.01*x*(x-10)")])]),
        persona="sniper",
    )
    game, obs = _game_and_obs(9, 1)  # the fitted line hits -> arc is a violation
    assert agent.act(game, obs) == "-0.01*x*(x-10)"
    assert agent.rung_history == ["CONSTRAINT_VIOLATION"]
    stats = agent.stats()
    assert stats.constraint_checks == 1
    assert stats.constraint_violations == 1
    assert stats.simulate_calls == 0  # the verifier's oracle call was FREE


def test_professor_verifier_reads_the_captured_assistant_text() -> None:
    text = "g(16.9) = 14.1\ng(22.6) = 3.1\n|g(-9.8) - (-3.6)| > 0.45\n0.05*x"
    agent = LLMAgent(
        model="fake-model",
        client=_ScriptedClient([_Response("end_turn", [_TextBlock(text)])]),
        persona="professor",
    )
    game, obs = _game_and_obs(21, 2)
    assert agent.act(game, obs) == "0.05*x"
    assert agent.rung_history == ["PASS"]

    # Without a stated teammate constraint the same emission violates.
    bare = LLMAgent(
        model="fake-model",
        client=_ScriptedClient([_Response("end_turn", [_TextBlock("g(16.9) = 14.1\n0.05*x")])]),
        persona="professor",
    )
    assert bare.act(game, obs) == "0.05*x"
    assert bare.rung_history == ["CONSTRAINT_VIOLATION"]


def test_magician_partial_rides_the_rung_line() -> None:
    agent = LLMAgent(
        model="fake-model",
        client=_ScriptedClient([_Response("end_turn", [_TextBlock("0.067796610169*x")])]),
        persona="master_magician",
    )
    game, obs = _game_and_obs(3, 2)  # the line hits exactly one of two enemies
    assert agent.act(game, obs) == "0.067796610169*x"
    assert agent.rung_history == ["MAGICIAN_PARTIAL(1,2)"]
    stats = agent.stats()
    assert stats.magician_partials == 1
    assert stats.constraint_checks == 1


# --- 10. roster grammar (plan A5) ---------------------------------------------------


def test_split_persona() -> None:
    assert _split_persona("m") == ("m", None)
    assert _split_persona("m@sniper") == ("m", "sniper")
    assert _split_persona("a@b@c") == ("a@b", "c")  # the LAST '@' wins
    with pytest.raises(ValueError, match="missing model"):
        _split_persona("@sniper")
    with pytest.raises(ValueError, match="missing persona"):
        _split_persona("m@")


def test_make_agent_grammar_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-token")
    agent = make_agent("llm:fake-model@sniper", seed=0)
    assert isinstance(agent, LLMAgent)
    assert agent.name == "llm:fake-model@sniper"
    assert agent._persona is PERSONAS["sniper"]

    plain = make_agent("llm:fake-model", seed=0)
    assert isinstance(plain, LLMAgent)
    assert plain.name == "llm:fake-model"
    assert plain._persona is None


def test_make_agent_unknown_persona_is_a_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-token")
    with pytest.raises(ValueError, match="unknown persona"):
        make_agent("llm:fake-model@gremlin", seed=0)


# --- 11. stats merge (plan A4) -------------------------------------------------------


class _PersonaStubAgent:
    """A minimal agent whose stats carry persona counters (merge test)."""

    name = "stub"

    def __init__(self) -> None:
        self._stats = AgentStats(
            constraint_checks=2,
            constraint_violations=1,
            magician_partials=1,
        )

    def act(self, game: Game, obs: object) -> str:  # type: ignore[no-untyped-def]
        return "0*x"

    def stats(self) -> AgentStats:
        return self._stats


def test_persona_counters_merge_through_play_match() -> None:
    result = play_match(
        21,
        _PersonaStubAgent(),
        StraightShotAgent(),
        MatchConfig(num_soldiers=2, max_turns=2),
    )
    stats = result.stats["stub"]
    assert stats.constraint_checks == 2
    assert stats.constraint_violations == 1
    assert stats.magician_partials == 1


def test_leaderboard_row_merges_persona_counters() -> None:
    row = AgentLeaderRow(agent="stub")
    row.merge(AgentMatchStats(agent="stub", constraint_checks=3, constraint_violations=2))
    row.merge(AgentMatchStats(agent="stub", constraint_checks=1, magician_partials=1))
    assert row.constraint_checks == 4
    assert row.constraint_violations == 2
    assert row.magician_partials == 1
