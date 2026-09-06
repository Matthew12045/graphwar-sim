"""LLMAgent: an Anthropic-Messages-API agent with a budgeted simulate tool.

The agent plays one soldier per turn through a **fresh conversation** (no
cross-turn memory):

- the **system prompt** is the shared core prepend adapted to the engine's
  REAL frame — centered world, shooter facing right, the auto vertical
  offset of ``graphwar_sim.physics.process_function_range`` (the frame
  contract of :class:`~agents.base.Observation`) — and the REAL
  :class:`~agents.simulate_tool.SimResult` fields. Persona style prompts are
  OUT of scope (the separate personas workstream,
  ``The_bridge_nobody_wrote_down.md``).
- the **per-turn user message** serializes the observation verbatim (the
  frame every other agent reads — coordinates are NOT re-derived, shifted,
  or re-centered) plus the live simulate budget ("simulate calls remaining:
  N", the M5.5.6 feedback format, fed by
  :attr:`BudgetedSimulator.remaining`).
- probes go through :class:`~agents.simulate_budget.BudgetedSimulator`
  (``DEFAULT_SIMULATE_BUDGET`` per turn); a denial returns a tool_result
  error that doubles as the loop's cost stop-signal. Denied calls never
  delegate (the wrapper raises first), so no oracle information leaks.
- a final answer is validated with the real parser; malformed emissions
  count ``parse_failures`` / ``retries`` (the same counters RandomAgent
  exposes) and get a reference-style correction quoting
  ``type(exc).__name__`` (the reference exception is message-free,
  ``docs/GROUND_TRUTH.md`` §3.6). Budget exhaustion (all attempts failed)
  returns the repo's safe dud ``"0*x"`` (``eval/runner.py:227``,
  ``agents/baselines.py:57``).
- ``stats()`` maps the counters the runner merges into the match stats
  (``eval/runner.py``): parse_failures, retries, simulate_calls, and
  simulate_denied (accumulated across turns).

Known limitation (documented, deliberately not engineered around): ``name``
is ``"llm:<model>"``, so a **mirror match** ``llm:X`` vs ``llm:X`` collides
in ``play_match``'s per-agent stats dict (keyed by agent name); round-robin
rosters cannot produce that pair. The UI match log carries the full name, so
spectators can still tell the sides apart.

The ``anthropic`` SDK is an OPTIONAL dependency
(``pip install 'graphwar-sim[llm]'``) and is imported lazily inside
:func:`_build_client` — no module-level import, so this module (and any
roster registration) is safe to import without the extra installed. Tests
inject fake clients via ``client=``; the agent relies only on
``messages.create(...)`` returning an object with ``.stop_reason`` and
``.content`` blocks (``.type``/``.text`` for text, ``.id``/``.name``/
``.input`` for tool_use) — never on SDK-internal types.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from graphwar_sim import Game, PolishNotationFunction, config
from graphwar_sim.parser import MalformedFunction

from .base import AgentStats, Observation
from .simulate_budget import BudgetedSimulator, SimulateBudgetExhausted
from .simulate_tool import SimResult

# The repo-wide last-resort emission: a guaranteed-parseable shot that goes
# nowhere (eval/runner.py:227, agents/baselines.py:57,92). # TUNABLE —
# not from source (repo convention).
SAFE_DUD: str = "0*x"

# Total emission attempts per turn (each attempt = API round-trips until a
# final answer, capped by _MAX_TOOL_ROUNDS_PER_TURN). Exhaustion returns the
# safe dud. # TUNABLE — not from source.
_DEFAULT_MAX_ATTEMPTS: int = 4

# API round-trips allowed within ONE attempt before that attempt is treated
# as failed (a model stuck emitting tool calls would otherwise never stop).
# Every dispatched round is answered with tool_results, so the conversation
# stays API-valid across attempts. # TUNABLE — not from source.
_MAX_TOOL_ROUNDS_PER_TURN: int = 8

# max_tokens for every messages.create call (the answer is one line).
# # TUNABLE — not from source.
_MAX_OUTPUT_TOKENS: int = 1024

# Error text for an over-budget simulate call (the loop's cost stop-signal).
_BUDGET_EXHAUSTED_TEXT = "simulate budget exhausted — commit your best expression now"

# Merge threshold (world units) for the compact terrain runs in the turn
# message; the observation grid step is ~0.97 world units (50 * 15 / 770).
# # TUNABLE — not from source (derives from observation.py's _TERRAIN_STEP).
_TERRAIN_RUN_GAP: float = 1.5

_SIMULATE_TOOL: dict[str, Any] = {
    "name": "simulate",
    "description": "Fire a candidate y=f(x) through the real physics WITHOUT "
    "applying kills. Probe before committing.",
    "input_schema": {
        "type": "object",
        "properties": {"expr": {"type": "string"}},
        "required": ["expr"],
    },
}

# The shared core prepend, adapted to the engine's REAL frame and grammar
# (the sketch's shooter-relative frame and {outcome, collision_point,
# nearest_miss_distance} return shape do NOT match the engine — see the
# module docstring). The SYNTAX section quotes the parser's tokenizer
# whitelist, not an embellishment of it (docs/GROUND_TRUTH.md §3.2).
_SYSTEM_PROMPT = """\
You are a Graphwar shot generator. You control one soldier.

FRAME (world coordinates — this is the exact space your expression is \
evaluated in): x in [-25, 25], y up. Your soldier 'M' sits at (sx, sy), \
given per turn; the projectile starts there and travels toward increasing \
x. The game shifts your curve VERTICALLY so it passes through the muzzle: \
effective curve = f(x) + (sy - f(sx)). Aim by choosing f so the shifted \
curve passes within hit radius (~0.45 world units) of an enemy's (x, y). \
Hitting a teammate is a critical failure. Terrain ('#' cells) ends the \
shot harmlessly. The game nudges the launch point along the curve's own \
tangent before the offset.

SYNTAX (the parser's exact tokenizer whitelist — nothing else exists): \
numbers, ( ) x + - * / ^, functions sqrt log (base 10) ln abs sin sen cos \
tan tg, constants e pi. Variable x ONLY (no y, no y'). Implicit \
multiplication works (2x, x(x+1), 2sin(x)). '-' is always unary (a-b is \
parsed as a+(-b)). Unknown characters are silently DROPPED by the \
tokenizer. Max 2000 chars; deeper than 64 nested terms is rejected \
(MalformedFunction → your shot is replaced by a safe dud).

TOOL: simulate(expr) fires a candidate through the real physics WITHOUT \
applying kills. Returns {parseable, hit_enemy, hit_teammate, num_hits, \
error}. You have a limited per-turn budget of calls (each turn message \
states the remaining count). Use them. Revise. Then commit.

OUTPUT: after your final simulate call, emit ONLY the bare expression on \
one line (no "y =", no prose, no code fence).\
"""

_LEADING_Y_EQUALS: re.Pattern[str] = re.compile(r"^y\s*=\s*", re.IGNORECASE)
_CODE_FENCE: re.Pattern[str] = re.compile(r"^```")


def _extract_candidate(text: str) -> str | None:
    """The candidate expression: the last non-empty, non-code-fence line,
    stripped of whitespace and an optional leading ``y =`` / ``y=``.
    ``None`` when no usable line survives (a max-tokens stop with no text
    lands here — the parse-failure path)."""
    kept = [line.strip() for line in text.splitlines()]
    kept = [line for line in kept if line and not _CODE_FENCE.match(line)]
    if not kept:
        return None
    expr = _LEADING_Y_EQUALS.sub("", kept[-1]).strip()
    return expr or None


def _validate(expr: str) -> str | None:
    """``None`` when ``expr`` is a legal emission; otherwise the correction
    reason (the reference-style exception name — the reference parser is
    message-free — or the harness length cap)."""
    if len(expr) > config.MAX_EXPR_CHARS:
        return f"expression exceeds the {config.MAX_EXPR_CHARS}-character cap"
    try:
        PolishNotationFunction(expr)
    except MalformedFunction as exc:
        return type(exc).__name__
    return None


def _sim_result_json(result: SimResult) -> str:
    """The simulate tool_result payload: exactly the fields the system prompt
    promises (``num_steps`` stays internal)."""
    return json.dumps(
        {
            "parseable": result.parseable,
            "hit_enemy": result.hit_enemy,
            "hit_teammate": result.hit_teammate,
            "num_hits": result.num_hits,
            "error": result.error,
        }
    )


def _response_text(response: Any) -> str:
    """All text blocks of a response, joined by newlines."""
    parts = [
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", "") == "text"
    ]
    return "\n".join(parts)


def _terrain_lines(blocks: tuple[tuple[float, float], ...]) -> list[str]:
    """Compact terrain serialization: rows sorted top-down, consecutive grid
    cells merged into ``a..b`` horizontal runs (the observation's coarse
    ~1-unit grid makes this lossless enough for aiming)."""
    rows: dict[float, list[float]] = {}
    for wx, wy in sorted(blocks, key=lambda p: (-p[1], p[0])):
        rows.setdefault(round(wy, 1), []).append(round(wx, 1))
    lines: list[str] = []
    for wy in sorted(rows, reverse=True):
        runs: list[tuple[float, float]] = []
        start = prev = rows[wy][0]
        for x in rows[wy][1:]:
            if x - prev <= _TERRAIN_RUN_GAP:
                prev = x
            else:
                runs.append((start, prev))
                start = prev = x
        runs.append((start, prev))
        parts = [f"{a:.1f}" if a == b else f"{a:.1f}..{b:.1f}" for a, b in runs]
        lines.append(f"y={wy:.1f}: " + " ".join(parts))
    return lines


def _turn_message(obs: Observation, remaining: int) -> str:
    """The per-turn user message: the observation verbatim + live budget."""

    def fmt(p: tuple[float, float]) -> str:
        return f"({p[0]:.1f}, {p[1]:.1f})"

    lines = [
        f"TURN {obs.turn_index} — TEAM {obs.team_id} fires. Frame: centered world, "
        "x in [-25, 25], y up — your expression is evaluated in EXACTLY these "
        "coordinates (do not shift, re-base, or re-center them).",
        "",
        "ASCII map ('M' you, 'S' ally, 'E' enemy, '#' terrain; top row y=+15, "
        "bottom y=-15, left x=-25, right x=+25):",
        obs.ascii_board,
        "",
        "Numeric positions (world coords, 1 decimal):",
        f"shooter (you): {fmt(obs.shooter)}",
        f"allies: {', '.join(fmt(p) for p in obs.own_soldiers) or 'none'}",
        f"enemies (nearest first): {', '.join(fmt(p) for p in obs.enemy_soldiers) or 'none'}",
        "terrain blocks (coarse ~1-unit grid, horizontal runs per row):",
        *_terrain_lines(obs.terrain_blocks),
        "",
        f"simulate calls remaining: {remaining}",
    ]
    return "\n".join(lines)


def _build_client() -> Any:
    """Construct the Anthropic client from the Claude-Code-style environment.

    Honors ``ANTHROPIC_BASE_URL`` (the 9arm-gateway convention) when set, and
    authenticates from ``ANTHROPIC_AUTH_TOKEN`` (Authorization: Bearer — the
    gateway's variable) falling back to ``ANTHROPIC_API_KEY`` (x-api-key).
    Raises BEFORE any network call when neither token variable is set,
    naming both — a misconfigured agent must fail at construction, never
    mid-match. The token value itself never enters the repo.
    """
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if token is None and api_key is None:
        raise RuntimeError(
            "LLMAgent needs an auth token: set ANTHROPIC_AUTH_TOKEN "
            "(Bearer; the Claude Code / 9arm-gateway convention) or "
            "ANTHROPIC_API_KEY (x-api-key) in the environment"
        )
    import anthropic  # optional dependency — imported only when a client is built

    kwargs: dict[str, Any] = {}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    if token is not None:
        kwargs["auth_token"] = token
    else:
        kwargs["api_key"] = api_key
    return anthropic.Anthropic(**kwargs)


class LLMAgent:
    """The M5.4 LLM shot generator (Anthropic tool-use + budgeted simulate)."""

    name: str

    def __init__(
        self,
        model: str,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self.name = f"llm:{model}"
        self._max_attempts = max(1, max_attempts)
        # Fail at construction on a missing token — never mid-match.
        self._client = client if client is not None else _build_client()
        self._stats = AgentStats()

    def act(self, game: Game, obs: Observation) -> str:
        """Emit this turn's expression (centered world frame).

        Opens a fresh conversation (system prompt + one per-turn user
        message), runs up to ``max_attempts`` attempts (see
        :meth:`_run_attempt`), and returns the safe dud on exhaustion. The
        :class:`BudgetedSimulator` is created and ``new_turn()``-ed per the
        wrapper's contract (the first call closes the empty construction
        turn — harmless; counters stay correct); its ledger counters are
        folded into :meth:`stats` after every turn.
        """
        sim = BudgetedSimulator(game)
        sim.new_turn()
        try:
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": _turn_message(obs, sim.remaining)}
            ]
            for _attempt in range(self._max_attempts):
                candidate = self._run_attempt(messages, sim)
                if candidate is not None:
                    return candidate
            return SAFE_DUD
        finally:
            self._stats.simulate_calls += sim.calls_used
            self._stats.simulate_denied += sim.denied_used

    def stats(self) -> AgentStats:
        return self._stats

    # -- internals ------------------------------------------------------------

    def _create(self, messages: list[dict[str, Any]]) -> Any:
        return self._client.messages.create(
            model=self._model,
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=_SYSTEM_PROMPT,
            tools=[_SIMULATE_TOOL],
            messages=messages,
        )

    def _run_attempt(self, messages: list[dict[str, Any]], sim: BudgetedSimulator) -> str | None:
        """One attempt: API round-trips until a final (non-tool_use) response.

        Returns the validated expression, or ``None`` when the attempt
        failed (malformed emission, no usable text, or the tool-round cap) —
        the caller moves on to the next attempt. Every failed emission
        increments the parse-failure/retry counters and queues a correction
        message for the re-call.
        """
        for _round in range(_MAX_TOOL_ROUNDS_PER_TURN):
            response = self._create(messages)
            if response.stop_reason == "tool_use":
                self._dispatch_tool_round(response, sim, messages)
                continue
            expr = _extract_candidate(_response_text(response))
            messages.append({"role": "assistant", "content": response.content})
            if expr is None:
                self._fail_attempt(messages, "no expression found in the response")
                return None
            reason = _validate(expr)
            if reason is None:
                return expr
            self._fail_attempt(messages, f"{reason} (the reference parser reports no diagnostics)")
            return None
        # Tool-round cap breached mid-attempt: every tool_use was already
        # answered inside the loop, so the conversation is API-valid; nudge
        # for a final answer and fail the attempt.
        nudge = {
            "type": "text",
            "text": "Too many tool rounds — emit ONLY the bare expression on "
            "one line, no more tool calls.",
        }
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list):
            last["content"].append(nudge)
        else:
            messages.append({"role": "user", "content": [nudge]})
        return None

    def _dispatch_tool_round(
        self,
        response: Any,
        sim: BudgetedSimulator,
        messages: list[dict[str, Any]],
    ) -> None:
        """Answer every tool_use block through the budgeted simulator.

        A denied call becomes a tool_result error WITHOUT delegating (the
        wrapper raises before touching the oracle); the JSON payload carries
        exactly the SimResult fields the system prompt promises. An unknown
        tool name gets an error result too, so the conversation stays
        API-valid.
        """
        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if getattr(block, "type", "") != "tool_use":
                continue
            tool_use_id = getattr(block, "id", "")
            tool_name = getattr(block, "name", "")
            raw_input = getattr(block, "input", {})
            expr = raw_input.get("expr", "") if isinstance(raw_input, dict) else ""
            if tool_name != "simulate":
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": f"unknown tool: {tool_name}",
                        "is_error": True,
                    }
                )
                continue
            try:
                result = sim.simulate(expr)
            except SimulateBudgetExhausted:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": _BUDGET_EXHAUSTED_TEXT,
                        "is_error": True,
                    }
                )
                continue
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": _sim_result_json(result),
                }
            )
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    def _fail_attempt(self, messages: list[dict[str, Any]], reason: str) -> None:
        """Record a malformed emission (the counters RandomAgent exposes) and
        queue the correction message for the re-call."""
        self._stats.parse_failures += 1
        self._stats.retries += 1
        messages.append(
            {
                "role": "user",
                "content": f"{reason} — emit ONLY the bare y = f(x) expression on one line.",
            }
        )


__all__ = ["LLMAgent"]
