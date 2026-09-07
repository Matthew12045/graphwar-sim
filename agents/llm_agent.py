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
``messages.create(...)`` / ``messages.stream(...)`` returning a message
object with ``.stop_reason`` and ``.content`` blocks (``.type``/``.text``
for text, ``.id``/``.name``/``.input`` for tool_use) — never on
SDK-internal types. Long thinking responses are STREAMED when the client
supports it (the gateway's Cloudflare proxy times out non-streaming
generations at 120s); the fakes exercise the non-streaming fallback.
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

# API round-trips allowed per TURN (the plan caps total round-trips per
# turn; a budget-burned round costs one but does not end an attempt).
# # TUNABLE — not from source.
_MAX_TOOL_ROUNDS_PER_TURN: int = 8

# The last rounds of a turn force a text-only commit (tool_choice "none") —
# a stochastic thinker that keeps calling tools otherwise. # TUNABLE —
# not from source.
_COMMIT_ROUNDS: int = 2

# max_tokens for every API call — the model's full 128k context budget
# (user-locked). History: at the DEFAULT reasoning effort (xhigh) the model
# burned whatever budget it got on hidden thinking and died at the gateway's
# ~125s Cloudflare wall, so 16384 was once the safe ceiling; with
# reasoning_effort=medium (see below) the thinking is CAPPED, rounds
# conclude in seconds-to-minutes, and the budget is pure headroom (measured:
# 128000 accepted, round concluded in 10.7s / 1741 tokens). Streaming keeps
# even the worst case inside the wall. # TUNABLE — not from source.
_MAX_OUTPUT_TOKENS: int = 128000

# Retries for transient gateway deaths on ONE API call (Cloudflare's ~120s
# proxy read limit truncates long thinking generations: 524 non-streaming,
# a truncated stream mid-flight). Each retry regenerates from scratch.
# # TUNABLE — not from source.
_MAX_API_RETRIES: int = 2

# Exception TYPE NAMES retried by _create (duck-typed: the anthropic SDK and
# its httpx transport raise these; the SDK does not auto-retry mid-stream
# disconnects). Non-retryable errors (auth, bad request) propagate.
_RETRYABLE_API_ERROR_NAMES: frozenset[str] = frozenset(
    {
        "APIConnectionError",
        "InternalServerError",
        "RemoteProtocolError",
        "ReadTimeout",
        "ConnectTimeout",
        "ReadError",
        "WriteError",
        "ConnectError",
    }
)


def _is_retryable_api_error(exc: Exception) -> bool:
    """True for transient transport/gateway failures (see
    ``_RETRYABLE_API_ERROR_NAMES``) — name-based so the module needs no
    import from the optional SDK or its transport."""
    return type(exc).__name__ in _RETRYABLE_API_ERROR_NAMES


# Reasoning effort for gateway-served reasoning models (litellm -> vLLM),
# merged as ``extra_body`` on every call. Measured on gateway.9arm.co /
# qwen3.8-27b-fp8 (2026-09-06): the DEFAULT effort (xhigh) thinks to the
# ENTIRE output budget in ~7/8 rounds — conclusions ~1/8 and unpredictable —
# so rounds die at the gateway's ~125s Cloudflare wall. "medium" (Claude
# Code's CLAUDE_CODE_EFFORT_LEVEL) caps the thinking: rounds concluded in
# 32-61s with valid tool calls (2/2). vLLM accepts xhigh (default) | medium
# | low only. Sent ONLY through gateways: real Anthropic APIs reject
# unknown body fields. # TUNABLE — not from source.
_REASONING_EFFORT: str = "medium"


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
# whitelist, not an embellishment of it (docs/GROUND_TRUTH.md §3.2). The
# THINK BRIEFLY paragraph targets the measured failure mode: the model
# happily burns its whole output budget on closed-form ballistic
# derivations; probing with simulate is both cheaper and the tool's point
# (see _MAX_OUTPUT_TOKENS for the gateway wall arithmetic).
_SYSTEM_PROMPT = """\
You are a Graphwar shot generator. You control one soldier.

FRAME: world coordinates — the exact space your expression is evaluated in. \
x in [-25, 25], y up. Your soldier 'M' sits at (sx, sy), given per turn; \
the projectile starts there and travels toward increasing x. The game \
shifts your curve VERTICALLY so it passes through the muzzle: effective \
curve = f(x) + (sy - f(sx)); it also nudges the launch point along the \
curve's own tangent first. Aim so the shifted curve passes within hit \
radius (~0.45 world units) of an enemy's (x, y). Hitting a teammate is a \
critical failure.

HARD RULES: the playable plane is y in [-14.6, +14.6]; the shot is KILLED \
the instant the curve leaves that band or touches '#' terrain — it must \
survive all the way to the enemy's x. Check the terrain immediately to the \
right of your own muzzle first: if it is a wall, launch DESCENDING so the \
auto-offset still lifts the curve while it slips under or around the rock.

SYNTAX (the parser's exact tokenizer whitelist — nothing else exists): \
numbers, ( ) x + - * / ^, functions sqrt log (base 10) ln abs sin sen cos \
tan tg, constants e pi. Variable x ONLY (no y, no y'). Implicit \
multiplication works (2x, x(x+1), 2sin(x)). '-' is always unary (a-b parses \
as a+(-b)). Unknown characters are silently DROPPED. Max 2000 chars; \
deeper than 64 nested terms is rejected (your shot is replaced by a safe \
dud).

THINK BRIEFLY. Do NOT derive trajectories analytically — probe with \
simulate and correct from the telemetry: fix the height when \
miss_direction says you passed on the wrong side, but FIRST make sure the \
shot reaches the enemy at all (stop_reason "short"/"terrain"/"off_map" \
means it died before getting there).

TOOL: simulate(expr) fires a candidate through the real physics WITHOUT \
applying kills. Returns {parseable, hit_enemy, hit_teammate, num_hits, \
error, nearest_miss, miss_direction, stopped_at_x, stop_reason}: \
nearest_miss = world-unit distance from the nearest enemy (~0.45 = hit), \
miss_direction = "high"/"low" (which side of the enemy the curve passed \
on), stopped_at_x = the world x where the shot ended, stop_reason = "hit" \
| "terrain" | "off_map" | "short" | "passed". You have a limited per-turn \
budget of calls (each turn message states the remaining count). Revise. \
Then commit.

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
    promises (``num_steps`` stays internal). The miss telemetry is the
    model's only gradient — a binary hit/miss hides "stopped short" from
    "wrong height" (live diagnosis 2026-09-07)."""
    nearest = round(result.nearest_miss, 3) if result.nearest_miss is not None else None
    stop_x = round(result.stopped_at_x, 2) if result.stopped_at_x is not None else None
    return json.dumps(
        {
            "parseable": result.parseable,
            "hit_enemy": result.hit_enemy,
            "hit_teammate": result.hit_teammate,
            "num_hits": result.num_hits,
            "error": result.error,
            "nearest_miss": nearest,
            "miss_direction": result.miss_direction,
            "stopped_at_x": stop_x,
            "stop_reason": result.stop_reason,
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


def _turn_message(obs: Observation, remaining: int | str) -> str:
    """The per-turn user message: the observation verbatim + live budget
    (``remaining`` is a count, or the string ``"unlimited"``)."""

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
        reasoning_effort: str | None = None,
        simulate_budget: int | None = None,
        tool_rounds: int | None = None,
    ) -> None:
        self._model = model
        self.name = f"llm:{model}"
        self._max_attempts = max(1, max_attempts)
        # User-locked live default (M5.4): UNLIMITED simulate calls per turn
        # — the model probes as much as it wants; the turn message states
        # "unlimited" and no denial stop-signal fires. Pass an int for the
        # M5.3-style ablation grid (BudgetedSimulator handles the cap and
        # the denial accounting either way).
        self._simulate_budget = simulate_budget
        # API round-trips per turn: the loop terminator (a turn ends on a
        # commit, an attempts exhaustion, or this cap). None -> module
        # default.
        self._tool_rounds = tool_rounds if tool_rounds is not None else _MAX_TOOL_ROUNDS_PER_TURN
        # Live gateway measurement (see _REASONING_EFFORT): the default
        # effort (xhigh) burns the whole output budget on hidden thinking
        # and rounds die at the gateway's ~125s wall; "medium" caps it and
        # rounds conclude reliably. Auto: apply through gateways only —
        # real Anthropic APIs reject unknown body fields.
        if reasoning_effort is None and os.environ.get("ANTHROPIC_BASE_URL"):
            reasoning_effort = _REASONING_EFFORT
        self._reasoning_effort = reasoning_effort
        # Fail at construction on a missing token — never mid-match.
        self._client = client if client is not None else _build_client()
        self._stats = AgentStats()
        # The last simulate-call expression this turn (the round-cap
        # fallback fires it when the model never emits a text commit).
        self._last_probe_expr: str | None = None

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
        sim = BudgetedSimulator(game, budget=self._simulate_budget)
        sim.new_turn()
        self._last_probe_expr = None
        try:
            budget_text: int | str = "unlimited" if sim.unlimited else sim.remaining
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": _turn_message(obs, budget_text)}
            ]
            rounds_left = self._tool_rounds
            for _attempt in range(self._max_attempts):
                candidate, rounds_used = self._run_attempt(messages, sim, rounds_left)
                rounds_left -= rounds_used
                if candidate is not None:
                    return candidate
                if rounds_left <= 0:
                    break
            # Round cap hit without a text commit: fire the model's LAST
            # probed expression — it chose it through real simulate feedback,
            # which beats the safe dud. (The gateway ignores
            # tool_choice "none", so a forced text commit cannot be relied
            # on; this fallback guarantees the probing still pays off.)
            return self._last_probe_expr or SAFE_DUD
        except Exception:  # noqa: BLE001 - the match must never crash on one turn
            # Unrecoverable after the API retries (e.g. the gateway killed
            # every regeneration): degrade this turn to the safe dud,
            # accounted like the runner's defensive malformed-emission branch.
            self._stats.parse_failures += 1
            self._stats.retries += 1
            return self._last_probe_expr or SAFE_DUD
        finally:
            self._stats.simulate_calls += sim.calls_used
            self._stats.simulate_denied += sim.denied_used

    def stats(self) -> AgentStats:
        return self._stats

    # -- internals ------------------------------------------------------------

    def _create(self, messages: list[dict[str, Any]], force_commit: bool = False) -> Any:
        """One API round-trip: streaming when the client supports it.

        The 9arm gateway serves a reasoning model that thinks for MINUTES
        before acting and forwards nothing until a whole block finishes, so
        Cloudflare's ~120s proxy read limit kills long generations mid-call
        (524 non-streaming, truncated stream mid-flight). Each retry
        regenerates from scratch — a fresh generation may land under the
        limit. ``force_commit`` sets ``tool_choice: "none"`` so the model
        MUST answer in text (the last rounds of a turn; a stochastic thinker
        that never commits otherwise). Clients that expose
        ``messages.stream`` get the streaming path; test fakes expose only
        ``messages.create`` and take the non-streaming fallback — the
        response surface is identical (``.stop_reason`` + ``.content``
        blocks).
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "system": _SYSTEM_PROMPT,
            "tools": [_SIMULATE_TOOL],
            "messages": messages,
        }
        if force_commit:
            kwargs["tool_choice"] = {"type": "none"}
        if self._reasoning_effort is not None:
            kwargs["extra_body"] = {"reasoning_effort": self._reasoning_effort}
        stream_factory = getattr(self._client.messages, "stream", None)
        for retry in range(_MAX_API_RETRIES + 1):
            try:
                if callable(stream_factory):
                    with stream_factory(**kwargs) as stream:
                        return stream.get_final_message()
                return self._client.messages.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - see _is_retryable_api_error
                if retry >= _MAX_API_RETRIES or not _is_retryable_api_error(exc):
                    raise
                continue
        raise AssertionError("unreachable")  # for the type checker

    def _run_attempt(
        self,
        messages: list[dict[str, Any]],
        sim: BudgetedSimulator,
        rounds_left: int,
    ) -> tuple[str | None, int]:
        """One attempt within the turn's TOTAL round budget (the plan caps
        API round-trips per TURN, not per attempt — a stochastic thinker
        needs every round it can get; a budget-burned round with no usable
        text is counted and corrected but does NOT consume an attempt).

        Returns ``(candidate | None, rounds_consumed)``. ``None`` moves the
        caller to the next attempt: a malformed candidate, or the turn
        round-cap breach. The last ``_COMMIT_ROUNDS`` rounds force a
        text-only commit (``tool_choice: "none"``). Every failed emission
        increments the parse-failure/retry counters and queues a correction
        message for the re-call.
        """
        used = 0
        while used < rounds_left:
            force_commit = rounds_left - used <= _COMMIT_ROUNDS
            response = self._create(messages, force_commit=force_commit)
            used += 1
            if response.stop_reason == "tool_use":
                self._dispatch_tool_round(
                    response, sim, messages, commit_warning=rounds_left - used <= _COMMIT_ROUNDS
                )
                continue
            expr = _extract_candidate(_response_text(response))
            messages.append({"role": "assistant", "content": response.content})
            if expr is None:
                # Budget burned on hidden thinking (no text at all): counted
                # like the plan's parse-failure path, but the attempt keeps
                # going — a fresh conclusion round may still land.
                self._fail_attempt(messages, "no expression found in the response")
                continue
            reason = _validate(expr)
            if reason is None:
                return expr, used
            self._fail_attempt(messages, f"{reason} (the reference parser reports no diagnostics)")
            return None, used
        return None, used

    def _dispatch_tool_round(
        self,
        response: Any,
        sim: BudgetedSimulator,
        messages: list[dict[str, Any]],
        commit_warning: bool = False,
    ) -> None:
        """Answer every tool_use block through the budgeted simulator.

        A denied call becomes a tool_result error WITHOUT delegating (the
        wrapper raises before touching the oracle); the JSON payload carries
        exactly the SimResult fields the system prompt promises. An unknown
        tool name gets an error result too, so the conversation stays
        API-valid. With unlimited budgets the DENIAL never fires, so
        ``commit_warning`` rides the last probing round instead: the same
        commit-now signal the denial text carries, as a user text block
        after the tool results (measured: the model commits on that signal;
        it never commits on its own, and the gateway ignores
        tool_choice "none").
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
            self._last_probe_expr = expr
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": _sim_result_json(result),
                }
            )
        if commit_warning:
            tool_results.append(
                {
                    "type": "text",
                    "text": "That was your last probe of the turn — commit your best "
                    "expression NOW: reply with ONLY the bare y = f(x) expression on "
                    "one line, no tool calls.",
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
