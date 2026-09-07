"""LLMAgent: an Anthropic-Messages-API agent with a budgeted simulate tool.

The agent plays one soldier per turn through a **fresh conversation** (no
cross-turn memory):

- the **system prompt** is the shared core prepend adapted to the engine's
  REAL frame — centered world, shooter facing right, the auto vertical
  offset of ``graphwar_sim.physics.process_function_range`` (the frame
  contract of :class:`~agents.base.Observation`) — and the REAL
  :class:`~agents.simulate_tool.SimResult` fields. An optional **persona**
  (``persona="<id>"``, one of :data:`agents.personas.PERSONAS`) appends its
  STYLE block after the byte-identical core and attaches the persona's
  machine verifier (M5.4.1): every turn's FIRED expression is verified and
  the verdict lands in ``rung_history`` (``PASS`` / ``CONSTRAINT_VIOLATION``
  / ``MAGICIAN_FULL`` / ``MAGICIAN_PARTIAL(n,m)``) plus the
  ``constraint_*`` stat counters. Verifier oracle calls are FREE (the pure
  :func:`agents.simulate_tool.simulate`, never the agent's budget).
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
  (``eval/runner.py``): parse_failures, retries, simulate_calls,
  simulate_denied, and guardrail_overrides (accumulated across turns).
  ``simulate_calls`` counts ALL oracle calls — probes AND the commit
  guardrail's check of the final candidate (M5.4; skipped when no probe ran).
- M5.4 commit guardrail: a validated candidate is oracle-checked through the
  same :class:`BudgetedSimulator` before it fires; when the turn's best
  probed expression scores STRICTLY better (:func:`_probe_score` — teammate
  hits rank worst, reach before distance, nearest_miss as the gradient) the
  probe fires instead and ``guardrail_overrides`` increments. Ties go to the
  commit.
- Cancellation (Slice C): the optional ``cancel_requested`` callback (a
  module-level :class:`threading.Event` in the UI server) is checked between
  API rounds and between stream deltas; a set callback raises
  :class:`TurnCancelled` — deliberately a ``BaseException`` so ``act``'s
  catch-all ``except Exception`` cannot convert a cancel into a dud shot.
  Eval passes no callback (the default never fires).
- Live activity feed (Slice B): the optional ``on_event`` callback (or a
  per-turn sink via :meth:`set_event_sink`) receives
  ``("round"|"text"|"tool_call"|"tool_result"|"guardrail"|"persona"|"commit"|"delta",
  payload)`` events as the turn progresses — the UI server routes them into
  its activity ring for the polling feed. Delta events forward the real SDK
  stream's ``content_block_delta`` text/thinking chunks (duck-typed, guarded;
  the gateway may expose no thinking at all, and the non-streaming test
  fakes get no deltas — both documented no-ops). A broken sink never crashes
  a turn.

Known limitation (documented, deliberately not engineered around): ``name``
is ``"llm:<model>"`` (or ``"llm:<model>@<persona>"``), so a **mirror match**
``llm:X`` vs ``llm:X`` collides in ``play_match``'s per-agent stats dict
(keyed by agent name); round-robin rosters cannot produce that pair, and
persona-suffixed rosters de-collide naturally (the suffix rides the name).

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
from collections.abc import Callable, Iterable
from contextlib import suppress
from typing import Any

from graphwar_sim import Game, PolishNotationFunction, config
from graphwar_sim.parser import MalformedFunction

from .base import AgentStats, Observation
from .personas import PERSONAS
from .personas.verifiers import VERIFIERS, VerdictKind, VerifierContext, VerifierVerdict
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


class TurnCancelled(BaseException):
    """The turn's cancel callback fired (Slice C).

    ``BaseException`` ON PURPOSE: :meth:`LLMAgent.act`'s catch-all
    ``except Exception`` must not convert a cancel into a safe-dud shot — a
    cancelled turn unwinds the whole emission loop and leaves the game
    untouched (the caller — the UI server's ``agent_turn`` — turns it into a
    409 "aborted"). Eval passes no cancel callback, so this never fires
    there.
    """


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

# Muzzle-wall warning scan radii (world units) for the turn message (M5.4
# Fix 3b): terrain within this box just RIGHT of the muzzle kills ascending
# launches (the seed-21 live diagnosis: every early probe died on the rock
# ~0.9 units right of the muzzle before the model learned to descend).
# # TUNABLE — not from source.
_MUZZLE_WALL_DX: float = 2.5
_MUZZLE_WALL_DY: float = 2.0

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


def _probe_score(result: SimResult) -> tuple[int, int, float]:
    """The commit guardrail's comparison key — LOWER IS BETTER.

    Rationale (M5.4): the first element ranks hits — a clean enemy hit is the
    only acceptable kill, a TEAMMATE hit is a critical failure that must never
    be preferred no matter how small its nearest_miss; the second ranks reach
    — a shot that arrived at the enemy's x (``"hit"``/``"passed"``) dominates
    one that died en route, because height is correctable but reach is the
    hard part on terrain-walled boards; ``nearest_miss`` breaks ties as the
    model's gradient. Unparseable probes carry all-None telemetry and land on
    ``(1, 1, inf)`` — worst. Pure and deterministic: no game state, no RNG.
    """
    if result.hit_teammate:
        hit_rank = 2
    elif result.hit_enemy:
        hit_rank = 0
    else:
        hit_rank = 1
    reach_rank = 0 if result.stop_reason in ("hit", "passed") else 1
    nearest = result.nearest_miss if result.nearest_miss is not None else float("inf")
    return (hit_rank, reach_rank, nearest)


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


def _muzzle_wall_warning(obs: Observation) -> str | None:
    """Deterministic warning when terrain sits just RIGHT of the muzzle —
    ascending launches die on it, so say so up front (M5.4 Fix 3b; the live
    diagnosis burned 3+ probes per turn re-learning this). Reports the
    NEAREST qualifying block. ``None`` when none qualifies."""
    mx, my = obs.shooter
    best: tuple[float, float, float] | None = None
    for bx, by in obs.terrain_blocks:
        dx = bx - mx
        if not 0.0 < dx <= _MUZZLE_WALL_DX:
            continue
        if abs(by - my) > _MUZZLE_WALL_DY:
            continue
        if best is None or dx < best[0]:
            best = (dx, bx, by)
    if best is None:
        return None
    _, bx, by = best
    return (
        f"WARNING: terrain wall at (x~{bx:.1f}, y~{by:.1f}) just right of your "
        "muzzle — ascending launches die on it; launch DESCENDING."
    )


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
    ]
    warning = _muzzle_wall_warning(obs)
    if warning is not None:
        lines.extend(["", warning])
    lines.extend(
        [
            "",
            f"simulate calls remaining: {remaining}",
        ]
    )
    return "\n".join(lines)


def _commit_echo(expr: str | None, result: SimResult | None) -> str:
    """The telemetry echo appended to the commit nudge (M5.4 Fix 2): repeat
    the last probe's identity + telemetry so the model can commit its best
    probe or fix exactly that failure instead of re-deriving from memory (the
    live diagnosis: commits ignored what earlier rounds had measured). One
    short block — the gateway's wall headroom. Empty when no probe ran (the
    echo cannot invent telemetry)."""
    if expr is None or result is None:
        return ""
    nearest = round(result.nearest_miss, 3) if result.nearest_miss is not None else None
    stop_x = round(result.stopped_at_x, 2) if result.stopped_at_x is not None else None
    return (
        f"Your last probe '{expr}' -> nearest_miss={nearest}, "
        f"miss_direction={result.miss_direction}, stopped_at_x={stop_x}, "
        f"stop_reason={result.stop_reason} — commit THAT expression if it was "
        "your best; otherwise fix exactly its failure (off_map: bring the "
        "curve's peak under y=14.6; terrain at the muzzle: launch descending; "
        "short: increase reach)."
    )


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
        persona: str | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if persona is not None and persona not in PERSONAS:
            raise ValueError(f"unknown persona: {persona!r} (known: {', '.join(sorted(PERSONAS))})")
        self._model = model
        self._persona = PERSONAS[persona] if persona is not None else None
        self.name = f"llm:{model}" + (f"@{persona}" if persona is not None else "")
        # The core prompt stays byte-identical; a persona appends its STYLE
        # block after it (plan A2).
        self._system_prompt = (
            _SYSTEM_PROMPT
            if self._persona is None
            else f"{_SYSTEM_PROMPT}\n\n{self._persona.style_text}"
        )
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
        # Slice C cancellation: a callback polled between API rounds (the UI
        # server passes its module-level threading.Event.is_set; eval passes
        # nothing). Checked between retries in _create and between rounds in
        # _run_attempt; mid-stream checks ride the Slice B delta iteration.
        self._cancel_requested = cancel_requested
        # Slice B live feed: the per-turn event sink (the server injects one
        # per agent_turn via set_event_sink — agents are shared instances).
        self._on_event = on_event
        # Turn-level API round counter for the ("round", ...) feed events.
        self._round_counter: int = 0
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
        # The last probe's full telemetry (the commit nudge's echo, M5.4).
        self._last_probe_result: SimResult | None = None
        # The turn's BEST probed expression by _probe_score, tracked across
        # the WHOLE turn (attempts share the conversation). The commit
        # guardrail (M5.4) fires it when the committed candidate oracle-checks
        # strictly worse. Ties keep the earlier probe.
        self._best_probe: tuple[str, SimResult] | None = None
        # The turn's last assistant text block (the professor verifier's
        # structural check reads it; the emission loop otherwise discards it).
        self._last_assistant_text: str | None = None
        # Per-turn persona verdicts (rung strings). Only maintained with a
        # persona attached, so persona-less agents keep the runner's
        # rung_counts output unchanged (M5.4.2 / plan A4).
        if self._persona is not None:
            self.rung_history: list[str] = []

    def act(self, game: Game, obs: Observation) -> str:
        """Emit this turn's expression (centered world frame).

        Opens a fresh conversation (system prompt + one per-turn user
        message), runs up to ``max_attempts`` attempts (see
        :meth:`_run_attempt`), and returns the safe dud on exhaustion. The
        :class:`BudgetedSimulator` is created and ``new_turn()``-ed per the
        wrapper's contract (the first call closes the empty construction
        turn — harmless; counters stay correct); its ledger counters are
        folded into :meth:`stats` after every turn.

        With a persona attached, the FIRED expression (after the commit
        guardrail's possible override, and on every degraded path too) is
        run through the persona's machine verifier (M5.4.1): the verdict
        appends to ``rung_history`` and updates the ``constraint_*``
        counters. A verifier bug degrades to "no verdict" — it must never
        crash the match.
        """
        sim = BudgetedSimulator(game, budget=self._simulate_budget)
        sim.new_turn()
        self._last_probe_expr = None
        self._last_probe_result = None
        self._best_probe = None
        self._last_assistant_text = None
        self._round_counter = 0
        expr = SAFE_DUD
        try:
            budget_text: int | str = "unlimited" if sim.unlimited else sim.remaining
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": _turn_message(obs, budget_text)}
            ]
            rounds_left = self._tool_rounds
            candidate: str | None = None
            for _attempt in range(self._max_attempts):
                found, rounds_used = self._run_attempt(messages, sim, rounds_left)
                rounds_left -= rounds_used
                if found is not None:
                    candidate = found
                    break
                if rounds_left <= 0:
                    break
            if candidate is not None:
                expr = self._guard_commit(candidate, sim)
            else:
                # Round cap hit without a text commit: fire the model's LAST
                # probed expression — it chose it through real simulate
                # feedback, which beats the safe dud. (The gateway ignores
                # tool_choice "none", so a forced text commit cannot be
                # relied on; this fallback guarantees the probing still pays
                # off.)
                expr = self._last_probe_expr or SAFE_DUD
        except Exception:  # noqa: BLE001 - the match must never crash on one turn
            # Unrecoverable after the API retries (e.g. the gateway killed
            # every regeneration): degrade this turn to the safe dud,
            # accounted like the runner's defensive malformed-emission branch.
            self._stats.parse_failures += 1
            self._stats.retries += 1
            # The guardrail's best-probe preference applies here too: the
            # best probed expression dominates the merely-last one (and both
            # beat the safe dud).
            if self._best_probe is not None:
                expr = self._best_probe[0]
            else:
                expr = self._last_probe_expr or SAFE_DUD
        finally:
            self._stats.simulate_calls += sim.calls_used
            self._stats.simulate_denied += sim.denied_used
        self._persona_verdict(expr, game, obs)
        self._emit("commit", {"expr": expr})
        return expr

    def stats(self) -> AgentStats:
        return self._stats

    def set_event_sink(self, sink: Callable[[str, dict[str, Any]], None] | None) -> None:
        """Attach (or detach) the per-turn activity sink (Slice B).

        The UI server calls this on its shared agent instances right before
        ``act()`` — the game lock makes the injection single-writer safe.
        Passing ``None`` detaches.
        """
        self._on_event = sink

    def _persona_verdict(self, expr: str, game: Game, obs: Observation) -> VerifierVerdict | None:
        """Run the attached persona's machine verifier on the FIRED
        expression (M5.4.1). Appends the verdict's rung string to
        ``rung_history`` (so ``eval.runner._peek_solver_rung`` and the UI
        ``[rung: ...]`` display work unchanged) and updates the
        ``constraint_*`` counters. Returns ``None`` when no persona is
        attached or the verifier itself failed (a verifier bug degrades to
        "no verdict" — it must never crash the match). The verifier's
        oracle calls are FREE: they go through the pure
        ``agents.simulate_tool.simulate`` inside the verifier, never the
        agent's ``BudgetedSimulator``."""
        if self._persona is None:
            return None
        verifier = VERIFIERS.get(self._persona.verifier)
        if verifier is None:  # pragma: no cover - manifest test pins the registry
            return None
        try:
            verdict = verifier(
                expr,
                game,
                obs,
                VerifierContext(assistant_text=self._last_assistant_text or ""),
            )
        except Exception:  # noqa: BLE001 - a verifier bug must never crash a match
            return None
        self._stats.constraint_checks += 1
        if verdict.kind is VerdictKind.VIOLATION:
            self._stats.constraint_violations += 1
        elif verdict.kind is VerdictKind.MAGICIAN_PARTIAL:
            self._stats.magician_partials += 1
        self.rung_history.append(verdict.rung)
        self._emit(
            "persona",
            {
                "verdict": verdict.rung,
                "reason": verdict.reason,
                "constraint": self._persona.constraint_type,
            },
        )
        return verdict

    # -- internals ------------------------------------------------------------

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        """Forward one feed event to the sink (Slice B). A broken sink is
        swallowed — the feed is observational and must never crash a turn."""
        if self._on_event is None:
            return
        with suppress(Exception):
            self._on_event(kind, payload)

    def _check_cancel(self) -> None:
        """Raise :class:`TurnCancelled` when the cancel callback is set (a
        no-op without one — the eval default)."""
        if self._cancel_requested is not None and self._cancel_requested():
            raise TurnCancelled()

    def _forward_stream_deltas(self, stream: Any) -> None:
        """Best-effort: iterate a REAL SDK stream and forward its
        ``content_block_delta`` text/thinking chunks as ``("delta", ...)``
        feed events (Slice B).

        Duck-typed on :class:`collections.abc.Iterable`: the real SDK stream
        is iterable (raw events); the test fakes only expose
        ``get_final_message`` and are not — a documented no-op there. The
        gateway may hide thinking entirely (no thinking deltas) — equally a
        no-op. The cancel check runs per delta batch: mid-round cancellation
        at seconds granularity (Slice C).
        """
        if not isinstance(stream, Iterable):
            return
        try:
            for event in stream:
                self._check_cancel()
                if getattr(event, "type", "") != "content_block_delta":
                    continue
                delta = getattr(event, "delta", None)
                delta_type = getattr(delta, "type", "")
                if delta_type == "text_delta":
                    self._emit("delta", {"kind": "text", "text": getattr(delta, "text", "")})
                elif delta_type == "thinking_delta":
                    self._emit(
                        "delta", {"kind": "thinking", "text": getattr(delta, "thinking", "")}
                    )
        except TurnCancelled:
            raise
        except Exception:  # noqa: BLE001 - a stream-surface mismatch is a no-op
            return

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
            "system": self._system_prompt,
            "tools": [_SIMULATE_TOOL],
            "messages": messages,
        }
        if force_commit:
            kwargs["tool_choice"] = {"type": "none"}
        if self._reasoning_effort is not None:
            kwargs["extra_body"] = {"reasoning_effort": self._reasoning_effort}
        stream_factory = getattr(self._client.messages, "stream", None)
        for retry in range(_MAX_API_RETRIES + 1):
            self._check_cancel()  # between (re)tries — a cancel beats a retry
            try:
                if callable(stream_factory):
                    with stream_factory(**kwargs) as stream:
                        self._forward_stream_deltas(stream)
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
            self._check_cancel()  # between rounds (and before the first)
            force_commit = rounds_left - used <= _COMMIT_ROUNDS
            self._round_counter += 1
            self._emit("round", {"n": self._round_counter, "force_commit": force_commit})
            response = self._create(messages, force_commit=force_commit)
            used += 1
            # Retain the turn's last assistant text (the professor verifier's
            # structural input; thinking-only rounds leave it untouched).
            text = _response_text(response)
            if text:
                self._last_assistant_text = text
                self._emit("text", {"text": text})
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

    def _guard_commit(self, candidate: str, sim: BudgetedSimulator) -> str:
        """The commit guardrail (M5.4 Fix 1): oracle-check the committed
        candidate through the SAME :class:`BudgetedSimulator` — one extra
        oracle call, the same physics ``Game.fire`` is about to run — and fire
        the turn's best probed expression instead when the commit is STRICTLY
        worse by :func:`_probe_score`. Ties go to the commit (the model's own
        word wins close calls). Skipped entirely when no probe ran: with no
        best probe the check cannot change the outcome (the guardrail cannot
        invent one). The check can only be denied on a finite simulate budget
        (:class:`SimulateBudgetExhausted`; ``budget=None`` never denies) — the
        documented degradation fires the commit as-is.
        """
        if self._best_probe is None:
            return candidate
        try:
            commit_result = sim.simulate(candidate)
        except SimulateBudgetExhausted:
            return candidate
        if _probe_score(self._best_probe[1]) < _probe_score(commit_result):
            self._stats.guardrail_overrides += 1
            self._emit("guardrail", {"fired": True, "expr": self._best_probe[0]})
            return self._best_probe[0]
        self._emit("guardrail", {"fired": False, "expr": candidate})
        return candidate

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
            self._emit("tool_call", {"expr": expr})
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
                self._emit("tool_result", {"denied": True, "expr": expr})
                continue
            self._last_probe_expr = expr
            self._last_probe_result = result
            # Track the turn's best probe across ALL attempts (parseable or
            # not — an unparseable probe simply scores worst). Strictly
            # better only: the earlier probe wins ties.
            score = _probe_score(result)
            if self._best_probe is None or score < _probe_score(self._best_probe[1]):
                self._best_probe = (expr, result)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": _sim_result_json(result),
                }
            )
            self._emit(
                "tool_result",
                {
                    "denied": False,
                    "expr": expr,
                    "hit_enemy": result.hit_enemy,
                    "hit_teammate": result.hit_teammate,
                    "nearest_miss": (
                        round(result.nearest_miss, 3) if result.nearest_miss is not None else None
                    ),
                    "miss_direction": result.miss_direction,
                    "stopped_at_x": (
                        round(result.stopped_at_x, 2) if result.stopped_at_x is not None else None
                    ),
                    "stop_reason": result.stop_reason,
                },
            )
        if commit_warning:
            text = (
                "That was your last probe of the turn — commit your best "
                "expression NOW: reply with ONLY the bare y = f(x) expression on "
                "one line, no tool calls."
            )
            echo = _commit_echo(self._last_probe_expr, self._last_probe_result)
            if echo:
                text = f"{text} {echo}"
            tool_results.append({"type": "text", "text": text})
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


__all__ = ["LLMAgent", "TurnCancelled"]
