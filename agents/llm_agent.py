"""LLMAgent: an Anthropic-Messages-API agent. SINGLE-SHOT per turn.

The agent plays one soldier per turn through a **fresh conversation** (no
cross-turn memory) with **exactly one API call and no tools**: whatever the
model outputs first is validated and fired — there are no probe rounds, no
re-attempts, no commit guardrail. Accuracy comes from the turn's context,
not from iteration:

- the **system prompt** is the shared core prepend adapted to the engine's
  REAL frame — centered world, shooter facing right, the auto vertical
  offset of ``graphwar_sim.physics.process_function_range`` (the frame
  contract of :class:`~agents.base.Observation`). An optional **persona**
  (``persona="<id>"``, one of :data:`agents.personas.PERSONAS`) appends its
  STYLE block after the byte-identical core and attaches the persona's
  machine verifier (M5.4.1): every turn's FIRED expression is verified and
  the verdict lands in ``rung_history`` (``PASS`` / ``CONSTRAINT_VIOLATION``
  / ``MAGICIAN_FULL`` / ``MAGICIAN_PARTIAL(n,m)``) plus the
  ``constraint_*`` stat counters. Verifier oracle calls are FREE (the pure
  :func:`agents.simulate_tool.simulate`, never charged anywhere).
- the **per-turn user message** serializes the observation verbatim (the
  frame every other agent reads — coordinates are NOT re-derived, shifted,
  or re-centered) plus deterministic CLEAR LANE intervals: the corridor's
  free vertical ranges per world-x sample (terrain- and crater-aware via
  :func:`graphwar_sim.corridor._column_free`, teammate disks pre-removed),
  so the model picks a provably open lane instead of guessing one.
- the single answer is validated with the real parser; a malformed emission
  counts ``parse_failures`` / ``retries`` (the same counters RandomAgent
  exposes) and fires the repo's safe dud ``"0*x"``
  (``eval/runner.py:227``, ``agents/baselines.py:57``).
- ``stats()`` maps the counters the runner merges into the match stats
  (``eval/runner.py``): parse_failures, retries, simulate_calls (always 0 —
  no oracle calls), simulate_denied (always 0), and guardrail_overrides
  (always 0, retained for the runner's merge shape).
- Cancellation (Slice C): the optional ``cancel_requested`` callback (a
  module-level :class:`threading.Event` in the UI server) is checked around
  the single call and between stream deltas; a set callback raises
  :class:`TurnCancelled` — deliberately a ``BaseException`` so ``act``'s
  catch-all ``except Exception`` cannot convert a cancel into a dud shot.
  Eval passes no callback (the default never fires).
- Live activity feed (Slice B): the optional ``on_event`` callback (or a
  per-turn sink via :meth:`set_event_sink`) receives
  ``("round"|"text"|"persona"|"commit"|"delta", payload)`` events as the
  turn progresses — the UI server routes them into its activity ring for
  the polling feed. Delta events forward the real SDK stream's
  ``content_block_delta`` text/thinking chunks (duck-typed, guarded; the
  gateway may expose no thinking at all, and the non-streaming test fakes
  get no deltas — both documented no-ops). A broken sink never crashes
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

# The repo-wide last-resort emission: a guaranteed-parseable shot that goes
# nowhere (eval/runner.py:227, agents/baselines.py:57,92). # TUNABLE —
# not from source (repo convention).
SAFE_DUD: str = "0*x"

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

# Clear-lane sample step (world x units) for the turn message's deterministic
# corridor hint. # TUNABLE — not from source.
_CLEAR_LANE_STEP: float = 2.0

# The shared core prepend, adapted to the engine's REAL frame and grammar
# (the sketch's shooter-relative frame does NOT match the engine — see the
# module docstring). The SYNTAX section quotes the parser's tokenizer
# whitelist, not an embellishment of it (docs/GROUND_TRUTH.md §3.2). The
# THINK paragraph targets the measured failure mode: the model happily burns
# its whole output budget on closed-form ballistic derivations instead of
# reading the CLEAR LANES and committing a simple curve that holds one
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
survive all the way to the enemy's x. Terrain is DESTRUCTIBLE: every shot \
ends in a blast that carves a crater (~0.8 world units wide) out of the rock \
at its impact point, and craters persist — simulate already sees them. Do not \
route through unblasted rock; a muzzle wall can be dug open with one \
sacrificial shot, then shot through next turn. Check the terrain immediately \
to the right of your own muzzle first: if it is a wall, launch DESCENDING or \
blast through it.

SYNTAX (the parser's exact tokenizer whitelist — nothing else exists): \
numbers, ( ) x + - * / ^, functions sqrt log (base 10) ln abs sin sen cos \
tan tg, constants e pi. Variable x ONLY (no y, no y'). Implicit \
multiplication works (2x, x(x+1), 2sin(x)). '-' is always unary (a-b parses \
as a+(-b)). Unknown characters are silently DROPPED. Max 2000 chars; \
deeper than 64 nested terms is rejected (your shot is replaced by a safe \
dud).

CLEAR LANES (world coords, the same frame your expression runs in): for \
x samples from your muzzle toward the enemies, the open vertical \
intervals — route inside them all the way to your target's x. Teammate \
disks are already removed from these intervals, and they are \
terrain-and-crater aware. "BLOCKED" means no open interval there: go \
around (over/under) or dig through with a sacrificial shot.

THINK, then emit ONE expression. You get exactly one shot per turn: there \
is no simulate tool and no second attempt — read the map, the positions, \
and the CLEAR LANES, pick a lane that stays open to your target, and \
commit. Prefer simple curves (lines, gentle quadratics, one kink at most); \
exotic functions miss more often than they thread. Do NOT derive \
trajectories analytically beyond that rough shape.

OUTPUT: emit ONLY the bare expression on one line (no "y =", no prose, no \
code fence).\
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


def _clear_lane_lines(game: Game, obs: Observation) -> list[str]:
    """Deterministic corridor hint (the single-shot accuracy strategy): free
    vertical intervals per world-x sample from the muzzle toward the
    farthest enemy, via :func:`graphwar_sim.corridor._column_free` — hence
    terrain- AND crater-aware, with teammate disks pre-removed at the M5.1
    soldier radius. Pure geometry, no oracle calls, no scipy."""
    from graphwar_sim.corridor import _TEAMMATE_RADIUS, _column_free

    circles = tuple(getattr(game, "circles", ()))
    carves = tuple(getattr(game, "carves", ()))
    mx, _my = obs.shooter
    ends = [p[0] for p in obs.enemy_soldiers]
    x_end = max(ends) if ends else mx
    teammates = [(p[0], p[1], _TEAMMATE_RADIUS) for p in obs.own_soldiers]
    lines: list[str] = []
    x = mx
    while x <= x_end + 1e-9:
        free = _column_free(x, circles, teammates, carves)
        parts = " ".join(f"[{a:.1f}..{b:.1f}]" for a, b in free)
        lines.append(f"x={x:.1f}: " + (parts if parts else "BLOCKED"))
        x += _CLEAR_LANE_STEP
    return lines


def _turn_message(game: Game, obs: Observation) -> str:
    """The per-turn user message: the observation verbatim + the
    deterministic CLEAR LANE intervals (same world frame as the emission)."""

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
        "CLEAR LANES (open vertical intervals per x — hold one to your target):",
        *_clear_lane_lines(game, obs),
    ]
    warning = _muzzle_wall_warning(obs)
    if warning is not None:
        lines.extend(["", warning])
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
    """The LLM shot generator: one API call, no tools, first output fires."""

    name: str

    def __init__(
        self,
        model: str,
        client: Any | None = None,
        reasoning_effort: str | None = None,
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
        # Slice C cancellation: a callback polled around the single call
        # (the UI server passes its module-level threading.Event.is_set;
        # eval passes nothing). Mid-stream checks ride the Slice B delta
        # iteration.
        self._cancel_requested = cancel_requested
        # Slice B live feed: the per-turn event sink (the server injects one
        # per agent_turn via set_event_sink — agents are shared instances).
        self._on_event = on_event
        # Turn-level API round counter for the ("round", ...) feed events
        # (always 1 now — retained so the feed shape is unchanged).
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
        # The turn's last assistant text block (the professor verifier's
        # structural check reads it; the emission otherwise discards it).
        self._last_assistant_text: str | None = None
        # Per-turn persona verdicts (rung strings). Only maintained with a
        # persona attached, so persona-less agents keep the runner's
        # rung_counts output unchanged (M5.4.2 / plan A4).
        if self._persona is not None:
            self.rung_history: list[str] = []

    def act(self, game: Game, obs: Observation) -> str:
        """Emit this turn's expression (centered world frame).

        Opens a fresh conversation (system prompt + one per-turn user
        message), makes EXACTLY ONE API call with no tools, and fires
        whatever the model outputs first (validated; the safe dud on empty
        or malformed output). No probes, no re-attempts, no guardrail.

        With a persona attached, the FIRED expression is run through the
        persona's machine verifier (M5.4.1): the verdict appends to
        ``rung_history`` and updates the ``constraint_*`` counters. A
        verifier bug degrades to "no verdict" — it must never crash the
        match.
        """
        self._last_assistant_text = None
        self._round_counter = 0
        expr = SAFE_DUD
        try:
            messages: list[dict[str, Any]] = [{"role": "user", "content": _turn_message(game, obs)}]
            self._check_cancel()
            self._round_counter = 1
            self._emit("round", {"n": 1})
            response = self._create(messages)
            text = _response_text(response)
            if text:
                self._last_assistant_text = text
                self._emit("text", {"text": text})
            candidate = _extract_candidate(text)
            if candidate is None:
                self._stats.parse_failures += 1
                self._stats.retries += 1
            elif _validate(candidate) is None:
                expr = candidate
            else:
                self._stats.parse_failures += 1
                self._stats.retries += 1
        except Exception:  # noqa: BLE001 - the match must never crash on one turn
            # Unrecoverable after the API retries (e.g. the gateway killed
            # the generation): degrade this turn to the safe dud, accounted
            # like the runner's defensive malformed-emission branch.
            self._stats.parse_failures += 1
            self._stats.retries += 1
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
        ``agents.simulate_tool.simulate`` inside the verifier."""
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

    def _create(self, messages: list[dict[str, Any]]) -> Any:
        """The turn's single API round-trip: streaming when the client
        supports it. No tools are offered, so the model MUST answer in text.

        The 9arm gateway serves a reasoning model that thinks for MINUTES
        before acting and forwards nothing until a whole block finishes, so
        Cloudflare's ~120s proxy read limit kills long generations mid-call
        (524 non-streaming, truncated stream mid-flight). Each retry
        regenerates from scratch — a fresh generation may land under the
        limit. Clients that expose ``messages.stream`` get the streaming
        path; test fakes expose only ``messages.create`` and take the
        non-streaming fallback — the response surface is identical
        (``.stop_reason`` + ``.content`` blocks).
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "system": self._system_prompt,
            "messages": messages,
        }
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


__all__ = ["LLMAgent", "TurnCancelled"]
