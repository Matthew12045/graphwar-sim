"""HybridAgent (M5.5): the LLM emits INTENT, CCF compiles the certified shot.

Division of labour (M5.5.1, non-negotiable): the LLM NEVER emits an
expression. It emits a waypoint PLAN (:mod:`agents.waypoints`); the certified
CCF solver compiles the plan into a curve, and that string — not the LLM's
text — is what reaches the parser. The LLM cannot violate the "bare
expression" rule because its output is JSON that never touches the parser;
that is the whole reason HybridAgent exists.

Turn structure:

1. the turn message carries the M5.5.2 shooter-relative geometry, a compact
   corridor summary (free vertical intervals per ``u`` — teammate bands
   PRE-applied per M5.5.7, so friendly-fire avoidance is by construction),
   the persona shape prior, the previous attempt's feedback, and the live
   simulate budget;
2. the LLM may probe through the budgeted simulate tool and finally returns
   ONLY the plan JSON. Schema-invalid plans get a correction message WITHOUT
   burning a solver attempt; ``schema_errors`` is its own counter (M5.5.3);
3. the plan's waypoints become CCF corridor TIGHTENINGS
   (:func:`graphwar_sim.ccf.solve_target` with ``waypoints=``); the
   relaxation ladder (M5.5.5) drops the lowest-priority waypoint on an
   infeasible solve (ties: most-binding = tightest tol — the documented
   approximation; LP duals are not on the solver's public surface), until
   feasible, ``UNREACHABLE`` (never the waypoints' fault — reported and
   stopped), or zero waypoints (reported EXPLICITLY per M5.5.5). When no
   curve certifies, the middle degradation rung fires the deterministic M2
   best-effort shot (:func:`graphwar_sim.solver.solve`) instead of the dud —
   certified first, best effort second, safe dud last;
4. the M5.5.6 feedback (outcome, binding + sigma, dropped/applied waypoint
   names, emitted length, remaining simulate budget — NEVER the expression)
   rides the next turn's message.

Isolation (M5.5.8): the LLM's raw text is never emitted — a valid expression
in ``rationale`` changes nothing (``rationale`` is truncated and logged
only). The wire format is parsed with :mod:`json` alone.

Persona overlays (M5.5.7): ``style`` prepends a shape prior to the turn
message; ``master_magician`` fills ``secondary_targets`` with all living
enemies (prompt-method + the M5.4 verifier carry the multi-hit game — no new
solver path, per the plan's M5.4.3 note).

Counters: ``schema_errors``, ``waypoints_applied`` / ``waypoints_dropped``
(the winning attempt's bookkeeping), the cert outcome counts
``ccf_certified`` / ``ccf_uncertified`` / ``ccf_infeasible`` /
``ccf_unreachable`` (``EMIT_OVERFLOW`` folds into ``ccf_infeasible``), and
``m2_fallbacks`` (turns the middle rung saved from the dud).

The M5.5.5 relaxation ladder is module-level (:func:`solve_plan_with_ladder`)
so the D4 battery (``eval/run_ccf_battery.py --waypoints``) drives the SAME
drop logic — the agent method is a thin delegate.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from graphwar_sim import Game
from graphwar_sim.ccf import (
    CCFCertificate,
    CCFOutcome,
    CCFSolution,
    CorridorWaypoint,
    solve_target,
)
from graphwar_sim.config import PLANE_GAME_LENGTH, PLANE_LENGTH
from graphwar_sim.corridor import (
    CCF_TEAMMATE_RADIUS,
    WORLD_RADIUS,
    ShooterFrame,
    map_y_bounds,
    shooter_frame,
)
from graphwar_sim.solver import solve as m2_solve

from .base import Observation
from .llm_agent import (
    _COMMIT_ROUNDS,
    SAFE_DUD,
    LLMAgent,
    _response_text,
)
from .personas import PERSONAS
from .simulate_budget import BudgetedSimulator
from .waypoints import PlanSchemaError, WaypointPlan, WaypointSpec, parse_plan

# Corridor-summary sample step (shooter-relative world units).
# # TUNABLE — not from source.
_SUMMARY_STEP: float = 2.0


def _hybrid_system_prompt() -> str:
    """The M5.5.2 system prompt (verbatim rules) + the M5.5.3 schema as JSON
    field rules (the model cannot match a schema it cannot see)."""
    return """\
You plan artillery shots. You do NOT write mathematical expressions.

A certified solver turns your plan into a curve. It proves the curve clears \
terrain and teammates, or it tells you exactly why it cannot. Your job is to \
choose targets and suggest a route shape. The solver's job is the math. \
Terrain is INDESTRUCTIBLE (destructible terrain is a planned mechanic, not \
yet live): the solver routes around rock, never through it — plan clearance \
accordingly.

Coordinates are SHOOTER-RELATIVE: your soldier is at (0, 0). Positive u is \
toward the target. All values you emit use this frame.

You will receive:
- enemy and teammate positions, terrain summary, map bounds
- a corridor summary: for each sample, the free vertical interval(s) \
(teammate bands are already removed)
- the outcome and binding constraint of your previous attempt, if any

Return ONLY a JSON object matching the schema. No prose, no markdown fences, \
no explanation outside the "rationale" field.

SCHEMA (exact field names):
  target_id: string; a LIVING enemy id from the turn message ("enemy_0", ...).
  secondary_targets: optional list of living enemy ids (multi-hit intent).
  branch_hint: "over" | "under" | "any".
  waypoints: list of 0 to 5 objects, strictly increasing u:
    u: number in (0, u_T) exclusive (endpoints are solver-owned)
    y: shooter-relative height inside the map band
    tol: number >= the hit radius (clamped up if smaller)
    priority: integer; 1 = dropped first when relaxing, higher survives longer
  style: persona id or null.
  rationale: <= 200 chars, logged only, never parsed.

WAYPOINTS ARE SUGGESTIONS. The solver treats them as soft corridor \
tightenings and will DROP the ones that make the problem infeasible, then \
tell you which it dropped. Do not try to force a curve by over-specifying: \
three loose waypoints beat eight tight ones. If you do not know where the \
curve should go, return an empty waypoint list and let the solver decide.

There is no cap on API rounds — take the probes you need, but spend \
them on genuine uncertainty, not on confirming a plan the solver \
already certified. Your per-turn simulate_tool allowance is stated in \
the turn message ("unlimited" means no cap)."""


# M5.5.7: the persona shape prior (prepended to the turn message; the persona
# sets defaults, it never gets its own output path).
_PERSONA_OVERLAYS: dict[str, str] = {
    "howitzer": "STYLE PRIOR (howitzer): seed a HIGH waypoint over the "
    'corridor\'s terrain ceiling and set branch_hint to "over".',
    "serpent": "STYLE PRIOR (serpent): seed waypoints at the corridor's gap "
    "centers; the solver walks the sigma ladder itself — loose waypoints at "
    "the gaps are your lever.",
    "bodyguard": "STYLE PRIOR (bodyguard): the teammate exclusion bands are "
    "ALREADY removed from the corridor you see — plan inside what remains and "
    "never route near a teammate.",
    "sniper": "STYLE PRIOR (sniper): waypoints are discouraged — prefer an "
    "empty waypoint list and let the solver take the direct route.",
    "professor": "STYLE PRIOR (professor): your rationale must name the "
    "binding constraint you expect (the corridor's tightest bound).",
    "master_magician": "STYLE PRIOR (master_magician): populate "
    "secondary_targets with ALL living enemies; one bump per target cluster, "
    "merging enemies that sit close together (tight clusters destroy the "
    "certificate).",
}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """The first balanced top-level ``{...}`` in ``text`` that parses as a
    JSON object (code fences tolerated; string contents respected).
    ``json`` only — the LLM's text is NEVER evaluated as an expression."""
    if not text:
        return None
    stripped = re.sub(r"```[a-zA-Z]*", "", text)
    try:
        parsed = json.loads(stripped)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for i, ch in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    parsed = json.loads(stripped[start : i + 1])
                except ValueError:
                    parsed = None
                if isinstance(parsed, dict):
                    return parsed
                start = None
    return None


@dataclass(frozen=True)
class _HybridSolve:
    """One plan -> solve -> relaxation outcome (M5.5.5)."""

    expression: str | None
    outcome: CCFOutcome
    feedback: str
    applied: int
    dropped: int
    # The winning attempt's full certificate (the battery records sigma /
    # slack / emitted length from it; the agent itself only reads outcome).
    cert: CCFCertificate


def _free_intervals_at(
    u: float,
    band: tuple[float, float],
    disks: tuple[tuple[float, float, float], ...],
) -> list[tuple[float, float]]:
    """The free vertical intervals at shooter-relative column ``u``: the map
    band minus every disk whose horizontal span covers the column.

    ``disks`` are ``(u_center, y_center, radius)`` in shooter-relative
    coords — terrain at its plane-pixel radius and teammates at the CCF
    blast-inflated radius (M5.5.7: the bands the solver enforces are the
    bands the model sees).
    """
    lo, hi = band
    intervals: list[tuple[float, float]] = [(lo, hi)]
    for du_c, dy_c, radius in disks:
        dx = u - du_c
        if abs(dx) >= radius:
            continue
        span = math.sqrt(radius * radius - dx * dx)
        block_lo = dy_c - span
        block_hi = dy_c + span
        nxt: list[tuple[float, float]] = []
        for a, b in intervals:
            if block_lo >= b or block_hi <= a:
                nxt.append((a, b))
                continue
            if block_lo > a:
                nxt.append((a, block_lo))
            if block_hi < b:
                nxt.append((block_hi, b))
        intervals = [(a, b) for a, b in nxt if b > a]
    return intervals


def _corridor_summary(
    mx: float,
    my: float,
    targets: tuple[tuple[float, float], ...],
    teammates: tuple[tuple[float, float], ...],
    circles: tuple[tuple[int, int, int], ...],
    inverted: bool,
) -> tuple[list[str], tuple[float, float]]:
    """The M5.5.2 corridor summary: per sample, the free vertical interval(s)
    in shooter-relative coords. Terrain circles transform exactly like
    :func:`graphwar_sim.corridor.shooter_frame`'s plane->world map; teammates
    subtract at the CCF blast-inflated radius — the solver's own envelope
    (M5.5.7: bands pre-applied BEFORE the model sees the corridor). Returns
    ``(lines, band)`` with ``band`` the shooter-relative playable band."""
    band_world = map_y_bounds()
    band = (band_world[0] - my, band_world[1] - my)
    scale = PLANE_GAME_LENGTH / PLANE_LENGTH
    disks: list[tuple[float, float, float]] = []
    for cx, cy, r in circles:
        px = PLANE_LENGTH - cx if inverted else cx
        wx = PLANE_GAME_LENGTH * (px - PLANE_LENGTH / 2.0) / PLANE_LENGTH
        wy = PLANE_GAME_LENGTH * (-cy + 225.0) / PLANE_LENGTH
        disks.append((wx - mx, wy - my, r * scale))
    for tx, ty in teammates:
        disks.append((tx - mx, ty - my, CCF_TEAMMATE_RADIUS))

    x1 = max((tx for tx, _ty in targets), default=mx)
    u_end = max(x1 - mx, _SUMMARY_STEP)
    lines: list[str] = []
    u = 0.0
    while u <= u_end + 1e-9:
        parts = [
            f"[{a:.1f}..{b:.1f}]"
            for a, b in _free_intervals_at(u, band, tuple(disks))
            if b - a >= 2.0 * WORLD_RADIUS
        ]
        lines.append(f"u={u:.1f}: " + (" ".join(parts) if parts else "BLOCKED"))
        u += _SUMMARY_STEP
    return lines, band


def _feedback_message(
    outcome: str,
    slack: float | None,
    binding: str | None,
    sigma: float | None,
    applied: list[str],
    dropped: list[str],
    emitted_length: int | None,
    char_limit: int,
    remaining: int | str,
) -> str:
    """The M5.5.6 feedback: names a SPECIFIC sigma/binding, lists dropped and
    applied waypoints, states the remaining budget — NEVER the expression
    (M5.5.6: the model cannot improve it and would hand-edit it)."""
    head = f"ATTEMPT — {outcome}"
    if slack is not None:
        head += f" (slack {slack:.2f})"
    lines = [
        head,
        f"  binding: {binding or 'none'}, sigma={sigma if sigma is not None else 'n/a'}",
        f"  dropped waypoints: [{', '.join(dropped) if dropped else ''}]",
        f"  applied waypoints: [{', '.join(applied) if applied else ''}]",
        f"  emitted length: {emitted_length if emitted_length is not None else 0} / {char_limit}",
        f"  simulate_tool calls remaining: {remaining}",
    ]
    if dropped:
        first_u = dropped[0].split(" ")[0].removeprefix("u=")
        lines.append(
            f"The dropped waypoint(s) made the problem infeasible. Raise the "
            f"waypoint at u={first_u} or widen its tol."
        )
    elif outcome in ("BASIS_INFEASIBLE", "UNREACHABLE"):
        lines.append(
            "Your plan was NOT the problem: the corridor stays empty/blocked "
            "even with zero waypoints. Pick a different target; do not just "
            "rewrite waypoints."
        )
    elif outcome == "EMIT_OVERFLOW":
        lines.append("The route shape needs more terms than the emission budget allows.")
    elif outcome == "UNCERTIFIED":
        lines.append("The curve exists but clearance is NOT proven — loosen the tightest waypoint.")
    return "\n".join(lines)


def _emit_event(
    on_event: Callable[[str, dict[str, Any]], None] | None,
    kind: str,
    payload: dict[str, Any],
) -> None:
    """Forward one feed event from the module-level solve path (a broken sink
    is swallowed — the feed is observational and must never crash a solve;
    same contract as ``LLMAgent._emit``)."""
    if on_event is None:
        return
    with suppress(Exception):
        on_event(kind, payload)


def solve_plan_with_ladder(
    fr: ShooterFrame,
    targets: Sequence[tuple[float, float]],
    plan: WaypointPlan,
    sim: BudgetedSimulator,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> _HybridSolve:
    """M5.5.4 + M5.5.5, agent-free: the plan's waypoints become CCF
    tightenings; on an infeasible solve, drop the lowest-priority waypoint
    (ties by the most-binding = tightest tol; the documented approximation —
    LP duals are not on the solver's public surface) and re-solve.
    Zero-waypoint infeasibility is reported EXPLICITLY (M5.5.5): the plan was
    never the problem.

    THE relaxation ladder — :class:`HybridAgent` delegates here and the D4
    battery (``eval/run_ccf_battery.py --waypoints``) calls this directly, so
    the drop logic exists exactly once.
    """
    target_index = int(plan.target_id.rsplit("_", 1)[1])
    ordered = sorted(plan.waypoints, key=lambda w: (w.priority, w.tol))
    current = list(ordered)
    dropped_total: list[str] = []
    sol: CCFSolution | None = None
    while True:
        wps = [CorridorWaypoint(u=w.u, y=w.y, tol=w.tol) for w in current]
        sol = solve_target(
            fr.mx,
            fr.my,
            targets,
            fr.teammates,
            fr.circles,
            fr.inverted,
            target_index,
            waypoints=wps,
        )
        outcome = sol.certificate.outcome
        if outcome in (CCFOutcome.CERTIFIED, CCFOutcome.UNCERTIFIED, CCFOutcome.UNREACHABLE):
            break  # a curve exists (or the corridor is simply unreachable)
        if not current:
            break  # zero-waypoint infeasible: report explicitly (M5.5.5)
        removed = current.pop(0)
        dropped_total.append(
            f"u={removed.u:.2f} (priority {removed.priority}) dropped: {outcome.value}"
        )
    assert sol is not None
    return _finalize_solve(sol, plan, sim, dropped_total, list(current), on_event)


def _finalize_solve(
    sol: CCFSolution,
    plan: WaypointPlan,
    sim: BudgetedSimulator,
    dropped_total: list[str],
    final_wps: list[WaypointSpec],
    on_event: Callable[[str, dict[str, Any]], None] | None,
) -> _HybridSolve:
    """The M5.5.6 feedback (never the expression) + the fired expression."""
    cert = sol.certificate
    candidates = sol.candidates
    expression = candidates[0].expression if candidates else None
    # branch_hint (M5.5.3) seeds the branch search: prefer a candidate
    # whose branch matches the hint; fall back to the solver's best.
    if plan.branch_hint in ("over", "under") and candidates:
        matching = [c for c in candidates if c.branch_kind == plan.branch_hint]
        if matching:
            expression = matching[0].expression
    dropped_all = dropped_total + list(cert.dropped_waypoints)
    applied = [f"u={w.u:.2f}" for w in final_wps]
    remaining: int | str = "unlimited" if sim.unlimited else sim.remaining
    feedback = _feedback_message(
        outcome=cert.outcome.value,
        slack=cert.slack_total,
        binding=cert.binding,
        sigma=cert.sigma,
        applied=applied,
        dropped=dropped_all,
        emitted_length=cert.emitted_length,
        char_limit=cert.char_limit,
        remaining=remaining,
    )
    _emit_event(
        on_event,
        "solve",
        {
            "outcome": cert.outcome.value,
            "applied": len(final_wps),
            "dropped": len(dropped_all),
            "sigma": cert.sigma,
        },
    )
    return _HybridSolve(
        expression=expression,
        outcome=cert.outcome,
        feedback=feedback,
        applied=len(final_wps),
        dropped=len(dropped_all),
        cert=cert,
    )


class HybridAgent(LLMAgent):
    """M5.5: plans through the LLM, shoots through the certified CCF solver.

    ``act`` returns the CCF-compiled expression for the LLM's waypoint plan
    (the M2 best-effort rung when no curve certifies, the safe dud only when
    no plan lands or that fails too). The LLM's raw text is never emitted
    and never reaches the parser (M5.5.8 isolation). Streaming/retry/cancel/
    event-sink/budgeted-simulate machinery is inherited from
    :class:`LLMAgent` (the Slice C cancel checks ride the same ``_create``
    path).
    """

    def __init__(
        self,
        model: str,
        max_attempts: int = 1,
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
        super().__init__(
            model=model,
            max_attempts=max_attempts,
            client=client,
            reasoning_effort=reasoning_effort,
            simulate_budget=simulate_budget,
            tool_rounds=tool_rounds,
            persona=None,  # the M5.4 verifier game does not apply here
            cancel_requested=cancel_requested,
            on_event=on_event,
        )
        self._style = persona
        self._system_prompt = _hybrid_system_prompt()
        self.name = f"hybrid:{model}" + (f"@{persona}" if persona is not None else "")
        # The previous attempt's M5.5.6 feedback (rides the next turn's
        # message; a fresh conversation has no other memory).
        self._last_feedback: str | None = None

    def act(self, game: Game, obs: Observation) -> str:
        """One hybrid turn: plan JSON -> validate -> CCF solve -> relax ->
        fire the solver's expression (the M2 best-effort rung when nothing
        certifies, the safe dud only when that fails too)."""
        sim = BudgetedSimulator(game, budget=self._simulate_budget)
        sim.new_turn()
        self._last_assistant_text = None
        self._round_counter = 0
        expr = SAFE_DUD
        feedback: str | None = None
        try:
            fr = shooter_frame(game)
            targets = sorted(fr.targets, key=lambda p: p[0])
            if targets:
                target_ids = [f"enemy_{i}" for i in range(len(targets))]
                messages: list[dict[str, Any]] = [
                    {"role": "user", "content": self._turn_message(fr, sim, obs.turn_index)}
                ]
                plan = self._collect_plan(messages, sim, target_ids, targets, fr.mx, fr.my)
                if plan is not None:
                    solve = self._solve_with_ladder(fr, targets, plan, sim)
                    feedback = solve.feedback
                    self._stats.waypoints_applied += solve.applied
                    self._stats.waypoints_dropped += solve.dropped
                    if solve.expression is not None:
                        expr = solve.expression
                    else:
                        # No certifiable curve (infeasible / unreachable /
                        # overflow): the middle degradation rung — the
                        # deterministic M2 best-effort shot — before the dud.
                        # A proved-blocked corridor still dud-commits when M2
                        # also passes, but any hittable board gets a real shot.
                        expr = self._m2_fallback(game)
                    self._bump_outcome_counter(solve.outcome)
        except Exception:  # noqa: BLE001 - the match must never crash on one turn
            self._stats.parse_failures += 1
            self._stats.retries += 1
        finally:
            self._stats.simulate_calls += sim.calls_used
            self._stats.simulate_denied += sim.denied_used
        self._last_feedback = feedback
        self._emit("commit", {"expr": expr})
        return expr

    # -- internals ------------------------------------------------------------

    def _bump_outcome_counter(self, outcome: CCFOutcome) -> None:
        if outcome is CCFOutcome.CERTIFIED:
            self._stats.ccf_certified += 1
        elif outcome is CCFOutcome.UNCERTIFIED:
            self._stats.ccf_uncertified += 1
        elif outcome is CCFOutcome.UNREACHABLE:
            self._stats.ccf_unreachable += 1
        else:
            self._stats.ccf_infeasible += 1

    def _m2_fallback(self, game: Game) -> str:
        """The middle degradation rung: no CCF curve certified, so fire the
        deterministic M2 solver's best-effort expression instead of the dud
        (the dud stays only for an M2 failure, which never happens in
        practice — :func:`graphwar_sim.solver.solve` always emits). The M2
        output is solver-generated, never LLM text, so M5.5.8 isolation is
        untouched; ``except Exception`` mirrors :meth:`act` (a
        ``TurnCancelled`` is a ``BaseException`` and still propagates).
        """
        self._stats.m2_fallbacks += 1
        try:
            return m2_solve(game).expression
        except Exception:  # noqa: BLE001 - the match must never crash on one turn
            return SAFE_DUD

    def _turn_message(self, fr: ShooterFrame, sim: BudgetedSimulator, turn_index: int) -> str:
        """The M5.5.2 turn message: shooter-relative geometry + corridor
        summary (teammate bands pre-applied) + persona overlay + previous
        feedback + live budget."""
        mx, my = fr.mx, fr.my
        band_world = map_y_bounds()
        targets = sorted(fr.targets, key=lambda p: p[0])
        lines: list[str] = [
            f"TURN {turn_index} — you are at (0, 0); u = x - {mx:.2f} "
            f"grows toward the targets; y_rel = y - {my:.2f}.",
            f"map band (shooter-relative y): [{band_world[0] - my:.2f}, {band_world[1] - my:.2f}]",
            "",
            "enemies (nearest first):",
        ]
        for i, (tx, ty) in enumerate(targets):
            lines.append(f"  enemy_{i}: u={tx - mx:.2f}, y_rel={ty - my:.2f}")
        lines.append("teammates (their exclusion bands are removed from the corridor):")
        if fr.teammates:
            for tx, ty in fr.teammates:
                lines.append(f"  (u={tx - mx:.2f}, y_rel={ty - my:.2f})")
        else:
            lines.append("  none")
        lines.append("")
        summary, _band = _corridor_summary(
            mx, my, tuple(targets), fr.teammates, fr.circles, fr.inverted
        )
        lines.append("corridor summary (free vertical intervals per u sample):")
        lines.extend(summary)
        overlay = _PERSONA_OVERLAYS.get(self._style or "")
        if overlay:
            lines.extend(["", overlay])
        if self._last_feedback:
            lines.extend(["", "previous attempt:", self._last_feedback])
        remaining: int | str = "unlimited" if sim.unlimited else sim.remaining
        lines.extend(["", f"simulate_tool calls remaining: {remaining}"])
        lines.append("Return ONLY the JSON plan object.")
        return "\n".join(lines)

    def _collect_plan(
        self,
        messages: list[dict[str, Any]],
        sim: BudgetedSimulator,
        target_ids: list[str],
        targets: list[tuple[float, float]],
        mx: float,
        my: float,
    ) -> WaypointPlan | None:
        """API rounds until a SCHEMA-VALID plan arrives (M5.5.3).

        Schema errors are corrected in-loop WITHOUT burning a solver attempt
        (``schema_errors`` is its own counter); the budgeted simulate tool
        stays available between rounds. ``None`` when an explicit round cap
        is spent without a valid plan (uncapped, the loop runs until a plan
        lands, the cancel fires, or an error aborts).
        """
        band_world = map_y_bounds()
        target_u_T = {tid: targets[i][0] - mx for i, tid in enumerate(target_ids)}
        rounds_left: float = float("inf") if self._tool_rounds is None else self._tool_rounds
        used = 0
        while used < rounds_left:
            self._check_cancel()
            force = rounds_left - used <= _COMMIT_ROUNDS
            self._round_counter += 1
            self._emit("round", {"n": self._round_counter, "force_commit": force})
            response = self._create(messages, force_commit=force)
            used += 1
            text = _response_text(response)
            if text:
                self._last_assistant_text = text
                self._emit("text", {"text": text})
            if response.stop_reason == "tool_use":
                self._dispatch_tool_round(response, sim, messages, commit_warning=False)
                continue
            raw = _extract_json_object(text)
            if raw is None:
                self._stats.schema_errors += 1
                messages.append(
                    {
                        "role": "user",
                        "content": "no JSON plan found — return ONLY the JSON object "
                        "matching the schema (no prose, no fences).",
                    }
                )
                continue
            try:
                plan = parse_plan(
                    raw,
                    target_u_T=target_u_T,
                    band_lo=band_world[0] - my,
                    band_hi=band_world[1] - my,
                    styles=set(PERSONAS),
                )
            except PlanSchemaError as exc:
                self._stats.schema_errors += 1
                messages.append(
                    {
                        "role": "user",
                        "content": f"schema error: {exc} — fix the plan and return "
                        "ONLY the JSON object.",
                    }
                )
                continue
            self._emit(
                "plan",
                {
                    "target": plan.target_id,
                    "n_waypoints": len(plan.waypoints),
                    "branch_hint": plan.branch_hint,
                },
            )
            return plan
        return None

    def _solve_with_ladder(
        self,
        fr: ShooterFrame,
        targets: list[tuple[float, float]],
        plan: WaypointPlan,
        sim: BudgetedSimulator,
    ) -> _HybridSolve:
        """The M5.5.5 relaxation ladder — the drop logic lives in
        :func:`solve_plan_with_ladder` (shared with the D4 battery)."""
        return solve_plan_with_ladder(fr, targets, plan, sim, on_event=self._on_event)


__all__ = ["HybridAgent", "solve_plan_with_ladder"]
