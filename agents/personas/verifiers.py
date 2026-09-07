"""Machine verifiers for the six personas (M5.4.1 / M5.4.2).

Constraints are VERIFIED, not prompted: every persona constraint gets a
machine-checkable verifier that runs on the EMITTED EXPRESSION after the
turn's emission is chosen. The LLM's self-report is not evidence.

Contract:

- Pure functions ``(expr, game, obs, ctx) -> VerifierVerdict``. ``ctx``
  carries the data the agent captured this turn (the professor's structural
  check reads the turn's last assistant text).
- All oracle use goes through the FREE :func:`agents.simulate_tool.simulate`
  — never the agent's :class:`~agents.simulate_budget.BudgetedSimulator`
  (referee context: verifier calls are free and must not touch the agent's
  budget or counters). ``simulate_tool`` stays UNMODIFIED (bridge M5.4.2).
- Geometric checks (howitzer/serpent/bodyguard) evaluate the emitted ``f``
  AS WRITTEN — the persona's own stated rule, before the game's one auto
  vertical offset. The verifier is a persona-discipline check ("did the bot
  play its own game?"), not a flight-safety check; the leaderboard keeps
  "lost the round" (outcome) separate from this verdict (M5.4.1).
- A violation NEVER crashes the round-robin: it is logged as
  ``CONSTRAINT_VIOLATION`` with the persona id (the runner merges the
  counters; see ``AgentStats.constraint_violations``).

Weaknesses, documented on purpose:

- sniper: the lower-rung candidates are closed-form fits; a missed fit can
  hide a real simpler shot (false negatives only — an oracle-verified hit
  is proof, so there are no false positives).
- serpent: crossings are detected by sampling, at the sampler's resolution.
- professor: the weakest verifier — a STRUCTURAL check that the turn's
  assistant text states a constraint per living enemy and per teammate
  (keyword/shape check on recorded text, not math).
- master_magician: enemy-hit count is inferred as
  ``num_hits - (1 if hit_teammate else 0)`` — exact for enemies (the
  trajectory's x is strictly increasing, physics.py:216-228, so each enemy
  is reached at most once, and the physics dedupes repeat hits), and it
  assumes at most one TEAMMATE is struck per shot (per the plan's formula;
  the persona never sacrifices teammates).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from graphwar_sim import (
    PLANE_GAME_LENGTH,
    PLANE_LENGTH,
    TEAM2,
    Game,
    PolishNotationFunction,
)
from graphwar_sim.corridor import WORLD_RADIUS
from graphwar_sim.emission import format_literal

from ..base import Observation, world_coords
from ..simulate_tool import simulate

# Sampling step (world units) for curve-shape scans (howitzer peak, serpent
# crossings). The corridor spans <= ~50 world units, so ~500 evaluations per
# check — cheap next to one physics integration. # TUNABLE — not from source.
_SCAN_STEP: float = 0.1

# The howitzer's own rule: peak clears corridor terrain by this many world
# units. # TUNABLE — from the persona's stated constraint (kept +8).
_HOWITZER_CLEARANCE: float = 8.0

# The bodyguard's safety pad on top of the hit radius (world units).
# # TUNABLE — from the persona's stated constraint (kept +3).
_BODYGUARD_PAD: float = 3.0


class VerdictKind(StrEnum):
    """The four verifier verdict kinds (plan A3)."""

    PASS = "PASS"
    VIOLATION = "CONSTRAINT_VIOLATION"
    MAGICIAN_FULL = "MAGICIAN_FULL"
    MAGICIAN_PARTIAL = "MAGICIAN_PARTIAL"


@dataclass(frozen=True)
class VerifierContext:
    """Per-turn data the agent captured, for the verifiers that need it.

    ``assistant_text`` is the turn's last assistant text block (the
    professor's structural check runs on it; every other verifier ignores
    it).
    """

    assistant_text: str = ""


@dataclass(frozen=True)
class VerifierVerdict:
    """One persona-verifier outcome (M5.4.1 / M5.4.2).

    ``rung`` is the string the agent appends to ``rung_history`` — the
    ``MAGICIAN_FULL`` / ``MAGICIAN_PARTIAL(n,m)`` / ``CONSTRAINT_VIOLATION``
    vocabulary the runner's ``[rung: ...]`` display already renders (the
    plain ``PASS`` is persona-neutral and collides with no solver rung).
    """

    kind: VerdictKind
    reason: str = ""
    # Magician bookkeeping: enemy hits achieved vs living enemies at turn
    # start (None for the non-magician verdicts).
    hits: int | None = None
    living: int | None = None

    @property
    def rung(self) -> str:
        if self.kind is VerdictKind.MAGICIAN_PARTIAL:
            return f"MAGICIAN_PARTIAL({self.hits},{self.living})"
        return self.kind.value


VerifierFn = Callable[[str, Game, Observation, VerifierContext], VerifierVerdict]


# --- shared helpers -----------------------------------------------------------


def _parse(expr: str) -> PolishNotationFunction | None:
    """The parsed emission, or ``None`` when it does not parse."""
    try:
        return PolishNotationFunction(expr)
    except Exception:  # noqa: BLE001 - the reference raises no message
        return None


def _eval(f: PolishNotationFunction, x: float) -> float | None:
    """``f(x)``, or ``None`` when the evaluation fails (a pole / domain
    error — Java returns NaN there; the ported parser raises)."""
    try:
        value = f.evaluate(x)
    except Exception:  # noqa: BLE001 - mirrors simulate_tool's tolerance
        return None
    if value != value:  # NaN
        return None
    return value


def _living_teammates(obs: Observation) -> tuple[tuple[float, float], ...]:
    """Living teammates OTHER than the shooter (``own_soldiers`` includes
    the shooter, whose band is trivially violated by the auto-offset)."""
    shooter = obs.shooter
    return tuple(p for p in obs.own_soldiers if p != shooter)


def _corridor_x_range(obs: Observation) -> tuple[float, float]:
    """The corridor the shot must fly: from the muzzle x to the farthest
    living enemy x."""
    x0 = obs.shooter[0]
    x1 = max((ex for ex, _ in obs.enemy_soldiers), default=x0)
    return x0, max(x1, x0 + _SCAN_STEP)


def _terrain_circles_world(game: Game, mirrored: bool) -> tuple[tuple[float, float, float], ...]:
    """Terrain circles in the shooter's world frame: ``(cx, cy, r)``."""
    scale = PLANE_GAME_LENGTH / PLANE_LENGTH
    return tuple((*world_coords(cx, cy, mirrored), r * scale) for cx, cy, r in game.circles)


def _column_blocked(x_c: float, circles: tuple[tuple[float, float, float], ...]) -> bool:
    """True when a terrain circle spans column ``x_c`` — the plan's
    "no terrain circle within hit radius horizontally": the horizontal gap
    from the crossing to the circle's rim must exceed the hit radius."""
    return any(abs(cx - x_c) <= r + WORLD_RADIUS for cx, _cy, r in circles)


def _coords_stated(text: str, point: tuple[float, float]) -> bool:
    """Weak structural check (professor): both coordinates of ``point``
    appear in the text, in the turn message's 1-decimal form (or its
    integer form), not as a substring of a longer number."""
    for value in point:
        forms = {f"{value:.1f}"}
        if value == int(value):
            forms.add(str(int(value)))
        if not any(_number_boundary(form, text) for form in forms):
            return False
    return True


def _number_boundary(literal: str, text: str) -> bool:
    """True when ``literal`` occurs in ``text`` unglued from other digits
    (not inside ``116.9`` when looking for ``16.9``, not inside ``16.95``)."""
    pattern = re.compile(rf"(?<![\d.]){re.escape(literal)}(?!\d)")
    return pattern.search(text) is not None


def _unparseable(persona: str) -> VerifierVerdict:
    return VerifierVerdict(
        VerdictKind.VIOLATION, f"{persona}: the emitted expression does not parse"
    )


# --- sniper -------------------------------------------------------------------

# Regexes over the whitespace-stripped emission. Best-effort classification
# (documented weakness): an unclassifiable emission is PASS — the verifier
# never alleges a violation it cannot prove.
_NUM: str = r"[+-]?\d+(?:\.\d+)?"
_LINE: re.Pattern[str] = re.compile(rf"^(?:{_NUM}\*)?x$")
_ARC: re.Pattern[str] = re.compile(rf"^{_NUM}\*?x\*?\(x-{_NUM}\)$")
_TILTED: re.Pattern[str] = re.compile(rf"^{_NUM}\*?x\*?\(x-{_NUM}\)\+{_NUM}\*?x$")
_SIGMOID: str = "/(1+e^"

# Ladder indices: 0 = m*x, 1 = a*x*(x-L), 2 = tilted arc, 3 = sigmoid.
_SNIPER_RUNG_NAMES: tuple[str, ...] = ("m*x", "a*x*(x-L)", "a*x*(x-L)+m*x", "sigmoid")


def _sniper_rung_index(expr: str) -> int | None:
    stripped = re.sub(r"\s+", "", expr)
    if _SIGMOID in stripped:
        return 3
    if _TILTED.match(stripped):
        return 2
    if _ARC.match(stripped):
        return 1
    if _LINE.match(stripped):
        return 0
    return None


def _sniper_candidates(rung: int, obs: Observation) -> list[str]:
    """Closed-form fitted candidates for one ladder rung, one or two per
    living enemy (the constants that put the shifted curve through the
    enemy, modulo the hit radius). Denominator-guards skip degenerate
    geometry."""
    sx, sy = obs.shooter
    candidates: list[str] = []
    for ex, ey in obs.enemy_soldiers:
        if rung == 0:
            # Line through muzzle and enemy: the shifted m*x hits (ex, ey).
            if abs(ex - sx) < 1e-9:
                continue
            m = (ey - sy) / (ex - sx)
            candidates.append(f"{format_literal(m)}*x")
        elif rung == 1:
            # a*x*(x-L) through the enemy: a*(ex*(ex-L) - sx*(sx-L)) = ey - sy.
            for L in (ex, ex - sx):
                denom = ex * (ex - L) - sx * (sx - L)
                if abs(denom) < 1e-9:
                    continue
                a = (ey - sy) / denom
                candidates.append(f"{format_literal(a)}*x*(x-{format_literal(L)})")
        elif rung == 2:
            # Tilted arc: (a, L) chosen independently of the enemy hit
            # equation, then m solved from it (non-degenerate tilt).
            for a in (-0.01, -0.02):
                # L solves a*(ex*(ex-L) - sx*(sx-L)) = ey - sy for fixed a:
                # L*(sx - ex) = (ey-sy)/a - (ex^2 - sx^2)
                if abs(sx - ex) < 1e-9:
                    continue
                L = ((ey - sy) / a - (ex * ex - sx * sx)) / (sx - ex)
                D = ex * (ex - L) - sx * (sx - L)
                if abs(ex - sx) < 1e-9:
                    continue
                m = (ey - sy - a * D) / (ex - sx)
                candidates.append(
                    f"{format_literal(a)}*x*(x-{format_literal(L)})+{format_literal(m)}*x"
                )
    return candidates


def verify_sniper(expr: str, game: Game, obs: Observation, ctx: VerifierContext) -> VerifierVerdict:
    """Simplest rung that works (M5.4.1): VIOLATION when a STRICTLY simpler
    ladder rung also achieves ``hit_enemy`` on this board. An unclassifiable
    or bottom-rung emission passes (nothing simpler exists to prove)."""
    del ctx  # the sniper's verdict is the oracle's, not the text's
    if _parse(expr) is None:
        return _unparseable("sniper")
    rung = _sniper_rung_index(expr)
    if rung is None or rung == 0:
        return VerifierVerdict(VerdictKind.PASS)
    for simpler in range(rung):
        for candidate in _sniper_candidates(simpler, obs):
            result = simulate(game, candidate)
            if result.hit_enemy:
                return VerifierVerdict(
                    VerdictKind.VIOLATION,
                    f"sniper: simpler rung '{_SNIPER_RUNG_NAMES[simpler]}' also hits ({candidate})",
                )
    return VerifierVerdict(VerdictKind.PASS)


# --- howitzer -----------------------------------------------------------------


def verify_howitzer(
    expr: str, game: Game, obs: Observation, ctx: VerifierContext
) -> VerifierVerdict:
    """Max of the emitted f over the corridor x-range must clear the
    corridor's terrain top by :data:`_HOWITZER_CLEARANCE` world units
    (M5.4.1). The curve is sampled on a fixed grid (documented resolution)."""
    del ctx  # structural verifiers ignore the captured text
    f = _parse(expr)
    if f is None:
        return _unparseable("howitzer")
    x0, x1 = _corridor_x_range(obs)
    mirrored = obs.team_id == TEAM2
    circles = [
        (cx, cy, r)
        for cx, cy, r in _terrain_circles_world(game, mirrored)
        if cx + r >= x0 and cx - r <= x1
    ]
    terrain_top = max((cy + r for _cx, cy, r in circles), default=float("-inf"))
    peak = float("-inf")
    x = x0
    while x <= x1:
        value = _eval(f, x)
        if value is not None:
            peak = max(peak, value)
        x += _SCAN_STEP
    if peak == float("-inf"):
        return VerifierVerdict(VerdictKind.VIOLATION, "howitzer: the curve is nowhere evaluable")
    if peak >= terrain_top + _HOWITZER_CLEARANCE:
        return VerifierVerdict(VerdictKind.PASS)
    return VerifierVerdict(
        VerdictKind.VIOLATION,
        f"howitzer: curve peak {peak:.2f} does not clear corridor terrain top "
        f"{terrain_top:.2f} by {_HOWITZER_CLEARANCE:.0f}",
    )


# --- serpent ------------------------------------------------------------------


def verify_serpent(
    expr: str, game: Game, obs: Observation, ctx: VerifierContext
) -> VerifierVerdict:
    """Every sign change of f over the corridor must lie in a terrain-free
    column — no terrain circle within hit radius horizontally (M5.4.1).
    Crossings are detected by sampling (documented resolution); a curve
    with no sign change passes (the dud case is scored by the round, not
    by this verifier)."""
    del ctx  # structural verifiers ignore the captured text
    f = _parse(expr)
    if f is None:
        return _unparseable("serpent")
    x0, x1 = _corridor_x_range(obs)
    mirrored = obs.team_id == TEAM2
    circles = _terrain_circles_world(game, mirrored)

    def sign(x: float) -> int:
        value = _eval(f, x)
        if value is None or value == 0.0:
            return 0
        return 1 if value > 0 else -1

    x = x0
    prev = sign(x)
    while x < x1:
        next_x = min(x + _SCAN_STEP, x1)
        next_sign = sign(next_x)
        if prev != 0 and next_sign != 0 and prev != next_sign and _column_blocked(x, circles):
            return VerifierVerdict(
                VerdictKind.VIOLATION,
                f"serpent: zero crossing near x={x:.2f} lies inside a terrain column",
            )
        x = next_x
        if next_sign != 0:
            prev = next_sign
    return VerifierVerdict(VerdictKind.PASS)


# --- bodyguard ----------------------------------------------------------------


def verify_bodyguard(
    expr: str, game: Game, obs: Observation, ctx: VerifierContext
) -> VerifierVerdict:
    """For each living teammate at (d_k, h_k): ``|f(d_k) - h_k|`` must
    exceed the hit radius plus :data:`_BODYGUARD_PAD` (M5.4.1). Any
    violation is a hard fail. Teammates are the living allies OTHER than
    the shooter (the auto-offset routes the curve through the shooter by
    construction)."""
    del game, ctx  # geometry-only check
    f = _parse(expr)
    if f is None:
        return _unparseable("bodyguard")
    limit = WORLD_RADIUS + _BODYGUARD_PAD
    for dk, hk in _living_teammates(obs):
        value = _eval(f, dk)
        if value is None:
            return VerifierVerdict(
                VerdictKind.VIOLATION,
                f"bodyguard: curve not evaluable at teammate x={dk:.2f}",
            )
        if not abs(value - hk) > limit:
            return VerifierVerdict(
                VerdictKind.VIOLATION,
                f"bodyguard: teammate at ({dk:.2f}, {hk:.2f}) inside the forbidden band "
                f"(|f(d)-h| = {abs(value - hk):.2f} <= {limit:.2f})",
            )
    return VerifierVerdict(VerdictKind.PASS)


# --- professor ----------------------------------------------------------------


def verify_professor(
    expr: str, game: Game, obs: Observation, ctx: VerifierContext
) -> VerifierVerdict:
    """STRUCTURAL, the weakest verifier (documented): the turn's captured
    assistant text must state a constraint per living enemy and per living
    teammate — detected as both numeric coordinates of the soldier appearing
    in the text. It checks that the constraint system was STATED, not that
    the algebra is right."""
    del game, expr  # the check is on the recorded text and the observation only
    text = ctx.assistant_text
    if not text.strip():
        return VerifierVerdict(
            VerdictKind.VIOLATION, "professor: no assistant text captured this turn"
        )
    for dx, hy in obs.enemy_soldiers:
        if not _coords_stated(text, (dx, hy)):
            return VerifierVerdict(
                VerdictKind.VIOLATION,
                f"professor: no stated constraint for the enemy at ({dx:.1f}, {hy:.1f})",
            )
    for dx, hy in _living_teammates(obs):
        if not _coords_stated(text, (dx, hy)):
            return VerifierVerdict(
                VerdictKind.VIOLATION,
                f"professor: no stated constraint for the teammate at ({dx:.1f}, {hy:.1f})",
            )
    return VerifierVerdict(VerdictKind.PASS)


# --- master magician (M5.4.2) ---------------------------------------------------


def verify_master_magician(
    expr: str, game: Game, obs: Observation, ctx: VerifierContext
) -> VerifierVerdict:
    """A single expression whose trajectory registers a hit on EVERY living
    enemy (M5.4.2), via the FREE oracle. Enemy hits are inferred from
    ``num_hits`` (enemy + teammate combined): the x-monotone trajectory
    reaches each enemy at most once (physics.py:216-228) and the physics
    dedupes repeat hits, so ``num_hits - (1 if hit_teammate else 0)`` is the
    enemy count under the plan's formula (assumes at most one teammate
    struck). FULL when every living enemy is hit; PARTIAL — a first-class
    outcome, not a failure — when some are; VIOLATION when none."""
    del ctx  # the magician's verdict is the oracle's, not the text's
    result = simulate(game, expr)
    if not result.parseable:
        return _unparseable("master_magician")
    living = len(obs.enemy_soldiers)
    enemy_hits = result.num_hits - (1 if result.hit_teammate else 0)
    if living == 0:  # pragma: no cover - a turn always has a living enemy
        return VerifierVerdict(VerdictKind.PASS)
    if enemy_hits >= living:
        return VerifierVerdict(VerdictKind.MAGICIAN_FULL, hits=enemy_hits, living=living)
    if enemy_hits > 0:
        return VerifierVerdict(
            VerdictKind.MAGICIAN_PARTIAL,
            f"master_magician: {enemy_hits} of {living} living enemies hit",
            hits=enemy_hits,
            living=living,
        )
    return VerifierVerdict(
        VerdictKind.VIOLATION,
        "master_magician: the shot hit no living enemy"
        + (" (a teammate was struck)" if result.hit_teammate else ""),
        hits=0,
        living=living,
    )


# --- registry -------------------------------------------------------------------

VERIFIERS: dict[str, VerifierFn] = {
    "verify_sniper": verify_sniper,
    "verify_howitzer": verify_howitzer,
    "verify_serpent": verify_serpent,
    "verify_bodyguard": verify_bodyguard,
    "verify_professor": verify_professor,
    "verify_master_magician": verify_master_magician,
}

__all__ = [
    "VERIFIERS",
    "VerifierContext",
    "VerifierFn",
    "VerifierVerdict",
    "VerdictKind",
]
