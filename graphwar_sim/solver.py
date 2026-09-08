"""Deterministic, no-LLM Graphwar solver (Phase 2 / Milestone M2).

Given a :class:`~graphwar_sim.state.Game` on the current turn, :func:`solve`
returns a :class:`SolverResult` whose ``expression`` is a **legal Graphwar
``y = f(x)`` string** (NORMAL_FUNC mode) that the simulator can fire. The
solver is a pure heuristic for *generating* candidate expressions; the
**simulator is the oracle** — every candidate is self-verified through
:func:`graphwar_sim.physics.process_function_range` before it is returned, so a
returned expression is guaranteed to be parseable and to have the reported
hit/no-friendly-fire outcome.

Coordinate frame
----------------
The reference evaluates ``f`` in **centered world** coordinates (x in
``[-25, 25]``, y up) and, for a TEAM2 shooter, mirrors the plane about the
vertical centre (Function.java:188-191, 247-250). We work in a
**shooter-facing** world frame: the muzzle at negative x, all enemies at
positive x, and the curve always integrates to increasing x (physics.py:198-202).
The TEAM2 mirror is folded into the frame transform below, so the fit is
identical for both sides.

Auto vertical offset
--------------------
The game adds ``c = -f(x0) + y0`` so the integrated curve passes through the
muzzle (Function.java:206). The *translated* curve is ``g(x) = f(x) + c`` with
``g(x0) = y0``. Baking this into a waypoint ``(tx, ty)`` gives the linear
constraint ``f(tx) - f(mx) = ty - my`` — no ``f(0) = 0`` term (that would force
the curve through map centre; see IMPLEMENTATION_PLAN.md, Phase 2).

Basis and certification
-----------------------
The primary basis is a sum of Gaussians ``f = Σ w_j e^{-b (x - c_j)²}``, linear
in the weights ``w``. Hitting waypoints is a plain least-squares solve
(``numpy.linalg.lstsq``); no ``cvxpy``/QP is required for this minimal slice.
The emitted curve is integrated as a piecewise-linear path, so the gap between
the true curve and the traced path is bounded by ``M · du² / 8`` where
``du`` is the integration step and ``M`` bounds ``|f''|``. For the Gaussian
basis ``|f''_j| ≤ (2b / √e) · |w_j|`` (the max of ``|z²-1| e^{-z²}`` is
``1/√e``), so we certify with the slightly looser frame-independent
``M = 2b · Σ|w_j|`` and ``du = STEP_SIZE``. On a non-finite bound the caller
would halve ``du``; here every rung yields a finite bound.

Degradation ladder (rung recorded on the result; a Phase 4 metric)
------------------------------------------------------------------
The multi-target Gaussian rungs lead (the plan's intended primary basis, and
the only rung that can multi-kill). Per 5.2.md §1 the M5.2 CCF rung sits
between the closed-form rungs and the fixed-grid rung, which is on death row —
it exists only until CCF beats it on the same seeds. The ``ccf`` rung's
certified-scipy LPs run lazily, only when reached (the ladder is a generator):

1. ``per_target_gaussian`` — one Gaussian centre per enemy, several widths.
2. ``parabola`` / ``line`` — closed-form single-target curves (auto-offset
   baked in).
3. ``ccf`` — certified corridor fit (5.2.md §1; ``graphwar_sim/ccf.py``).
4. ``fixed_grid_gaussian`` — a fixed uniform grid of centres (the plan's named
   primary; it underperforms on the seeded maps — the documented divergence,
   and on death row behind CCF).
5. ``wall_climb`` — flat-launch obstacle clearers (climb kinks + delayed
   sigmoids aimed at the climb point past the muzzle wall).
6. ``dud`` — a deliberate safe dud that hits nothing (never crashes).

Acceptance rule (one-shot max-kill)
-----------------------------------
``_verify`` returns the enemy-hit **count** (the engine is multi-kill:
soldier hits never stop the shot, physics.py:276-290 — only terrain/OOB/NaN
break integration; the hit test is strict ``< SOLDIER_RADIUS=7``, deduped).
``solve`` walks the ladder in order, rejects any teammate grazer, returns
immediately on a perfect shot (``kills == n_targets`` — for ``n_targets == 1``
this reproduces first-hit behavior exactly), otherwise keeps the first
candidate with strictly more kills than any earlier one and, among verified
equal-kill shots, the one with the larger pre-first-kill rock clearance
(strict ``>`` on both: ladder order is the deterministic tie-break within
exact ties; kills always dominate) and fires it after the ladder exhausts.
Only zero-hit-anywhere falls to the rotated duds (ranked: a lucky hit, then
an off-map / rock-sparing exit, then rotation order). The corridor pre-check
and dig path are unchanged: a corridor-blocked board cannot kill-all until
dug open. ``SolverResult.kills`` carries the fired shot's enemy kills
(``notes`` appends ``kills=k/n``); ``rung`` still records the producing
rung.

Emission note: numeric literals are emitted as plain decimals (``_num``),
never scientific notation — the parser's ``-``→``+-`` rewrite would corrupt
``1e-06`` into ``1e+-06`` (parsed as ``1*e - 6``).

Clean-room note: the reference is GPL-licensed; this module cites
``ref/graphwar/`` behaviour but does not copy its code. No ``eval``/``exec`` —
expressions are built from literals and parsed by the faithful
:class:`PolishNotationFunction`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import config
from .emission import format_literal as _num
from .emission import gauss_term as _gauss_term
from .parser import MalformedFunction, PolishNotationFunction
from .physics import ShotResult, Soldier, process_function_range
from .state import Game

if TYPE_CHECKING:
    from .ccf import CCFCertificate

# --- Tunable solver parameters (NOT from the reference source) --------------
# Gaussian width (curvature) of the primary basis. # TUNABLE — chosen by the
# seeded-map experiments in the M2 session (see docs/OPEN_QUESTIONS.md).
_B_WIDTHS: tuple[float, ...] = (0.35, 0.2, 0.6)
# Number of centres for the fixed-grid rung. # TUNABLE.
_GRID_N_CENTERS: int = 12
# Curvature sweep for the terrain-aware arc rung: the free quadratic
# coefficient ``A`` is swept over this grid per enemy. # TUNABLE — the range
# and step were chosen by the seeded-map experiments in the M2 session: a
# narrow fine grid clears terrain on the most dud maps, and widening it
# recovers nothing while costing more (see docs/OPEN_QUESTIONS.md).
_ARC_A_MIN: float = -0.1
_ARC_A_MAX: float = 0.1
_ARC_A_STEP: float = 0.005
# Integration step used for the certification bound (Constants.java:88).
_DU: float = config.STEP_SIZE

# Rung labels (M5.1 taxonomy — replaces the single ``dud`` bucket).
RUNG_PER_TARGET = "per_target_gaussian"
RUNG_FIXED_GRID = "fixed_grid_gaussian"
RUNG_LINE = "line"
RUNG_PARABOLA = "parabola"
# Obstacle-clearing rung (this plan): flat-launch curves that thread the
# muzzle wall, then maneuver past mid-field rock — placed AFTER every cheaper
# kill-seeking rung (kills-first: it only wins turns the old ladder scored
# 0–partial on), BEFORE duds/dig.
RUNG_WALL = "wall_climb"
# 5.2.md §1 ladder placement: CCF sits between line and fixed_grid_gaussian.
RUNG_CCF = "ccf"
RUNG_ARC = "arc"
# M5.1: the corridor pre-check proved no monotone-x trajectory can reach any
# enemy (PASS_UNREACHABLE) vs. a reachable target the ladder failed to convert
# (SOLVER_FAILED). The old ``dud`` bucket collapsed these two events.
RUNG_PASS_UNREACHABLE = "PASS_UNREACHABLE"
RUNG_SOLVER_FAILED = "SOLVER_FAILED"
# Destructible terrain: no clean path exists NOW, but a shot can blast rock
# open for a later turn. A ``sacrificial`` turn digs (carves) rather than
# passing — it is a real shot (a MISS when it hits nobody), not a pass.
RUNG_SACRIFICIAL = "sacrificial"


@dataclass
class SolverResult:
    """Outcome of :func:`solve`.

    - ``expression``: a legal Graphwar ``y = f(x)`` string (always parseable).
    - ``rung``: which degradation rung produced it (a Phase 4 metric).
    - ``bound``: the certification bound ``M · du² / 8`` (finite for every rung).
    - ``notes``: free-form diagnostics (e.g. self-verification hit summary).
    - ``cert``: the CCF certificate when the CCF rung produced the expression,
      else ``None``.
    - ``kills``: enemy kills for the fired shot (``len(targets)`` denominator
      in ``notes``); ``0`` on clean duds, ``None`` only when never set.
    """

    expression: str
    rung: str
    bound: float
    notes: str = ""
    cert: CCFCertificate | None = None
    kills: int | None = None


@dataclass
class _Candidate:
    """An internal candidate: an emitted expression plus its certification data.

    ``cert`` is the CCF certificate (when this candidate came from the CCF
    rung), else ``None``.
    """

    expression: str
    rung: str
    m_bound: float  # bound on |f''| (world units); 0 for an exact line
    cert: CCFCertificate | None = None


# --- Frame transforms (mirror physics.py exactly) ---------------------------


def _plane_to_world(px: float, py: float, mirrored: bool) -> tuple[float, float]:
    """Plane/pixel -> centered world, applying the TEAM2 mirror if ``mirrored``.

    Mirrors the muzzle transform in ``process_function_range``
    (physics.py:154-160).
    """
    if mirrored:
        px = config.PLANE_LENGTH - px
    x = config.PLANE_GAME_LENGTH * (px - config.PLANE_LENGTH / 2.0) / config.PLANE_LENGTH
    y = config.PLANE_GAME_LENGTH * (-py + config.PLANE_HEIGHT / 2.0) / config.PLANE_LENGTH
    return x, y


@dataclass
class _Frame:
    """The shooter-facing world frame for one shot."""

    mx: float
    my: float
    targets: list[tuple[float, float]]  # enemy (x, y), all x > mx
    shooter: Soldier
    enemies: list[Soldier]
    inverted: bool
    circles: tuple[tuple[int, int, int], ...]  # terrain circles in plane px
    teammates: list[tuple[float, float]]  # alive same-side soldiers excl. shooter
    carves: tuple[tuple[int, int, int], ...] = ()  # crater disks, TRUE plane coords


def _build_frame(game: Game) -> _Frame:
    """Assemble the shooter-facing world frame (muzzle + live enemies).

    Teammates (alive same-side soldiers excluding the shooter) are gathered
    through the SAME ``_plane_to_world`` transform as enemies, in the
    shooter-facing world frame, alongside the raw terrain circles in plane
    pixel coords (both feed the CCF rung, mirroring
    ``corridor.shooter_frame``).
    """
    team = game.state.current_team()
    shooter = team.current_soldier()
    inverted = team.team == config.TEAM2
    mx, my = _plane_to_world(shooter.x, shooter.y, inverted)
    targets: list[tuple[float, float]] = []
    enemies: list[Soldier] = []
    teammates: list[tuple[float, float]] = []
    for t in game.state.teams:
        if t.team == team.team:
            for s in t.soldiers:
                if not s.alive or s is shooter:
                    continue
                teammates.append(_plane_to_world(s.x, s.y, inverted))
            continue  # own side (teammates) — never a target
        for s in t.soldiers:
            if not s.alive:
                continue
            ex, ey = _plane_to_world(s.x, s.y, inverted)
            targets.append((ex, ey))
            enemies.append(s)
    return _Frame(
        mx=mx,
        my=my,
        targets=targets,
        shooter=shooter,
        enemies=enemies,
        inverted=inverted,
        circles=tuple(getattr(game, "circles", ())),
        teammates=teammates,
        carves=tuple(getattr(game, "carves", ())),
    )


# --- Emission ---------------------------------------------------------------
# The plain-decimal formatter and the Gaussian term emitter moved to
# :mod:`graphwar_sim.emission` (M5.2: the CCF rung shares them; the module is
# the single source of truth, validated by tests/test_emission.py). The names
# stay importable here for the M2 round-trip tests.


def _gauss_expression(weights: np.ndarray, centers: np.ndarray, b: float) -> str:
    """Emit ``Σ w_j e^{-b (x - c_j)²}`` as a Graphwar expression string."""
    return "+".join(
        _gauss_term(float(w), float(c), b) for w, c in zip(weights, centers, strict=True)
    )


def _gauss_m_bound(weights: np.ndarray, b: float) -> float:
    """Bound on ``|f''|`` for the Gaussian basis: ``2b · Σ|w_j|`` (see module
    docstring for the derivation)."""
    return 2.0 * b * float(np.sum(np.abs(weights)))


# --- Candidate generators ----------------------------------------------------


def _per_target_gaussians(frame: _Frame, b: float) -> _Candidate | None:
    """One Gaussian centre per enemy; ``lstsq`` on the offset-baked waypoints."""
    targets = frame.targets
    if not targets:
        return None
    centers = np.array([tx for tx, _ in targets], dtype=float)
    k = centers.size
    a = np.zeros((len(targets), k))
    rhs = np.zeros(len(targets))
    for i, (tx, ty) in enumerate(targets):
        for j in range(k):
            a[i, j] = math.exp(-b * (tx - centers[j]) ** 2) - math.exp(
                -b * (frame.mx - centers[j]) ** 2
            )
        rhs[i] = ty - frame.my
    w, *_ = np.linalg.lstsq(a, rhs, rcond=None)
    if not np.all(np.isfinite(w)):
        return None
    return _Candidate(
        expression=_gauss_expression(w, centers, b),
        rung=RUNG_PER_TARGET,
        m_bound=_gauss_m_bound(w, b),
    )


def _fixed_grid_gaussians(frame: _Frame, b: float) -> _Candidate | None:
    """A fixed uniform grid of centres spanning the enemy region (the plan's
    named primary basis; fewer centres than targets, so a true least-squares
    fit rather than interpolation)."""
    targets = frame.targets
    if not targets:
        return None
    lo = frame.mx
    hi = max(tx for tx, _ in targets)
    n = _GRID_N_CENTERS
    centers = np.array([hi]) if hi - lo < 1e-9 else np.linspace(lo, hi, n)
    k = centers.size
    a = np.zeros((len(targets), k))
    rhs = np.zeros(len(targets))
    for i, (tx, ty) in enumerate(targets):
        for j in range(k):
            a[i, j] = math.exp(-b * (tx - centers[j]) ** 2) - math.exp(
                -b * (frame.mx - centers[j]) ** 2
            )
        rhs[i] = ty - frame.my
    w, *_ = np.linalg.lstsq(a, rhs, rcond=None)
    if not np.all(np.isfinite(w)):
        return None
    return _Candidate(
        expression=_gauss_expression(w, centers, b),
        rung=RUNG_FIXED_GRID,
        m_bound=_gauss_m_bound(w, b),
    )


def _line_candidate(frame: _Frame) -> _Candidate | None:
    """Closed-form straight line through the muzzle and the nearest enemy."""
    if not frame.targets:
        return None
    tx, ty = min(frame.targets, key=lambda p: p[0])
    if abs(tx - frame.mx) < 1e-9:
        return None
    m = (ty - frame.my) / (tx - frame.mx)
    return _Candidate(expression=f"({_num(m)})*x", rung=RUNG_LINE, m_bound=0.0)


def _parabola_candidate(frame: _Frame) -> _Candidate | None:
    """Closed-form single arc ``a·x·(x - tx)`` through muzzle and nearest enemy.

    With the auto-offset baked in, ``g(x) = a·x·(x-tx) + c`` satisfies
    ``g(mx) = my`` and ``g(tx) = ty`` for ``a = (my - ty) / (mx·(mx - tx))``.
    """
    if not frame.targets:
        return None
    tx, ty = min(frame.targets, key=lambda p: p[0])
    denom = frame.mx * (frame.mx - tx)
    if abs(denom) < 1e-9:
        return None
    a = (frame.my - ty) / denom
    return _Candidate(
        expression=f"({_num(a)})*x*(x-({_num(tx)}))", rung=RUNG_PARABOLA, m_bound=2.0 * abs(a)
    )


def _arc_candidates(frame: _Frame) -> list[_Candidate]:
    """Terrain-aware arcs: sweep the free quadratic curvature per enemy.

    A single-valued ``y = f(x)`` can only reach a target along a monotone-in-x
    path, but it may *arc over or under* terrain. The closed-form parabola rung
    is one specific member of the 2-parameter family of quadratics through the
    muzzle and an enemy; here we sweep the remaining degree of freedom.

    For an enemy ``(tx, ty)`` the auto-offset constraint
    ``f(tx) - f(mx) = ty - my`` pins the linear coefficient once the quadratic
    coefficient ``A`` is chosen: writing ``f(x) = A·x² + B·x`` (the constant
    term is absorbed by the auto-offset), the secant slope
    ``s = (ty - my)/(tx - mx)`` requires ``B = s - A·(tx + mx)``. Sweeping
    ``A`` over ``[_ARC_A_MIN, _ARC_A_MAX]`` yields a ladder of arcs that all
    pass through the muzzle and the enemy but bulge to different extents, so
    the integrator (the oracle) can find one that clears the rocks. Candidates
    are ordered nearest-enemy-first, then by ``|A|`` (flattest first, so the
    cheapest clear arc is tried before the most extreme).
    """
    if not frame.targets:
        return []
    mx, my = frame.mx, frame.my
    a_values = _arc_grid()
    cands: list[_Candidate] = []
    for tx, ty in sorted(frame.targets, key=lambda p: p[0]):
        if abs(tx - mx) < 1e-9:
            continue
        s = (ty - my) / (tx - mx)
        for A in a_values:
            B = s - A * (tx + mx)
            cands.append(
                _Candidate(
                    expression=f"({_num(A)})*x^2+({_num(B)})*x",
                    rung=RUNG_ARC,
                    m_bound=2.0 * abs(A),
                )
            )
        cands.extend(_sharp_candidates(mx, my, tx, ty))
    cands.sort(key=lambda c: c.m_bound)
    return cands


def _sharp_candidates(mx: float, my: float, tx: float, ty: float) -> list[_Candidate]:
    """Sharp-turn single-target curves through muzzle and enemy (RUNG_ARC).

    Quadratics alone cannot turn sharply: their curvature is constant. These
    closed-form families supply what the arc sweep lacks, all through the
    auto-offset constraint ``f(tx) - f(mx) = ty - my`` so every candidate
    passes through the muzzle and the enemy by construction (the oracle still
    verifies clearance):

    - ``abs`` kink ``a*|x-c|``: a V corner at ``c`` (piecewise linear,
      ``m_bound = 0``) — a sharp turn with no singularity.
    - ``sqrt`` launch ``a*sqrt(x-c)`` with ``c`` behind the muzzle: near-
      vertical tangent at ``c`` that flattens downrange.
    - reciprocal ``a/(x-c)`` with the pole outside ``[mx, tx]``: a steep
      terminal dive (pole beyond the target) or steep launch (pole behind
      the muzzle).
    - ``tan`` bend ``a*tan((x-mx)/w)`` with the first pole past the target:
      a smooth but sharply steepening climb.

    Degenerate placements (zero denominator, non-finite weight, pole inside
    the span for the smooth families) are skipped. All bounds are finite.
    """
    out: list[_Candidate] = []
    span = tx - mx
    dy = ty - my
    if abs(span) < 1e-9:
        return out

    def _finite(a: float) -> bool:
        return math.isfinite(a) and abs(a) < 1e9

    # --- abs kink at fractions of the span ---------------------------------
    for frac in (0.25, 0.5, 0.75):
        c = mx + frac * span
        denom = abs(tx - c) - abs(mx - c)
        if abs(denom) < 1e-9:
            continue
        a = dy / denom
        if not _finite(a):
            continue
        out.append(
            _Candidate(
                expression=f"({_num(a)})*abs(x-({_num(c)}))",
                rung=RUNG_ARC,
                m_bound=0.0,
            )
        )

    # --- sqrt launch: pole behind the muzzle --------------------------------
    for d in (0.5, 2.0, 5.0):
        c = mx - d
        denom = math.sqrt(tx - c) - math.sqrt(mx - c)
        if abs(denom) < 1e-9:
            continue
        a = dy / denom
        if not _finite(a):
            continue
        # |f''| = |a| / (4 (x-c)^{3/2}), max at the muzzle (nearest the pole).
        m = abs(a) / (4.0 * (mx - c) ** 1.5)
        if not math.isfinite(m):
            continue
        out.append(
            _Candidate(
                expression=f"({_num(a)})*sqrt(x-({_num(c)}))",
                rung=RUNG_ARC,
                m_bound=m,
            )
        )

    # --- reciprocal: pole outside the span -----------------------------------
    for c in (tx + 1.0, tx + 3.0, tx + 8.0, mx - 1.0, mx - 3.0, mx - 8.0):
        if min(mx, tx) < c < max(mx, tx):
            continue  # mid-span pole: the shot dies at the asymptote
        denom = 1.0 / (tx - c) - 1.0 / (mx - c)
        if abs(denom) < 1e-12:
            continue
        a = dy / denom
        if not _finite(a):
            continue
        nearest = min(abs(mx - c), abs(tx - c))
        m = 2.0 * abs(a) / (nearest**3)
        if not math.isfinite(m) or m > 1e9:
            continue
        out.append(
            _Candidate(
                expression=f"({_num(a)})/(x-({_num(c)}))",
                rung=RUNG_ARC,
                m_bound=m,
            )
        )

    # --- tan bend: first pole past the target ---------------------------------
    for k in (0.75, 1.5, 3.0):
        w = span * k
        if abs(w) < 1e-9:
            continue
        # Phase 0 at the muzzle; first pole at mx + w*pi/2, past tx for k > 2/pi.
        if abs(w) * math.pi / 2.0 <= abs(span):
            continue
        t_tx = math.tan((tx - mx) / w)
        denom = t_tx - math.tan(0.0)
        if abs(denom) < 1e-12:
            continue
        a = dy / denom
        if not _finite(a):
            continue
        # |f''| = 2|a|/w^2 * sec^2(u)|tan(u)|, bounded by the endpoint max.
        t_max = max(abs(t_tx), 0.0)
        m = 2.0 * abs(a) * (1.0 + t_max * t_max) * t_max / (w * w)
        if not math.isfinite(m):
            continue
        out.append(
            _Candidate(
                expression=f"({_num(a)})*tan((x-({_num(mx)}))/({_num(w)}))",
                rung=RUNG_ARC,
                m_bound=m,
            )
        )
    return out


def _arc_grid() -> list[float]:
    """The ``A`` sweep grid ``[_ARC_A_MIN, _ARC_A_MAX]`` step ``_ARC_A_STEP``.

    Built in integer steps to avoid float accumulation drift.
    """
    n = int(round((_ARC_A_MAX - _ARC_A_MIN) / _ARC_A_STEP))
    return [_ARC_A_MIN + i * _ARC_A_STEP for i in range(n + 1)]


# --- Obstacle-clearing rung (flat-launch wall climbers) ----------------------
# Families chosen by the Task-2 probe on first-turn SOLVER_FAILED seeds
# (2v2 seeds 2/18/23/27): every secant-slope shape dies on the muzzle wall
# within ~60 steps, while flat-launch + delayed-maneuver shapes survive
# 800–1500 steps and thread it. Findings baked in here:
#
# - The killer is the muzzle wall (rock in the first 1–9 world units), NOT
#   the widest mid-field blocker: over/under variants aimed at the widest
#   blocker all die early because their secant-slope launch crosses
#   near-muzzle rock first. The maneuver centre goes at the CLIMB POINT
#   (where open space reappears past the wall), never at the blocker.
# - Launch slope must hold the muzzle sliver: both shapes below launch flat
#   by construction (the kink's ``m == a`` cancels the corner's left slope;
#   the sigmoid/bump tails are ~0 at the muzzle).
# - Pole families are disqualified inside the span (mid-span poles die at
#   the asymptote; near-vertical segments trip the step-halving floor or
#   exit via the saturating int cast). Cube-root-via-``^`` of a negative
#   base is NaN per the parser's Java ``Math.pow`` semantics, so no
#   ``(x-c)^(1/3)`` variants: only shapes legal for all ``x >= mx``.
# - Plateaus sit at the enemy height (in-band by construction — enemies are
#   always inside the plane), so over-variants cannot band-exit at the top.
# Every candidate still passes through the ``_verify`` oracle before
# selection: a bad placement just loses.
#
# Under-pairing falls out of the sign of ``ty - my`` (descending targets
# get descending maneuvers). A true dip-under (below a blocker, then back
# up — non-monotone in height) needs two maneuvers and is out of scope.

# Maneuver-centre scan step (world units) and the free-interval height that
# counts as "open space past the wall". # TUNABLE — not from source.
_WALL_SCAN_STEP: float = 0.5
_WALL_OPEN_HEIGHT: float = 2.0
# Sigmoid steepness sweep (steeper first: it holds the plateau longer and
# clips less). # TUNABLE — not from source.
_WALL_SIGMOID_B: tuple[float, ...] = (1.0, 0.5)
# Bound on ``|σ(1-σ)(1-2σ)|`` over [0,1] (attained at u=(1±1/√3)/2), so the
# sigmoid ``c·σ(b(x-k))`` has ``|f''| ≤ |c|·b²/(6√3)``.
_SIGMOID_CURVE: float = 6.0 * math.sqrt(3.0)


def _x_minus(c: float) -> str:
    """Emit ``x - c`` folding a negative centre's sign (cf.
    :func:`emission.gauss_term`: a negative centre must not double-negate
    under the parser's ``-`` → ``+-`` rewrite)."""
    return f"x-({_num(c)})" if c >= 0 else f"x+({_num(-c)})"


def _frame_rock(
    frame: _Frame,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Rock circles/carves in the shooter-facing frame (the TEAM2 mirror
    applied once, exactly like ``corridor.sweep_target``)."""
    if frame.inverted:
        mirc = tuple((config.PLANE_LENGTH - cx, cy, r) for cx, cy, r in frame.circles)
        mcarv = tuple((config.PLANE_LENGTH - cx, cy, r) for cx, cy, r in frame.carves)
        return mirc, mcarv
    return frame.circles, frame.carves


def _wall_end(
    mcirc: tuple[tuple[int, int, int], ...],
    mcarv: tuple[tuple[int, int, int], ...],
    mx: float,
    tx: float,
) -> float | None:
    """First column past the muzzle whose free set holds an interval of
    height ≥ ``_WALL_OPEN_HEIGHT`` (rock-only: teammate disks are the
    oracle's call, not the blocker's). ``None`` when the whole span stays
    narrow (a full-height wall — undiggable by flight, stays a dig)."""
    from .corridor import _column_free

    wx = mx
    while wx < tx:
        free = _column_free(wx, mcirc, (), mcarv)
        if any(hi - lo >= _WALL_OPEN_HEIGHT for lo, hi in free):
            return wx
        wx += _WALL_SCAN_STEP
    return None


def _wall_candidates(frame: _Frame) -> list[_Candidate]:
    """Flat-launch wall climbers, one maneuver per target (RUNG_WALL).

    Per target (nearest first): maneuver centres at the wall end (when
    strictly inside the span) plus span fractions; per centre a climb kink
    (``m == a`` flat launch, ``m_bound = 0``) and delayed sigmoids
    (endpoint-exact ``c``, plateau at the enemy height). Deterministic, no
    RNG.
    """
    mcirc, mcarv = _frame_rock(frame)
    mx, my = frame.mx, frame.my
    cands: list[_Candidate] = []
    for tx, ty in sorted(frame.targets, key=lambda p: p[0]):
        span = tx - mx
        if abs(span) < 1e-9:
            continue
        ks: list[float] = []
        end = _wall_end(mcirc, mcarv, mx, tx)
        if end is not None:
            ks.append(end)
        ks.extend((mx + 0.25 * span, mx + 0.45 * span))
        # Plateau-step second corners (downrange of the first): pairs
        # (k1, k2) with k1 at the wall end and k2 at span fractions.
        k2s = (mx + 0.6 * span, mx + 0.8 * span)
        seen: set[float] = set()
        for k in ks:
            if not (mx + 0.3 < k < tx - 0.3):
                continue
            key = round(k, 2)
            if key in seen:
                continue
            seen.add(key)
            # Climb kink ``m*x + a*|x-k|``: left of the corner the slope is
            # ``m - a`` (``k > mx``), so ``m == a`` launches flat through
            # the muzzle sliver, then corners up at the climb point. The
            # auto-offset constraint fixes ``a = (ty-my)/(2(tx-k))``.
            a = (ty - my) / (2.0 * (tx - k))
            if math.isfinite(a) and abs(a) < 1e9:
                cands.append(
                    _Candidate(
                        expression=f"({_num(a)})*x+({_num(a)})*abs({_x_minus(k)})",
                        rung=RUNG_WALL,
                        m_bound=0.0,
                    )
                )
            # Delayed sigmoid ``c/(1+e^(-b(x-k)))``: tails are ~0 at the
            # muzzle (flat launch) and ~c downrange; ``c`` is solved
            # endpoint-exact through the enemy.
            for b in _WALL_SIGMOID_B:
                try:
                    s_tx = 1.0 / (1.0 + math.exp(-b * (tx - k)))
                    s_mx = 1.0 / (1.0 + math.exp(-b * (mx - k)))
                except OverflowError:
                    continue
                d = s_tx - s_mx
                if abs(d) < 1e-9:
                    continue
                c = (ty - my) / d
                if not math.isfinite(c) or abs(c) >= 1e9:
                    continue
                m = abs(c) * b * b / _SIGMOID_CURVE
                if not math.isfinite(m):
                    continue
                cands.append(
                    _Candidate(
                        expression=f"({_num(c)})/(1+e^(-{_num(b)}*({_x_minus(k)})))",
                        rung=RUNG_WALL,
                        m_bound=m,
                    )
                )
            # Plateau step ``a*|x-k1| - a*|x-k2|`` (k1 = this corner): flat
            # plateaus outside (launch slope 0, right plateau at the enemy
            # height) with a linear ramp between — the piecewise-linear
            # sigmoid. Covers wall + second maneuver when the ramp threads
            # both (``a = (ty-my)/(2(k2-k1))`` via the auto-offset).
            for k2 in k2s:
                if not (k + 0.5 < k2 < tx - 0.3):
                    continue
                a2 = (ty - my) / (2.0 * (k2 - k))
                if not math.isfinite(a2) or abs(a2) >= 1e9:
                    continue
                cands.append(
                    _Candidate(
                        expression=(
                            f"({_num(a2)})*abs({_x_minus(k)})+({_num(-a2)})*abs({_x_minus(k2)})"
                        ),
                        rung=RUNG_WALL,
                        m_bound=0.0,
                    )
                )
    return cands


def _ccf_candidates(frame: _Frame) -> list[_Candidate]:
    """CCF rung candidates (5.2.md §1/§6-§9): certified-corridor-fit shots."""
    from . import ccf  # local import: scipy is heavy and CCF fires mid-ladder only

    result = ccf.solve_for_frame(
        frame.mx,
        frame.my,
        frame.targets,
        frame.teammates,
        frame.circles,
        frame.inverted,
        carves=frame.carves,
    )
    certs = {c.target_index: c for c in result.certificates}
    return [
        _Candidate(
            expression=c.expression,
            rung=RUNG_CCF,
            m_bound=c.m_bound,
            cert=certs.get(c.target_index),
        )
        for c in result.candidates
    ]


def _predict_carve(result: ShotResult, inverted: bool) -> tuple[int, int, int]:
    """The blast center :meth:`Game.fire` will carve for this shot.

    Replicates ``state.Game.fire`` exactly (GameData.java:1023-1026: the TEAM2
    mirror applies to the ``(int)``-narrowed x): ``PLANE_LENGTH - int(last_x)``
    when inverted, else ``int(last_x)``.
    """
    from .physics import _java_int_cast

    ex = (
        config.PLANE_LENGTH - _java_int_cast(result.last_x)
        if inverted
        else _java_int_cast(result.last_x)
    )
    return (ex, _java_int_cast(result.last_y), config.EXPLOSION_RADIUS)


def _carve_hits_rock(game: Game, carve: tuple[int, int, int]) -> bool:
    """True when the predicted crater disk covers any *remaining* rock.

    Ground truth is the live grid (out-of-bounds never counts — the physics'
    ``collide_point`` returns True there, so OOB is excluded explicitly).
    A carve that hits no rock digs nothing and is never a useful sacrifice.
    """
    ex, ey, er = carve
    r2 = er * er
    for yy in range(max(0, ey - er), min(config.PLANE_HEIGHT, ey + er + 1)):
        dy = yy - ey
        for xx in range(max(0, ex - er), min(config.PLANE_LENGTH, ex + er + 1)):
            dx = xx - ex
            if dx * dx + dy * dy <= r2 and game.terrain.collide_point(xx, yy):
                return True
    return False


def _digging_candidates(frame: _Frame) -> list[_Candidate]:
    """Small cheap-first digging set aimed at the nearest enemies.

    The full ladder is too expensive for 1-ply lookahead (~200 integrations +
    sweeps per blocked turn); this set (quadratic arcs on a coarse grid plus
    one kink and one steep dive per target, plus near-flat shots) covers the
    useful digging directions at ~20 simulations. All carry RUNG_SACRIFICIAL;
    a candidate that turns out to hit cleanly is relabeled RUNG_ARC by the
    caller (corridor proof gap — a real hit, not a sacrifice).
    """
    cands: list[_Candidate] = []
    mx = frame.mx
    for tx, ty in sorted(frame.targets, key=lambda p: p[0])[:2]:
        if abs(tx - mx) < 1e-9:
            continue
        s = (frame.my - ty) / (mx - tx) if abs(mx - tx) > 1e-9 else 0.0
        for A in (-0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06):
            B = s - A * (tx + mx)
            cands.append(
                _Candidate(
                    expression=f"({_num(A)})*x^2+({_num(B)})*x",
                    rung=RUNG_SACRIFICIAL,
                    m_bound=2.0 * abs(A),
                )
            )
        c = (mx + tx) / 2.0
        denom = abs(tx - c) - abs(mx - c)
        if abs(denom) > 1e-9:
            a = (ty - frame.my) / denom
            if math.isfinite(a) and abs(a) < 1e9:
                cands.append(
                    _Candidate(
                        expression=f"({_num(a)})*abs(x-({_num(c)}))",
                        rung=RUNG_SACRIFICIAL,
                        m_bound=0.0,
                    )
                )
        c = tx + 3.0
        denom = 1.0 / (tx - c) - 1.0 / (mx - c)
        if abs(denom) > 1e-12:
            a = (ty - frame.my) / denom
            nearest = min(abs(mx - c), abs(tx - c))
            m = 2.0 * abs(a) / (nearest**3)
            if math.isfinite(a) and math.isfinite(m) and m < 1e9:
                cands.append(
                    _Candidate(
                        expression=f"({_num(a)})/(x-({_num(c)}))",
                        rung=RUNG_SACRIFICIAL,
                        m_bound=m,
                    )
                )
    for e in ("0*x", "(0.05)*x", "(-0.05)*x"):
        cands.append(_Candidate(expression=e, rung=RUNG_SACRIFICIAL, m_bound=0.0))
    seen: set[str] = set()
    uniq: list[_Candidate] = []
    for cd in cands:
        if cd.expression not in seen:
            seen.add(cd.expression)
            uniq.append(cd)
    uniq.sort(key=lambda c: c.m_bound)
    return uniq


def _sacrificial_candidate(game: Game, frame: _Frame) -> _Candidate | None:
    """1-ply digging lookahead for corridor-blocked turns.

    For each digging candidate: simulate (never applying kills), predict the
    blast center, and re-run the corridor sweep with that hypothetical crater.
    Returns the cheapest candidate that OPENS a corridor (any target
    reachable), else the candidate whose crater bites DEEPEST toward the enemy
    (shooter-facing world x) — progressive tunneling: each turn's blast lands
    as far inside the rock as the current tunnel allows, so the face advances
    every turn until the corridor opens. Returns None when no candidate
    touches rock (the block is teammate disks, not terrain, or every shot
    dies off-map — the caller passes).

    Blocked-column advance is NOT the metric: on a thick wall every shot dies
    at the near flank while the sweep dies deep inside, so one crater never
    moves the blocked column even as the tunnel advances. Penetration depth is
    the honest nibbling signal.

    Teammate-grazing candidates are rejected; a candidate that hits an enemy
    cleanly is returned relabeled RUNG_ARC (corridor proof gap, a real hit).
    """
    from . import corridor

    base = tuple(getattr(game, "carves", ()))
    best: _Candidate | None = None
    best_depth = -math.inf
    for cand in _digging_candidates(frame):
        hit_e, hit_t, result = _verify(game, frame, cand)
        if hit_t:
            continue
        if hit_e:
            return _Candidate(expression=cand.expression, rung=RUNG_ARC, m_bound=cand.m_bound)
        carve = _predict_carve(result, frame.inverted)
        if not _carve_hits_rock(game, carve):
            continue
        new = corridor.sweep_targets(
            frame.mx,
            frame.my,
            frame.targets,
            frame.teammates,
            frame.circles,
            frame.inverted,
            carves=tuple(list(base) + [carve]),
        )
        if any(r.reachable for r in new):
            return cand
        wcx, _wcy = _plane_to_world(float(carve[0]), float(carve[1]), frame.inverted)
        if wcx > best_depth:
            best_depth = wcx
            best = cand
    return best


def _dud_candidates(salt: int = 0, rung: str = RUNG_PASS_UNREACHABLE) -> list[_Candidate]:
    """Safe fall-back expressions, rotated by ``salt`` (M5.1: emitted only as
    a PASS_UNREACHABLE / SOLVER_FAILED placeholder, never as a rung).

    A flat line travels horizontally and, on unreachable maps, stops at terrain
    or the map edge without reaching any enemy; it is still self-verified
    before emission. The rotation (keyed by ``len(game.carves)`` at the call
    site) cycles near-flat variants so consecutive dead turns do not repeat
    the identical expression: each digs a slightly different crater.
    Deterministic in the game state (no RNG), so ``test_solve_deterministic``
    still holds for fresh games.
    """
    variants = (
        "0*x",
        "(0.05)*x",
        "(-0.05)*x",
        "(0.02)*x^2",
        "(-0.02)*x^2",
        "(0.1)*x",
        "(-0.1)*x",
    )
    ordered = [variants[(salt + i) % len(variants)] for i in range(len(variants))]
    return [_Candidate(expression=e, rung=rung, m_bound=0.0) for e in ordered]


def _ordered_candidates(frame: _Frame) -> Iterator[_Candidate]:
    """The degradation ladder, best first; ``None`` entries dropped.

    Ladder order per 5.2.md §1: ``per_target_gaussian`` (×``_B_WIDTHS``) ->
    ``parabola`` -> ``line`` -> ``ccf`` -> ``fixed_grid_gaussian`` ->
    ``arc`` -> ``wall_climb``. The multi-target Gaussian rungs lead (the plan's intended primary
    basis, and the only rung that can multi-kill), then the closed-form
    single-target rungs, then the M5.2 CCF rung, then the plan's named
    fixed-grid basis, and finally the terrain-aware arc sweep and the
    obstacle-aware wall climbers.

    The ``ccf`` rung sits between ``line`` and ``fixed_grid_gaussian``, which
    is now on death row — it exists only until CCF beats it on the same seeds
    (5.2.md §1; end state is four rungs with ``arc`` last). Every CCF
    candidate still passes through the same ``_verify`` oracle loop as the
    other rungs: a CCF certificate proves *clearance*, but whether the shot
    *hits* an enemy is the simulator's call.

    This is a LAZY generator: CCF's expensive scipy LP/MILP work is only paid
    for when it is reached (i.e. after every cheaper rung fails to land a
    verified clean hit), so the common path never touches scipy at all.
    """
    for b in _B_WIDTHS:
        cand = _per_target_gaussians(frame, b)
        if cand is not None:
            yield cand
    cand = _parabola_candidate(frame)
    if cand is not None:
        yield cand
    cand = _line_candidate(frame)
    if cand is not None:
        yield cand
    yield from _ccf_candidates(frame)
    cand = _fixed_grid_gaussians(frame, _B_WIDTHS[0])
    if cand is not None:
        yield cand
    # The terrain-aware arc sweep is the last generic attempt to hit
    # something: it only runs on the maps where every cheaper rung failed to
    # land a shot (i.e. would otherwise dud), so its per-candidate
    # integration cost is paid only there. It generalises the parabola rung
    # by sweeping curvature so the curve can clear terrain
    # (see docs/OPEN_QUESTIONS.md).
    yield from _arc_candidates(frame)
    # The obstacle-aware wall climbers run dead last among kill-seeking
    # rungs (kills-first: they only win turns the whole older ladder scored
    # 0–partial on). Their flat launch threads muzzle walls the secant-slope
    # families structurally cannot (see _wall_candidates).
    yield from _wall_candidates(frame)


# --- Self-verification (the simulator is the oracle) ------------------------


def _verify(game: Game, frame: _Frame, cand: _Candidate) -> tuple[int, bool, ShotResult]:
    """Fire ``cand`` through the real integrator.

    Returns ``(enemy_kills, hit_any_teammate, result)`` where ``enemy_kills``
    counts distinct enemy soldiers hit (the engine is multi-kill: soldier
    hits never stop the shot, physics.py:276-290). A ``MalformedFunction``
    (should not happen for our own emission) is reported as a dud that hits
    nothing. Truthiness is unchanged for existing call sites: ``0`` is
    falsy, ``>= 1`` truthy.
    """
    try:
        f = PolishNotationFunction(cand.expression)
    except MalformedFunction:
        return 0, False, ShotResult()
    result = process_function_range(
        f, frame.shooter, game.all_soldiers(), game.terrain, frame.inverted
    )
    enemy_ids = {(s.player_index, s.soldier_index) for s in frame.enemies}
    enemy_kills = 0
    hit_any_teammate = False
    for player_index, soldier_index, _pos in result.hits:
        if (player_index, soldier_index) in enemy_ids:
            enemy_kills += 1
        else:
            hit_any_teammate = True
    return enemy_kills, hit_any_teammate, result


# --- Clearance margin (equal-kill tie-break) ---------------------------------


def _clearance(
    frame: _Frame,
    result: ShotResult,
    mcirc: tuple[tuple[int, int, int], ...],
    mcarv: tuple[tuple[int, int, int], ...],
) -> float:
    """Pre-first-kill rock clearance of a verified shot (world units).

    The min over subsampled trajectory points (up to the first hit) of the
    distance to the nearest blocked edge in the rock-only free set
    (``corridor._column_free`` margins — exact w.r.t. the crisp pixel grid;
    teammate disks excluded: this is terrain cleanliness, not safety).
    Points inside rock / off-map score 0. Deterministic, no RNG. Kills
    always compare first — this only orders verified equal-kill shots.
    """
    from .corridor import _column_free

    points = result.points
    if not points:
        return 0.0
    end = len(points)
    if result.hits:
        end = min(int(pos) + 1 for _, _, pos in result.hits)
        end = max(1, min(end, len(points)))
    step = max(1, end // 40)
    worst = math.inf
    for i in range(0, end, step):
        px, py = points[i]
        if frame.inverted:
            px = config.PLANE_LENGTH - px
        wx = config.PLANE_GAME_LENGTH * (px - config.PLANE_LENGTH / 2.0) / config.PLANE_LENGTH
        wy = config.PLANE_GAME_LENGTH * (-py + config.PLANE_HEIGHT / 2.0) / config.PLANE_LENGTH
        margin = 0.0
        for lo, hi in _column_free(wx, mcirc, (), mcarv):
            if lo <= wy <= hi:
                margin = min(wy - lo, hi - wy)
                break
        if margin < worst:
            worst = margin
            if worst <= 0.0:
                break
    return worst if math.isfinite(worst) else 0.0


# --- Public API --------------------------------------------------------------


def solve(game: Game) -> SolverResult:
    """Solve the current turn: return a legal, self-verified expression.

    M5.1 flow, extended for destructible terrain and max-kill acceptance:

    1. **Corridor pre-check** (:mod:`graphwar_sim.corridor`, carve-aware): if
       no enemy is reachable by any clean monotone trajectory, try a
       **sacrificial dig** (:func:`_sacrificial_candidate`): a 1-ply lookahead
       that fires into rock to blast a crater — opening the corridor now or
       biting deeper toward the enemy (``RUNG_SACRIFICIAL``). Only when no
       candidate touches rock (the block is teammate disks, not terrain) does
       the turn pass as ``RUNG_PASS_UNREACHABLE`` — a proof, not a failure.
       Dud variants rotate with ``len(game.carves)`` so consecutive dead
       turns never repeat the identical expression.
    2. Otherwise walk the degradation ladder for **max kills**: verify every
       candidate (lazy ``_ordered_candidates``, order unchanged); reject on
       teammate hit; return immediately on a perfect shot
       (``enemy_kills == n_targets`` — for ``n_targets == 1`` this reproduces
       first-hit behavior exactly); otherwise track ``best`` = first
       candidate with strictly more kills than any earlier one (strict ``>``
       keeps ladder order as the deterministic tie-break). After the ladder
       exhausts, fire ``best`` (a partial kill, ≥1) instead of falling to
       duds. Only when ``best is None`` (zero enemy hits anywhere) fall to
       the rotated duds → ``RUNG_SOLVER_FAILED``.
    3. If the ladder exhausts on a *reachable* map with no hit anywhere,
       return a rotated safe dud classified ``RUNG_SOLVER_FAILED`` (a fit
       gap, distinct from unreachable), ranked hit > off-map-or-sparing >
       rotation order among verified no-friendly-fire variants.

    :func:`solve` always returns a parseable expression and never raises for a
    well-formed :class:`Game`. Deterministic in the game state (no RNG): dud
    rotation and digging are keyed by ``len(game.carves)`` and geometry.
    """
    frame = _build_frame(game)
    salt = len(getattr(game, "carves", ()))
    if frame.targets:
        from . import corridor  # local import: corridor is a solver utility

        any_reachable = any(r.reachable for r in corridor.reachability(game))
        if not any_reachable:
            dig = _sacrificial_candidate(game, frame)
            if dig is not None:
                hit_e, hit_t, result = _verify(game, frame, dig)
                if not hit_t:
                    return _make_result(dig, result, frame, kills=int(hit_e))
                # Rejected on friendly fire (the re-verify is the authority;
                # the lookahead already skips teammate hitters).
            for dud in _dud_candidates(salt, RUNG_PASS_UNREACHABLE):
                hit_enemy, hit_teammate, result = _verify(game, frame, dud)
                if not hit_enemy and not hit_teammate:
                    return _make_result(dud, result, frame, kills=int(hit_enemy))
            # Corridor says unreachable yet every dud variant hits someone, or
            # a flat dud self-verifies a hit: a contradiction (corridor model
            # gap). Fall through to the ladder rather than misclassify the
            # turn (recorded in OPEN_QUESTIONS if it ever fires).
    n_targets = len(frame.targets)
    mcirc, mcarv = _frame_rock(frame)
    best: _Candidate | None = None
    best_result: ShotResult | None = None
    best_kills = 0
    best_clear = 0.0
    for cand in _ordered_candidates(frame):
        enemy_kills, hit_teammate, result = _verify(game, frame, cand)
        if hit_teammate:
            continue  # friendly fire — reject, fall through
        if n_targets > 0 and enemy_kills == n_targets:
            return _make_result(cand, result, frame, kills=int(enemy_kills))
        if enemy_kills > best_kills:
            best = cand
            best_result = result
            best_kills = int(enemy_kills)
            best_clear = _clearance(frame, result, mcirc, mcarv) if enemy_kills > 0 else 0.0
        elif enemy_kills == best_kills and enemy_kills > 0:
            # Equal kills: prefer the cleaner pre-kill path (larger rock
            # clearance). Strict ``>`` keeps ladder order on exact ties;
            # kills still dominate (this branch never fires across counts).
            clear = _clearance(frame, result, mcirc, mcarv)
            if clear > best_clear:
                best = cand
                best_result = result
                best_clear = clear
    if best is not None and best_result is not None:
        return _make_result(best, best_result, frame, kills=best_kills)
    # Ladder exhausted with zero enemy hits on a reachable (or
    # corridor-unchecked) map: a solver gap, distinct from an unreachable
    # target. Rotated safe dud (never the same expression twice in a row).
    # Among verified no-friendly-fire duds, kills still come first (a lucky
    # dud hit outranks any clean miss), then a dud that exits off-map or
    # carves no rock (a sky-exit plows nothing — the carve clamp drops
    # fully-off-map blasts, and ``_carve_hits_rock`` proves the rest)
    # outranks one that plows rock; rotation order breaks exact ties, so
    # consecutive dead turns still vary.
    clean: _Candidate | None = None
    clean_result: ShotResult | None = None
    clean_kills = 0
    clean_key: tuple[int, int, int, int] = (0, 0, 0, 0)
    for order, dud in enumerate(_dud_candidates(salt, RUNG_SOLVER_FAILED)):
        hit_e, hit_t, result = _verify(game, frame, dud)
        if hit_t:
            continue
        not_carves = not _carve_hits_rock(game, _predict_carve(result, frame.inverted))
        key = (
            -int(hit_e > 0),
            -int(result.stop == "oob" or not_carves),
            -int(not_carves),
            order,
        )
        if clean is None or key < clean_key:
            clean, clean_result, clean_kills, clean_key = dud, result, int(hit_e), key
    if clean is not None and clean_result is not None:
        return _make_result(clean, clean_result, frame, kills=clean_kills)
    failed = _Candidate(expression="0*x", rung=RUNG_SOLVER_FAILED, m_bound=0.0)
    _hit_e, _hit_t, result = _verify(game, frame, failed)
    return _make_result(failed, result, frame, kills=int(_hit_e))


def _make_result(
    cand: _Candidate, result: ShotResult, frame: _Frame, kills: int | None = None
) -> SolverResult:
    bound = cand.m_bound * _DU * _DU / 8.0
    n_hits = len(result.hits)
    n_steps = result.num_steps
    if kills is None:
        enemy_ids = {(s.player_index, s.soldier_index) for s in frame.enemies}
        kills = sum(1 for pi, si, _pos in result.hits if (pi, si) in enemy_ids)
    notes = (
        f"hits={n_hits} steps={n_steps} enemies={len(frame.enemies)}"
        f" kills={kills}/{len(frame.targets)}"
    )
    if cand.cert is not None:
        cert = cand.cert
        notes += (
            f" ccf={cert.outcome.value}"
            f" sigma={cert.sigma if cert.sigma is not None else '?'}"
            f" branch={cert.branch_kind if cert.branch_kind is not None else '?'}"
        )
    return SolverResult(
        expression=cand.expression,
        rung=cand.rung,
        bound=bound,
        notes=notes,
        cert=cand.cert,
        kills=kills,
    )


@dataclass
class BatteryResult:
    """Per-seed outcome of :func:`run_battery` (for the M2 acceptance log)."""

    seed: int
    rung: str
    hit_any_enemy: bool
    hit_teammate: bool
    parseable: bool
    certified: bool


def run_battery(
    seeds: Sequence[int],
    num_teams: int = 2,
    num_soldiers: int = config.INITIAL_NUM_SOLDIERS,
) -> list[BatteryResult]:
    """Run :func:`solve` on a battery of seeded maps and record the acceptance
    metrics: parseable + certified per seed, plus the hit / no-friendly-fire
    outcome and the degradation rung used."""
    out: list[BatteryResult] = []
    for seed in seeds:
        game = Game.create(seed, num_teams=num_teams, num_soldiers=num_soldiers)
        res = solve(game)
        try:
            PolishNotationFunction(res.expression)
            parseable = True
        except MalformedFunction:
            parseable = False
        # Re-verify the outcome on a fresh copy of the same seed.
        game2 = Game.create(seed, num_teams=num_teams, num_soldiers=num_soldiers)
        frame = _build_frame(game2)
        cand = _Candidate(expression=res.expression, rung=res.rung, m_bound=0.0)
        hit_e, hit_t, _r = _verify(game2, frame, cand)
        out.append(
            BatteryResult(
                seed=seed,
                rung=res.rung,
                hit_any_enemy=bool(hit_e),
                hit_teammate=hit_t,
                parseable=parseable,
                certified=math.isfinite(res.bound),
            )
        )
    return out
