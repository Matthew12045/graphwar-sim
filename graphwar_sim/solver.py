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
the only rung that can multi-kill); the fixed-grid rung sits low because it
underperforms the others on the seeded battery and is the only one that
produced friendly fire (see ``docs/OPEN_QUESTIONS.md`` for the recorded
divergence from the plan's fixed-grid-primary ordering):

1. ``per_target_gaussian`` — one Gaussian centre per enemy, several widths.
2. ``parabola`` / ``line`` — closed-form single-target curves (auto-offset
   baked in).
3. ``fixed_grid_gaussian`` — a fixed uniform grid of centres (the plan's named
   primary; it underperforms on the seeded maps — the documented divergence).
4. ``dud`` — a deliberate safe dud that hits nothing (never crashes).

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
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import config
from .parser import MalformedFunction, PolishNotationFunction
from .physics import ShotResult, Soldier, process_function_range
from .state import Game

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

# Rung labels (Phase 4 metric).
RUNG_PER_TARGET = "per_target_gaussian"
RUNG_FIXED_GRID = "fixed_grid_gaussian"
RUNG_LINE = "line"
RUNG_PARABOLA = "parabola"
RUNG_ARC = "arc"
RUNG_DUD = "dud"


@dataclass
class SolverResult:
    """Outcome of :func:`solve`.

    - ``expression``: a legal Graphwar ``y = f(x)`` string (always parseable).
    - ``rung``: which degradation rung produced it (a Phase 4 metric).
    - ``bound``: the certification bound ``M · du² / 8`` (finite for every rung).
    - ``notes``: free-form diagnostics (e.g. self-verification hit summary).
    """

    expression: str
    rung: str
    bound: float
    notes: str = ""


@dataclass
class _Candidate:
    """An internal candidate: an emitted expression plus its certification data."""

    expression: str
    rung: str
    m_bound: float  # bound on |f''| (world units); 0 for an exact line


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


def _build_frame(game: Game) -> _Frame:
    """Assemble the shooter-facing world frame (muzzle + live enemies)."""
    team = game.state.current_team()
    shooter = team.current_soldier()
    inverted = team.team == config.TEAM2
    mx, my = _plane_to_world(shooter.x, shooter.y, inverted)
    targets: list[tuple[float, float]] = []
    enemies: list[Soldier] = []
    for t in game.state.teams:
        if t.team == team.team:
            continue  # own side (teammates) — never a target
        for s in t.soldiers:
            if not s.alive:
                continue
            ex, ey = _plane_to_world(s.x, s.y, inverted)
            targets.append((ex, ey))
            enemies.append(s)
    return _Frame(
        mx=mx, my=my, targets=targets, shooter=shooter, enemies=enemies, inverted=inverted
    )


# --- Emission (validated sign-folding; see tests/test_solver.py) ------------


def _num(v: float) -> str:
    """Format a numeric literal as plain decimal, never scientific notation.

    The parser rewrites every ``-`` to ``+-`` (unary negation), so a literal
    such as ``1e-06`` would become ``1e+-06`` and misparse as ``1*e - 6``.
    We therefore never emit an exponent. Twelve decimal places round-trip a
    double far inside the ~1e-9 tolerance the round-trip test asserts.
    """
    s = f"{v:.12f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _gauss_term(w: float, c: float, b: float) -> str:
    """Emit one Gaussian term, folding the centre's sign into the operator.

    The parser rewrites every ``-`` to ``+-`` (unary negation), so a negative
    centre inside ``x - c`` would double-negate. We emit ``x+(|c|)`` for a
    negative centre instead. Validated against the real parser to ~1e-16.
    """
    center = f"x-({_num(c)})" if c >= 0 else f"x+({_num(-c)})"
    return f"({_num(w)})*e^(-{_num(b)}({center})^2)"


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
    cands.sort(key=lambda c: c.m_bound)
    return cands


def _arc_grid() -> list[float]:
    """The ``A`` sweep grid ``[_ARC_A_MIN, _ARC_A_MAX]`` step ``_ARC_A_STEP``.

    Built in integer steps to avoid float accumulation drift.
    """
    n = int(round((_ARC_A_MAX - _ARC_A_MIN) / _ARC_A_STEP))
    return [_ARC_A_MIN + i * _ARC_A_STEP for i in range(n + 1)]


def _dud_candidates() -> list[_Candidate]:
    """Deliberate safe duds (last resort): flat / steep-up / shallow-arc.

    A flat line at the muzzle height travels horizontally and, on the seeded
    maps, stops at terrain or the map edge without reaching any enemy (which
    are on the far side). Each is still self-verified to hit nothing.
    """
    return [
        _Candidate(expression="0*x", rung=RUNG_DUD, m_bound=0.0),
        _Candidate(expression="(0.05)*x", rung=RUNG_DUD, m_bound=0.0),
        _Candidate(expression="(0.001)*x*(x-25)", rung=RUNG_DUD, m_bound=0.002),
    ]


def _ordered_candidates(frame: _Frame) -> list[_Candidate]:
    """The degradation ladder, best first; ``None`` entries dropped.

    The multi-target Gaussian rungs lead (the plan's intended primary basis,
    and the only rung that can multi-kill): per-target-centre Gaussians first,
    then the closed-form single-target rungs (parabola, line), then the plan's
    named fixed-grid basis, and finally the safe duds. The fixed-grid rung is
    kept low — and below the closed-form rungs — because on the seeded battery
    its isolated hit rate trails the others and it is the only rung that
    produced friendly fire (see ``docs/OPEN_QUESTIONS.md`` for the recorded
    divergence from the plan's fixed-grid-primary ordering).
    """
    cands: list[_Candidate | None] = []
    for b in _B_WIDTHS:
        cands.append(_per_target_gaussians(frame, b))
    cands.append(_parabola_candidate(frame))
    cands.append(_line_candidate(frame))
    cands.append(_fixed_grid_gaussians(frame, _B_WIDTHS[0]))
    # The terrain-aware arc sweep is the last real attempt to hit something:
    # it only runs on the maps where every cheaper rung failed to land a shot
    # (i.e. would otherwise dud), so its per-candidate integration cost is paid
    # only there. It generalises the parabola rung by sweeping curvature so the
    # curve can clear terrain (see docs/OPEN_QUESTIONS.md).
    cands.extend(_arc_candidates(frame))
    cands.extend(_dud_candidates())
    return [c for c in cands if c is not None]


# --- Self-verification (the simulator is the oracle) ------------------------


def _verify(game: Game, frame: _Frame, cand: _Candidate) -> tuple[bool, bool, ShotResult]:
    """Fire ``cand`` through the real integrator.

    Returns ``(hit_any_enemy, hit_any_teammate, result)``. A ``MalformedFunction``
    (should not happen for our own emission) is reported as a dud that hits
    nothing.
    """
    try:
        f = PolishNotationFunction(cand.expression)
    except MalformedFunction:
        return False, False, ShotResult()
    result = process_function_range(
        f, frame.shooter, game.all_soldiers(), game.terrain, frame.inverted
    )
    enemy_ids = {(s.player_index, s.soldier_index) for s in frame.enemies}
    hit_any_enemy = False
    hit_any_teammate = False
    for player_index, soldier_index, _pos in result.hits:
        if (player_index, soldier_index) in enemy_ids:
            hit_any_enemy = True
        else:
            hit_any_teammate = True
    return hit_any_enemy, hit_any_teammate, result


# --- Public API --------------------------------------------------------------


def solve(game: Game) -> SolverResult:
    """Solve the current turn: return a legal, self-verified expression.

    Walks the degradation ladder and returns the first candidate that is
    parseable, certified (finite bound), hits no teammate, and (for the
    non-dud rungs) hits at least one enemy. The final rung is a safe dud that
    hits nothing, so :func:`solve` always returns a parseable expression and
    never raises for a well-formed :class:`Game`.
    """
    frame = _build_frame(game)
    for cand in _ordered_candidates(frame):
        hit_enemy, hit_teammate, result = _verify(game, frame, cand)
        if hit_teammate:
            continue  # friendly fire — reject, fall through
        if cand.rung == RUNG_DUD:
            if not hit_enemy:
                return _make_result(cand, result, frame)
            continue  # a dud that somehow hit an enemy is not safe; try next
        if hit_enemy:
            return _make_result(cand, result, frame)
    # Absolute fallback (should be unreachable: the flat dud hits nothing).
    fallback = _Candidate(expression="0*x", rung=RUNG_DUD, m_bound=0.0)
    _hit_e, _hit_t, result = _verify(game, frame, fallback)
    return _make_result(fallback, result, frame)


def _make_result(cand: _Candidate, result: ShotResult, frame: _Frame) -> SolverResult:
    bound = cand.m_bound * _DU * _DU / 8.0
    n_hits = len(result.hits)
    n_steps = result.num_steps
    return SolverResult(
        expression=cand.expression,
        rung=cand.rung,
        bound=bound,
        notes=f"hits={n_hits} steps={n_steps} enemies={len(frame.enemies)}",
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
                hit_any_enemy=hit_e,
                hit_teammate=hit_t,
                parseable=parseable,
                certified=math.isfinite(res.bound),
            )
        )
    return out
