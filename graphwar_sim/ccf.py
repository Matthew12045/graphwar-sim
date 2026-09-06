"""CCF — Certified Corridor Fit (M5.2, per ``5.2.md`` v2).

A certified fit of ``y = f(u)`` (shooter-relative ``u``, muzzle at ``u = 0``)
through the cell-wise corridor from :mod:`graphwar_sim.corridor`: the emitted
expression is PROVEN — not merely observed — to keep the fired trajectory
inside the free geometry from the muzzle to the goal column, or the outcome
says exactly why not. The simulator stays the oracle for *hits*; CCF is the
proof layer for *clearance*.

Basis (5.2.md §3)
-----------------
    phi_j(u) = exp(-(u - c_j)^2 / (2 sigma^2)),   c_j uniform on [0, u_T]
    plus the affine terms  phi_lin(u) = u  and  phi_const(u) = 1.

The affine terms have ``f'' = 0`` — free curvature capacity. They absorb the
endpoint constraints (``f(0) = 0``, ``f(u_T) = dy_T``) instead of burning
Gaussian weight, are excluded from ``||w||_1`` and from ``s``, and get a small
separate penalty only to keep the string short.

Curvature bound (5.2.md §4)
---------------------------
With ``t = (u - c_j)/sigma``:

    phi_j''(u) = (1/sigma^2) * (t^2 - 1) * exp(-t^2 / 2)

``g(t) = (t^2 - 1) e^{-t^2/2}`` has critical points at ``t = 0`` (``|g| = 1``)
and ``t = ±sqrt(3)`` (``|g| = 2 e^{-1.5} ≈ 0.446``), and ``g → 0`` as
``|t| → ∞``, so ``sup_u |phi_j''(u)| = 1/sigma^2`` and, with the affine
contributing nothing,

    M = max|f''|  <=  sigma^-2 * ||w||_1  =  sigma^-2 * s.

``M`` is bounded by a NORM of the decision variables, so the required
interpolation margin ``M du^2 / 8`` is LINEAR in them and lives inside the LP.
No fit-then-check, no fixed-point loop.

Emission budget (5.2.md §2)
---------------------------
A J-term Gaussian sum is emitted with additions nested as a BALANCED tree
(:func:`graphwar_sim.emission.balanced_sum`, depth log2 J). The budget is
computed, not guessed: :func:`emission_budget` emits one representative term
with the plain-decimal formatter, measures it, and derives
``J_max = (MAX_EXPR_CHARS - affine_cost - safety) // per_term_cost`` from
``config.MAX_EXPR_CHARS`` (a harness cap — the reference imposes no length
limit, docs/OPEN_QUESTIONS.md (e)). Literals are plain decimals only; the
parser's ``-``→``+-`` rewrite corrupts exponents.

Truncation, exactly reconciled with the grammar (5.2.md §3)
-----------------------------------------------------------
The LP sets ``phi_j(u_k) = 0`` where ``|u_k - c_j| > 6 sigma`` (sparse
constraint matrix, no exp underflow). The EMITTED expression cannot truncate —
the grammar has no domain cut — and a term treated as 0 in the LP but alive in
the emission would invalidate the certificate. Reconciliation: the true tail
value beyond ``6 sigma`` is at most ``e^{-18}`` per unit weight, so every
margin row charges the tail term ``e^{-18} · s``. The certificate then holds
for the *emitted* expression exactly, tail included.

Launch-nudge allowance (GROUND_TRUTH.md §2.4)
---------------------------------------------
The game nudges the start point along the converged tangent by the game radius
``r`` and vertically offsets the whole curve so it passes through the nudged
point. The fired curve is therefore ``g(u) = f_ccf(u) + my + delta`` with

    delta = r sin(theta) - [f_ccf(r cos(theta)) - f_ccf(0)].

Bound (both terms use ``M = max|f''| <= s/sigma^2``; ``theta`` is the game's
fixed-point angle, ``tan theta = f'(xi*)`` with ``|xi*| <= r_eff = r +
STEP_SIZE`` — ``getStartAngle`` evaluates the tangent at
``x + radius·cos(angle)``):

1. Taylor remainder: ``|f(r cos t) - f(0) - r cos t · f'(0)| <= M r^2 / 2``
   (``|cos| <= 1``).
2. Angle term: ``sin theta = h(f'(xi*))`` with ``h(x) = x/sqrt(1+x^2)``, which
   is 1-Lipschitz (``h' = (1+x^2)^{-3/2} <= 1``); the game's
   ``cos theta = 1/sqrt(1+f'(xi*)^2)``, so
   ``|r sin theta - r cos(theta) f'(0)| = r |h(f'(xi*)) - h(f'(0))|
     <= r |f'(xi*) - f'(0)| <= M r r_eff``.

Hence ``|delta| <= (3/2) r_eff^2 M`` — a margin coefficient
``(3/2) r_eff^2 / sigma^2``, still LINEAR in ``s``. In the default *allowance*
mode the LP charges this coefficient and the fired curve is then provably
inside the cell envelopes with NO post-hoc step. In *tight* mode (tried when
the allowance LP is infeasible for the same branch) the coefficient is dropped
and the certificate is recovered exactly: the fired curve ``g`` is
reconstructed through the real launch-angle code
(:func:`graphwar_sim.physics._get_start_angle`) and re-checked at every
corridor boundary sample against the interpolation margin; the chord bound
(``g'' = f''``) covers the continuous curve between samples.

Outcomes (5.2.md §10)
---------------------
``CERTIFIED`` (slack 0; proven), ``UNCERTIFIED`` (slack > 0; usable, not
proven), ``EMIT_OVERFLOW`` (the math worked, the grammar did not — after the
MILP cardinality path and prune-to-fit), ``BASIS_INFEASIBLE`` (infeasible at
every sigma and branch), ``UNREACHABLE`` (corridor empty — M5.1's verdict).
Every escalation carries the bound, the ``u_k``, the sigma and the branch.

Clean-room note: the reference is GPL-licensed; this module cites behaviour,
does not copy code. No ``eval``/``exec``/dynamic import; expressions are built
from literals and parsed by the faithful
:class:`graphwar_sim.parser.PolishNotationFunction`. No cvxpy — the LP and the
cardinality MILP both run on scipy's HiGHS.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np

# scipy ships no py.typed (checked 1.18); the imports are untyped-library
# ignores, the call sites wrap results in explicit types/casts.
from scipy.optimize import Bounds, LinearConstraint, linprog, milp  # type: ignore[import-untyped]
from scipy.sparse import coo_matrix, csr_matrix, hstack  # type: ignore[import-untyped]

from . import config, corridor
from .corridor import Chain
from .emission import balanced_sum, format_literal, gauss_term
from .parser import MalformedFunction, PolishNotationFunction
from .physics import _get_start_angle

# --- Tunables (all # TUNABLE — not from the reference source) ----------------

# Sigma ladder, coarse -> fine; first feasible sigma wins (5.2.md §9). The
# values are chosen so 1/(2 sigma^2) is an exact short decimal (emission
# budget) and the coarsest entry resolves a full band height.
_SIGMA_LADDER: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5, 0.25)
# K best branches per sigma (5.2.md §8: "K is TUNABLE, default small").
_MAX_BRANCHES: int = 4
# Maximum chains the sweep may return. Now ONLY the sweep's chain cap
# feeding the §8/§9 reachability oracle (reachable / first_blocked_column);
# branch geometry comes from the dedicated _branch_paths layered-DAG DP, not
# from these backtraced chains.
_SWEEP_CHAINS: int = 16
# Nearest reachable targets attempted per invocation.
_MAX_TARGETS: int = 2
# Slack penalty: large enough that slack is used only when the hard problem is
# infeasible (5.2.md §6).
_LAMBDA_SLACK: float = 1.0e4
# Separate small penalty on |a| + |b| (keeps the affine string short; the
# affine is excluded from s by design).
_MU_AFFINE: float = 1.0e-3
# Prune threshold for LP weights; pruned solutions are RE-VERIFIED (§9).
_PRUNE_EPS: float = 1.0e-6
# Truncation radius in sigmas (§3). The tail term is exp(-this^2 / 2).
_TRUNCATION_SIGMAS: float = 6.0
# Slack total below this (world units) counts as zero (CERTIFIED).
_SLACK_TOLERANCE: float = 1.0e-6
# MILP time budget, seconds (§7).
_MILP_TIME_BUDGET: float = 2.0
# Emission-budget safety margin, characters (§2).
_EXPR_CHARS_SAFETY: int = 16
# Candidates surfaced per invocation (best first; the ladder's oracle tries
# them in order).
_MAX_CANDIDATES: int = 3
# Hard cap on LP/MILP solves per target (runtime guard).
_MAX_SOLVES_PER_TARGET: int = 24
# Numeric tolerance for post-hoc corridor re-checks (world units; 1e-9 world
# = 1.5e-8 px — far below any physical relevance).
_RECHECK_TOL: float = 1.0e-9
# Feasibility tolerance when re-checking an LP-produced solution: HiGHS'
# default primal feasibility is 1e-7, so a stricter re-check would spuriously
# downgrade LP-feasible fits. # TUNABLE — matches the solver default.
_LP_ROW_TOL: float = 1.0e-7
# Big-M multiplier for the cardinality MILP (§7; derivation at _big_m).
_BIGM_FACTOR: float = 4.0
# Node penalty for the K-paths diversity re-runs (§8): large enough to make the
# DP prefer never-used components whenever one exists. # TUNABLE.
_PATH_PENALTY: float = 1.0e3

# --- Derived constants --------------------------------------------------------

# Column spacing: the corridor's own sample density (Phase 0 STEP_SIZE).
_DU: float = config.STEP_SIZE
# Game radius nudged along the tangent at launch (GROUND_TRUTH.md §2.2/2.4).
_NUDGE_RADIUS: float = config.PLANE_GAME_LENGTH * config.SOLDIER_RADIUS / config.PLANE_LENGTH
# Effective nudge window: the converged tangent is evaluated within this range
# of the muzzle (getStartAngle evaluates at x + radius*cos(angle) ± STEP_SIZE).
_NUDGE_RADIUS_EFF: float = _NUDGE_RADIUS + config.STEP_SIZE
# Launch-nudge margin coefficient: (3/2) * r_eff^2 (times M = s/sigma^2).
_NUDGE_COEFF: float = 1.5 * _NUDGE_RADIUS_EFF * _NUDGE_RADIUS_EFF
# Gaussian tail beyond the truncation radius (unit weight).
_TAIL: float = math.exp(-_TRUNCATION_SIGMAS * _TRUNCATION_SIGMAS / 2.0)


class CCFOutcome(StrEnum):
    """Machine-readable CCF outcomes (5.2.md §10)."""

    CERTIFIED = "CERTIFIED"
    UNCERTIFIED = "UNCERTIFIED"
    EMIT_OVERFLOW = "EMIT_OVERFLOW"
    BASIS_INFEASIBLE = "BASIS_INFEASIBLE"
    UNREACHABLE = "UNREACHABLE"


_OUTCOME_PRECEDENCE: dict[CCFOutcome, int] = {
    CCFOutcome.CERTIFIED: 0,
    CCFOutcome.UNCERTIFIED: 1,
    CCFOutcome.EMIT_OVERFLOW: 2,
    CCFOutcome.BASIS_INFEASIBLE: 3,
    CCFOutcome.UNREACHABLE: 4,
}


@dataclass(frozen=True)
class EmissionBudget:
    """The emission budget derived from the real formatter (5.2.md §2)."""

    char_limit: int
    per_term_cost: int
    affine_cost: int
    safety: int
    j_max: int


@dataclass
class CCFCandidate:
    """One emittable CCF shot with its certificate data."""

    expression: str
    target_index: int
    sigma: float
    branch_index: int
    branch_kind: str  # "over" | "under" (midline vs. the muzzle->target line)
    certified: bool
    slack_total: float
    m_bound: float  # bound on |f''|: s / sigma^2 (world units)
    mode: str  # "allowance" | "tight"
    nonzero_weights: int
    binding: str | None = None  # tightest bound + u_k (escalation reason, §10)


@dataclass
class CCFCertificate:
    """Per-invocation instrumentation (5.2.md §12) + escalation reasons (§10)."""

    outcome: CCFOutcome
    target_index: int | None = None
    sigma: float | None = None
    ladder_depth: int | None = None
    branch_index: int | None = None
    branch_count: int = 0
    branch_kind: str | None = None
    slack_total: float | None = None
    nonzero_weights: int | None = None
    emitted_length: int | None = None
    char_limit: int = config.MAX_EXPR_CHARS
    milp_fired: bool = False
    solve_seconds: float = 0.0
    binding: str | None = None  # e.g. "lower@u=184.00" / "infeasible@sigma=0.5"
    trace: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class CCFSolution:
    """The CCF result for ONE target: candidates + its certificate."""

    target_index: int
    candidates: list[CCFCandidate]
    certificate: CCFCertificate


@dataclass
class CCFResult:
    """The CCF result for one invocation (all attempted targets)."""

    outcome: CCFOutcome
    candidates: list[CCFCandidate]  # best first (certified, then uncertified)
    certificates: tuple[CCFCertificate, ...]  # per attempted target


# --- Emission budget (5.2.md §2) ---------------------------------------------


def emission_budget(char_limit: int = config.MAX_EXPR_CHARS) -> EmissionBudget:
    """Derive ``J_max`` from the REAL formatter, not a guess (5.2.md §2).

    One representative Gaussian term (negative weight, negative centre, the
    coarsest ladder sigma) is emitted and measured; ``+3`` per term accounts
    for the balanced tree's ``+()`` wrapper share. The affine cost is measured
    the same way from representative slope/constant magnitudes.
    """
    per_term = len(gauss_term(-1.0, -12.3456, _sigma_b(_SIGMA_LADDER[0]))) + 3
    affine = (
        len(f"({format_literal(-1.234567890123)})*x")
        + len(f"({format_literal(-12.345678901234)})")
        + 6  # both affine terms' balanced-tree wrappers
    )
    j_max = (char_limit - affine - _EXPR_CHARS_SAFETY) // per_term
    return EmissionBudget(
        char_limit=char_limit,
        per_term_cost=per_term,
        affine_cost=affine,
        safety=_EXPR_CHARS_SAFETY,
        j_max=max(0, int(j_max)),
    )


def _sigma_b(sigma: float) -> float:
    """The emitted Gaussian coefficient ``b = 1/(2 sigma^2)``.

    The ladder sigmas are chosen so this is an exact short decimal.
    """
    return 1.0 / (2.0 * sigma * sigma)


# --- Basis --------------------------------------------------------------------


def _phi(u: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """Truncated basis matrix ``(len(u), J)``: ``exp(-(u-c)^2/(2 sigma^2))``
    where ``|u - c| <= 6 sigma`` (5.2.md §3), else 0. The tail beyond the cut
    is charged to the margin via :data:`_TAIL`, keeping the certificate exact
    for the untruncated emitted expression (see module docstring)."""
    d = np.abs(u[:, None] - centers[None, :])
    out = np.asarray(np.exp(-(d * d) / (2.0 * sigma * sigma)), dtype=float)
    out[d > _TRUNCATION_SIGMAS * sigma] = 0.0
    return out


def _margin_coef(sigma: float, mode: str) -> float:
    """Margin coefficient multiplying ``s`` in every corridor row.

    ``du^2/(8 sigma^2)`` (interpolation, §4/§6) + ``e^{-18}`` (Gaussian tail
    beyond the truncation cut) +, in allowance mode, the launch-nudge
    allowance ``(3/2) r_eff^2 / sigma^2`` (see module docstring).
    """
    m = _DU * _DU / (8.0 * sigma * sigma) + _TAIL
    if mode == "allowance":
        m += _NUDGE_COEFF / (sigma * sigma)
    return m


# --- The linear program (5.2.md §6) -------------------------------------------
# Variables: x = [w (J), t (J), s, a, b, slack (K), aa, ab]  (+ z (J) for MILP)


@dataclass
class _LPSolution:
    """Extracted LP/MILP solution."""

    w: np.ndarray
    s: float
    a: float
    b: float
    slack_total: float
    status: str  # "optimal" | "timeout" (infeasible/failed -> None)


def _lp_layout(J: int, K: int) -> tuple[int, int, int, int, int, int]:
    """Column indices: (i_s, i_a, i_b, i_slack0, i_aa, i_ab)."""
    i_s, i_a, i_b = 2 * J, 2 * J + 1, 2 * J + 2
    i_slack0, i_aa, i_ab = 2 * J + 3, 2 * J + 3 + K, 2 * J + 4 + K
    return i_s, i_a, i_b, i_slack0, i_aa, i_ab


class _LPRows:
    """COO accumulator for one sparse block of LP rows."""

    def __init__(self, n_cols: int) -> None:
        self._n_cols = n_cols
        self._rows: list[int] = []
        self._cols: list[int] = []
        self._vals: list[float] = []
        self.b: list[float] = []

    def add(self, entries: list[tuple[int, float]], rhs: float) -> None:
        r = len(self.b)
        for c, v in entries:
            self._rows.append(r)
            self._cols.append(c)
            self._vals.append(v)
        self.b.append(rhs)

    def matrix(self) -> csr_matrix:
        n = len(self.b)
        coo = coo_matrix(
            (
                np.asarray(self._vals, dtype=float),
                (np.asarray(self._rows, dtype=int), np.asarray(self._cols, dtype=int)),
            ),
            shape=(n, self._n_cols),
        )
        return cast(csr_matrix, coo.tocsr())


def _build_lp(
    centers: np.ndarray,
    phi: np.ndarray,
    us: np.ndarray,
    Lrel: np.ndarray,
    Hrel: np.ndarray,
    exempt: np.ndarray,
    sigma: float,
    u_T: float,
    dy_T: float,
    mode: str,
) -> tuple[
    csr_matrix,
    np.ndarray,
    csr_matrix,
    np.ndarray,
    list[tuple[float | None, float | None]],
    np.ndarray,
]:
    """Build the CCF LP (5.2.md §6) as sparse ``A_ub/b_ub, A_eq/b_eq``.

    Returns ``(A_ub, b_ub, A_eq, b_eq, bounds, c)`` for ``linprog``. Rows:

    - ``±w_j - t_j <= 0`` (``t_j >= |w_j|``); ``sum_j t_j <= s`` (``s >= ||w||_1``)
    - ``±a - aa <= 0``, ``±b - ab <= 0`` (the affine's separate penalty)
    - per non-exempt sample: ``Lrel_k + m·s <= f(u_k) <= Hrel_k - m·s`` with
      ``m = _margin_coef(sigma, mode)``, slack-absorbed
    - equalities: ``f(0) = 0`` (muzzle), ``f(u_T) = dy_T`` (target)

    The target endpoint is exempt from the margin (destination, not obstacle);
    the sample immediately before it is NOT exempt (§6).
    """
    K, J = phi.shape
    i_s, i_a, i_b, i_slack0, i_aa, i_ab = _lp_layout(J, K)
    n_var = 2 * J + K + 5
    m = _margin_coef(sigma, mode)

    rows = _LPRows(n_var)
    for j in range(J):
        rows.add([(j, 1.0), (J + j, -1.0)], 0.0)  # w_j - t_j <= 0
        rows.add([(j, -1.0), (J + j, -1.0)], 0.0)  # -w_j - t_j <= 0
    rows.add([(J + j, 1.0) for j in range(J)] + [(i_s, -1.0)], 0.0)  # sum t <= s
    rows.add([(i_a, 1.0), (i_aa, -1.0)], 0.0)
    rows.add([(i_a, -1.0), (i_aa, -1.0)], 0.0)
    rows.add([(i_b, 1.0), (i_ab, -1.0)], 0.0)
    rows.add([(i_b, -1.0), (i_ab, -1.0)], 0.0)

    for k_idx in np.nonzero(~exempt)[0]:
        k = int(k_idx)
        phi_row = phi[k]
        nzj = np.nonzero(phi_row)[0]
        uk = float(us[k])
        rows.add(
            [(int(j), -float(phi_row[j])) for j in nzj]
            + [(i_a, -uk), (i_b, -1.0), (i_s, -m), (i_slack0 + k, 1.0)],
            -float(Lrel[k]),
        )  # -f(u_k) - m s + slack_k <= -Lrel_k
        rows.add(
            [(int(j), float(phi_row[j])) for j in nzj]
            + [(i_a, uk), (i_b, 1.0), (i_s, -m), (i_slack0 + k, 1.0)],
            float(Hrel[k]),
        )  # f(u_k) - m s + slack_k <= Hrel_k
    A_ub = rows.matrix()
    b_ub = np.asarray(rows.b, dtype=float)

    # Equalities with the SAME truncated basis (§3: identical to emission).
    eq = _LPRows(n_var)
    phi_end = _phi(np.array([0.0, float(u_T)]), centers, sigma)
    eq.add(
        [(int(j), float(phi_end[0, j])) for j in range(J)] + [(i_b, 1.0)],
        0.0,
    )  # f(0) = 0
    eq.add(
        [(int(j), float(phi_end[1, j])) for j in range(J)] + [(i_a, float(u_T)), (i_b, 1.0)],
        float(dy_T),
    )  # f(u_T) = dy_T
    A_eq = eq.matrix()
    b_eq = np.asarray(eq.b, dtype=float)

    def _bound(lo: float | None, hi: float | None) -> tuple[float | None, float | None]:
        return (lo, hi)

    bounds: list[tuple[float | None, float | None]] = (
        [_bound(None, None) for _ in range(J)]  # w free
        + [_bound(0.0, None) for _ in range(J)]  # t >= 0
        + [_bound(0.0, None)]  # s >= 0
        + [_bound(None, None), _bound(None, None)]  # a, b free
        + [_bound(0.0, None) for _ in range(K)]  # slack >= 0
        + [_bound(0.0, None) for _ in range(2)]  # aa, ab >= 0
    )
    c = np.zeros(n_var)
    c[i_s] = 1.0
    c[i_aa] = c[i_ab] = _MU_AFFINE
    c[i_slack0 : i_slack0 + K] = _LAMBDA_SLACK
    return A_ub, b_ub, A_eq, b_eq, bounds, c


def _solve_lp(
    centers: np.ndarray,
    phi: np.ndarray,
    us: np.ndarray,
    Lrel: np.ndarray,
    Hrel: np.ndarray,
    exempt: np.ndarray,
    sigma: float,
    u_T: float,
    dy_T: float,
    mode: str,
) -> _LPSolution | None:
    """Solve the §6 LP with HiGHS. ``None`` on any non-optimal status."""
    K, J = phi.shape
    i_s, i_a, i_b, i_slack0, _i_aa, _i_ab = _lp_layout(J, K)
    A_ub, b_ub, A_eq, b_eq, bounds, c = _build_lp(
        centers, phi, us, Lrel, Hrel, exempt, sigma, u_T, dy_T, mode
    )
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0 or res.x is None:
        return None
    x = np.asarray(res.x, dtype=float)
    w = x[:J]
    if not np.all(np.isfinite(w)):
        return None
    return _LPSolution(
        w=w,
        s=float(x[i_s]),
        a=float(x[i_a]),
        b=float(x[i_b]),
        slack_total=float(np.sum(x[i_slack0 : i_slack0 + K])),
        status="optimal",
    )


# --- Pruning + re-verification (5.2.md §9) ------------------------------------


def _affine_from_endpoints(
    w: np.ndarray, centers: np.ndarray, sigma: float, u_T: float, dy_T: float
) -> tuple[float, float]:
    """Exactly re-derive ``(a, b)`` from the two endpoint equalities.

    After pruning, the moved curve must still satisfy ``f(0) = 0`` and
    ``f(u_T) = dy_T`` exactly (a residual in ``f(0)`` shifts the whole fired
    curve relative to the corridor and voids the certificate). Two linear
    equations in two unknowns:

        b    = -sum_j w_j phi_j(0)
        a    = (dy_T - sum_j w_j phi_j(u_T) - b) / u_T
    """
    phi_end = _phi(np.array([0.0, float(u_T)]), centers, sigma)
    b = -float(phi_end[0] @ w)
    a = (dy_T - float(phi_end[1] @ w) - b) / u_T
    return a, b


def _recheck_corridor(
    w: np.ndarray,
    a: float,
    b: float,
    centers: np.ndarray,
    sigma: float,
    us: np.ndarray,
    Lrel: np.ndarray,
    Hrel: np.ndarray,
    exempt: np.ndarray,
    mode: str,
    tol: float = _RECHECK_TOL,
) -> tuple[bool, str | None]:
    """Re-check every corridor constraint with the FINAL weights (§9 step 2).

    Returns ``(ok, binding)``; ``binding`` names the tightest bound and u_k.
    ``tol`` is the violation tolerance: the default is exact-arithmetic noise;
    re-checking an LP solution inherits the LP's own feasibility tolerance
    (pass ``_LP_ROW_TOL`` there)."""
    f_vals = _phi(us, centers, sigma) @ w + a * us + b
    s_final = float(np.sum(np.abs(w)))
    m = _margin_coef(sigma, mode) * s_final
    worst = -math.inf
    binding: str | None = None
    for k_idx in np.nonzero(~exempt)[0]:
        k = int(k_idx)
        lo_viol = (float(Lrel[k]) + m) - float(f_vals[k])
        hi_viol = float(f_vals[k]) - (float(Hrel[k]) - m)
        if lo_viol > tol or hi_viol > tol:
            side = "lower" if lo_viol >= hi_viol else "upper"
            return False, f"{side}@u={us[k]:.2f}"
        usage = max(lo_viol, hi_viol)
        if usage > worst:
            worst = usage
            side = "lower" if lo_viol >= hi_viol else "upper"
            binding = f"{side}@u={us[k]:.2f}"
    return True, binding


def _prune_and_reverify(
    w: np.ndarray,
    a: float,
    b: float,
    centers: np.ndarray,
    sigma: float,
    us: np.ndarray,
    Lrel: np.ndarray,
    Hrel: np.ndarray,
    exempt: np.ndarray,
    u_T: float,
    dy_T: float,
    mode: str,
) -> tuple[np.ndarray, float, float]:
    """Prune ``|w_j| < eps`` and RE-VERIFY (5.2.md §9); restore on failure.

    Pruning lowers ``||w||_1`` (smaller margin, good) but MOVES the curve
    (bad): the affine terms are re-derived exactly from the endpoint
    equalities and every corridor constraint is re-checked with the recomputed
    ``s``. On any violation the unpruned LP solution is restored unchanged
    (§9 step 3) — never emit a pruned expression that was not re-verified.
    """
    mask = np.abs(w) >= _PRUNE_EPS
    if bool(np.all(mask)):
        return w, a, b
    # Full-length weight vector with pruned terms zeroed: identical curve to
    # the masked subset (zero terms contribute nothing) and keeps the caller's
    # (w, centers) pairing aligned for emission.
    w2 = np.where(mask, w, 0.0)
    a2, b2 = _affine_from_endpoints(w2, centers, sigma, u_T, dy_T)
    ok, _binding = _recheck_corridor(w2, a2, b2, centers, sigma, us, Lrel, Hrel, exempt, mode)
    if ok:
        return w2, a2, b2
    return w, a, b


# --- Emission (5.2.md §2: balanced tree, plain decimals) ----------------------


def _emit(w: np.ndarray, centers: np.ndarray, sigma: float, a: float, b: float, mx: float) -> str:
    """Emit the certified curve as a Graphwar expression.

    The LP fits the SHOOTER-RELATIVE curve ``f_ccf(u) = sum w_j phi_j(u) +
    a u + b``; the game evaluates in centered world ``x``, so terms are
    re-expressed at ``u = x - mx``: Gaussian centres at ``mx + c_j`` and the
    affine as ``a·x + (b - a·mx)``. The constant is EMITTED (not left to the
    auto-offset): the certificate is about the absolute curve, and the
    auto-offset's job is only to cancel ``f_ccf`` at the nudged muzzle.
    """
    b_coeff = _sigma_b(sigma)
    terms = [
        gauss_term(float(wj), mx + float(cj), b_coeff)
        for wj, cj in zip(w, centers, strict=True)
        if wj != 0.0
    ]
    b2 = b - a * mx
    terms.append(f"({format_literal(a)})*x")
    terms.append(f"({format_literal(b2)})")
    return balanced_sum(terms)


# --- Fired-curve reconstruction (tight-mode certificate) -----------------------


def _fired_curve(expression: str, mx: float, my: float) -> Callable[[float], float] | None:
    """The curve the game actually fires: ``g(u) = f_emitted(mx + u) + offset``.

    Mirrors the muzzle setup of ``process_function_range`` (physics.py:154-206)
    using the REAL launch-angle fixed point on the REAL parsed expression —
    the nudge, the auto vertical offset, everything the game does — with ``u``
    sampled on the caller's grid instead of the integrator's.

    The curve is ABSOLUTE (world ``y`` = ``f_ccf(u) + my + delta``).  Its
    certificate consumer, the tight-mode posthoc gate, therefore compares
    ``g(u) - my`` — the FIRED RELATIVE curve ``f_ccf(u) + delta`` — against the
    SHOOTER-RELATIVE corridor bounds ``Lrel``/``Hrel``; the world-frame value
    is what the physics property test plots against terrain.
    """
    try:
        fn = PolishNotationFunction(expression)
    except MalformedFunction:
        return None
    angle = _get_start_angle(fn, mx, _NUDGE_RADIUS)
    x0, y0 = mx, my
    if not (math.isnan(angle) or math.isinf(angle)):
        x0 += _NUDGE_RADIUS * math.cos(angle)
        y0 += _NUDGE_RADIUS * math.sin(angle)
    offset = -fn.evaluate(x0) + y0
    return lambda u: fn.evaluate(mx + u) + offset


def _posthoc_gate(
    g: Callable[[float], float],
    us: np.ndarray,
    Lrel: np.ndarray,
    Hrel: np.ndarray,
    exempt: np.ndarray,
    m_prime: float,
) -> tuple[bool, str | None]:
    """Tight-mode certificate gate: the MEASURED FIRED RELATIVE curve at every
    corridor boundary sample must satisfy the margin rows; the chord bound
    (``g'' = f''``, ``|g - chord| <= M du^2 / 8``) covers the continuous curve
    between samples. Returns ``(ok, binding)``.

    Frame contract: ``g`` is the fired curve in the SHOOTER-RELATIVE frame —
    ``g(u) = f_ccf(u) + delta`` — and ``Lrel``/``Hrel`` are the same-frame
    corridor bounds.  ``_finalize_solution`` passes ``g_abs(u) - my`` where
    ``g_abs = _fired_curve(...)`` is the world-frame curve, so the gate
    verifies the fired RELATIVE curve against the RELATIVE bounds."""
    worst = -math.inf
    binding: str | None = None
    for k_idx in np.nonzero(~exempt)[0]:
        k = int(k_idx)
        v = g(float(us[k]))
        lo_viol = (float(Lrel[k]) + m_prime) - v
        hi_viol = v - (float(Hrel[k]) - m_prime)
        if lo_viol > _RECHECK_TOL or hi_viol > _RECHECK_TOL:
            side = "lower" if lo_viol >= hi_viol else "upper"
            return False, f"{side}@u={us[k]:.2f}"
        usage = max(lo_viol, hi_viol)
        if usage > worst:
            worst = usage
            side = "lower" if lo_viol >= hi_viol else "upper"
            binding = f"{side}@u={us[k]:.2f}"
    return True, binding


# --- Branch selection (5.2.md §8) ----------------------------------------------


def _branch_paths(
    mx: float,
    my: float,
    tx: float,
    ty: float,
    circles: Sequence[tuple[int, int, int]],
    exclusions: Sequence[tuple[float, float, float]],
    inverted: bool,
    target_index: int,
    k_paths: int,
) -> list[Chain]:
    """K branch paths for one target from a dedicated branch-path DP (5.2.md §8).

    This replaces the M5.1 sweep's backtraced chains as the BRANCH GEOMETRY
    source.  The sweep's union-merge keeps ONE parent per merged interval, so a
    backtraced "chain" splices a top-branch path onto a bottom-branch parent --
    a vertical plunge of ~29 world units inside one cell.  That is slope-feasible
    (``S · du`` exceeds the band height) but chord-uncertifiable, and the
    cell-wise envelope therefore kills every such chain; each target ended
    BASIS_INFEASIBLE/UNCERTIFIED on dense maps.  Reconstructing branch geometry
    independently fixes that.

    The optimisation is EXACT.  Build a layered DAG whose layers are the sample
    columns ``k``; the nodes of layer ``k`` are ``(k, i)`` = free component ``i``
    of ``corridor._column_free(columns[k], mirrored_circles, exclusions)``, the
    components being already merged and disjoint.  An edge ``(k,i) -> (k+1,j)``
    exists exactly when the slope-cap dilation of ``comp_i`` touches ``comp_j``:
    ``dilate(comp_i, S · du) ∩ comp_j ≠ ∅``, with closed-touch
    counted as ``clo <= ahi + d and chi >= alo - d`` for ``comp_i = (alo, ahi)``,
    ``comp_j = (clo, chi)`` and ``d = corridor._SLOPE_CAP * _DU``.  Edge cost is
    the §8 curvature-demand proxy ``du · slope^2`` with
    ``slope = (mid_j - mid_i)/du`` and ``mid`` the component midpoint.  The
    layered structure makes a left-to-right column-sweep DP equivalent to
    Dijkstra on the DAG: every producer state is settled once by the time its
    successors are processed, so the DP is exact, not greedy.

    Terminals are the columns ``k`` with ``|columns[k] - tx| <= WORLD_RADIUS``
    whose component interval intersects the target disk ``[ty - R, ty + R]``
    (the sweep's goal-disk rule); the cheapest finite eligible terminal ends a
    branch, and the per-column parent arrays backtrace into a synthetic
    ``Chain(L=(lo,...), H=(hi,...), goal_column=k)``.  The first layer is
    restricted to components containing the muzzle ``my`` -- the muzzle column
    is free before we are called (``reach.reachable``), but its free set can
    contain several components, and starting in a different one would emit a
    phantom branch.  If no column-0 component contains the muzzle, ``[]`` is
    returned.

    K paths come from re-running the DP with ``+_PATH_PENALTY`` added to every
    node used by an earlier path.  This is a documented DEVIATION from exact
    K-best (recorded in ``docs/OPEN_QUESTIONS.md`` entry (j)): exact K-best
    needs a (path, node) state space that is not worth it at K=4, and the
    penalty picks out the K most diverse cheap branches, which is what the cell
    envelope actually needs.  Emission stops at the first re-run with no finite
    eligible terminal; the returned list is cheapest-first because each re-run's
    chosen terminal is the next-cheapest distinct path.
    """
    circ = [(config.PLANE_LENGTH - cx, cy, r) for cx, cy, r in circles] if inverted else circles
    radius = corridor.WORLD_RADIUS
    d = corridor._SLOPE_CAP * _DU

    # Same column grid as corridor.sweep_target: muzzle -> far edge of the goal
    # disk. Stop at the first fully-blocked column (the sweep already guarantees
    # the muzzle column is free before we are called).
    span = tx - mx + radius
    k_max = int(math.ceil(span / _DU))
    columns: list[float] = []
    comps_by_col: list[list[tuple[float, float]]] = []
    comps0 = corridor._column_free(mx, circ, exclusions)
    if not comps0:
        return []
    columns.append(mx)
    comps_by_col.append(comps0)
    for k in range(1, k_max + 1):
        wx = mx + k * _DU
        comps = corridor._column_free(wx, circ, exclusions)
        if not comps:
            break
        columns.append(wx)
        comps_by_col.append(comps)

    start = [i for i, (lo, hi) in enumerate(comps_by_col[0]) if lo <= my <= hi]
    if not start:
        return []

    used: list[set[int]] = [set() for _ in columns]
    paths: list[Chain] = []
    n_cols = len(columns)

    for _ in range(k_paths):
        dists: list[list[float]] = []
        parents: list[list[int | None]] = []
        row0 = [math.inf] * len(comps_by_col[0])
        for i in start:
            row0[i] = _PATH_PENALTY if i in used[0] else 0.0
        dists.append(row0)
        parents.append([None] * len(comps_by_col[0]))
        for k in range(1, n_cols):
            prev = dists[k - 1]
            prev_comps = comps_by_col[k - 1]
            cur_comps = comps_by_col[k]
            drow = [math.inf] * len(cur_comps)
            prow: list[int | None] = [None] * len(cur_comps)
            for j, (clo, chi) in enumerate(cur_comps):
                jmid = (clo + chi) / 2.0
                best = math.inf
                arg: int | None = None
                for i, (alo, ahi) in enumerate(prev_comps):
                    if clo > ahi + d or chi < alo - d:
                        continue
                    if not math.isfinite(prev[i]):
                        continue
                    imid = (alo + ahi) / 2.0
                    slope = (jmid - imid) / _DU
                    cand = prev[i] + _DU * slope * slope
                    if cand < best:
                        best = cand
                        arg = i
                if best < math.inf and j in used[k]:
                    best += _PATH_PENALTY
                drow[j] = best
                prow[j] = arg
            dists.append(drow)
            parents.append(prow)

        terminals: list[tuple[float, int, int]] = []  # (cost, k, i)
        for k in range(n_cols):
            if abs(columns[k] - tx) > radius:
                continue
            for i, (lo, hi) in enumerate(comps_by_col[k]):
                if lo <= ty + radius and hi >= ty - radius and math.isfinite(dists[k][i]):
                    terminals.append((dists[k][i], k, i))
        if not terminals:
            break

        _, kwin, iwin = min(terminals)
        stack: list[tuple[int, int]] = []
        k, i = kwin, iwin
        while True:
            stack.append((k, i))
            p = parents[k][i]
            if p is None:
                break
            i = p
            k -= 1
        stack.reverse()
        paths.append(
            Chain(
                target_index=target_index,
                L=tuple(comps_by_col[kk][ii][0] for kk, ii in stack),
                H=tuple(comps_by_col[kk][ii][1] for kk, ii in stack),
                goal_column=kwin,
            )
        )
        for kk, ii in stack:
            used[kk].add(ii)

    return paths


def _branch_kind(chain: Chain, u_T: float, dy_T: float) -> str:
    """ "over" | "under": the chain midline at mid-span vs. the straight
    muzzle->target line (the M5.5 ``branch_hint`` vocabulary)."""
    mid = (np.asarray(chain.L, dtype=float) + np.asarray(chain.H, dtype=float)) / 2.0
    idx = len(mid) // 2
    u = idx * _DU
    line = dy_T * (u / u_T) if u_T > 0 else 0.0
    return "over" if mid[idx] > line else "under"


# --- MILP cardinality path (5.2.md §7) -----------------------------------------


def _big_m(Lrel: np.ndarray, Hrel: np.ndarray, dy_T: float, u_T: float) -> float:
    """Big-M for ``|w_j| <= B z_j`` (§7).

    Derivation: any corridor-feasible curve takes values confined to the value
    span ``F = max(|Lrel|, |Hrel|, |dy_T|)`` (the affine's ``a·u`` reach is
    itself pinned to that scale by the corridor rows and the ``f(0) = 0``
    equality). A weight ``|w_j|`` beyond ``B`` can only exist as pure
    cancellation against other weights of like magnitude — doubling ``s`` —
    which the L1 objective never prefers while the affine terms (excluded from
    ``s``, mu-penalized) can express any corridor-spanning trend for ``O(mu)``
    cost. ``B`` therefore cuts no optimal cardinality-constrained solution.
    Conservative by construction.
    """
    F_span = max(
        float(np.max(np.abs(Lrel))) if Lrel.size else 0.0,
        float(np.max(np.abs(Hrel))) if Hrel.size else 0.0,
        abs(dy_T),
    )
    return _BIGM_FACTOR * (F_span + abs(dy_T) + u_T + 1.0)


def _solve_milp(
    centers: np.ndarray,
    phi: np.ndarray,
    us: np.ndarray,
    Lrel: np.ndarray,
    Hrel: np.ndarray,
    exempt: np.ndarray,
    sigma: float,
    u_T: float,
    dy_T: float,
    mode: str,
    j_max: int,
    time_budget: float,
) -> _LPSolution | None:
    """Cardinality-controlled re-solve (§7): ``|w_j| <= B z_j``, ``sum z_j <=
    J_max``, binary ``z``; HiGHS via :func:`scipy.optimize.milp`. Returns the
    solution, a ``timeout`` sentinel on the time limit, or ``None``."""
    K, J = phi.shape
    i_s, i_a, i_b, i_slack0, _i_aa, _i_ab = _lp_layout(J, K)
    n_lp = 2 * J + K + 5
    n_var = n_lp + J  # z appended
    i_z0 = n_lp

    A_lp, b_lp, A_eq, b_eq, bounds_lp, c_lp = _build_lp(
        centers, phi, us, Lrel, Hrel, exempt, sigma, u_T, dy_T, mode
    )
    # Pad the LP blocks to the extended variable count (the z columns are zero
    # for every row that does not reference them).
    pad_ub = csr_matrix((A_lp.shape[0], J))
    A_lp = cast(csr_matrix, hstack([A_lp, pad_ub]).tocsr())
    A_eq = cast(csr_matrix, hstack([A_eq, csr_matrix((A_eq.shape[0], J))]).tocsr())

    big = _big_m(Lrel, Hrel, dy_T, u_T)
    card = _LPRows(n_var)
    for j in range(J):
        card.add([(j, 1.0), (i_z0 + j, -big)], 0.0)
        card.add([(j, -1.0), (i_z0 + j, -big)], 0.0)
    card.add([(i_z0 + j, 1.0) for j in range(J)], float(j_max))
    A_card = card.matrix()
    b_card = np.asarray(card.b, dtype=float)

    c = np.concatenate([c_lp, np.zeros(J)])
    integrality = np.zeros(n_var)
    integrality[i_z0 : i_z0 + J] = 1
    lb = np.array([lo for lo, _hi in bounds_lp] + [0.0] * J, dtype=float)
    ub = np.array([hi for _lo, hi in bounds_lp] + [1.0] * J, dtype=float)

    res = milp(
        c=c,
        constraints=[
            LinearConstraint(A_lp, -np.inf, b_lp),
            LinearConstraint(A_eq, b_eq, b_eq),
            LinearConstraint(A_card, -np.inf, b_card),
        ],
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options={"time_limit": time_budget, "disp": False},
    )
    if res.status == 1:  # time/iteration limit reached
        return _LPSolution(w=np.zeros(J), s=0.0, a=0.0, b=0.0, slack_total=0.0, status="timeout")
    if res.status != 0 or res.x is None:
        return None
    x = np.asarray(res.x, dtype=float)
    w = x[:J].copy()
    z = x[i_z0 : i_z0 + J]
    w[z < 0.5] = 0.0  # enforce the cardinality gate exactly
    if not np.all(np.isfinite(w)):
        return None
    return _LPSolution(
        w=w,
        s=float(x[i_s]),
        a=float(x[i_a]),
        b=float(x[i_b]),
        slack_total=float(np.sum(x[i_slack0 : i_slack0 + K])),
        status="optimal",
    )


# --- Prune-to-fit (the §7 timeout fallback) -------------------------------------


def _prune_to_fit(
    w: np.ndarray,
    a: float,
    b: float,
    centers: np.ndarray,
    sigma: float,
    us: np.ndarray,
    Lrel: np.ndarray,
    Hrel: np.ndarray,
    exempt: np.ndarray,
    u_T: float,
    dy_T: float,
    mode: str,
    mx: float,
    char_limit: int,
) -> tuple[np.ndarray, float, float] | None:
    """Drop the smallest-magnitude terms until the emission fits, RE-VERIFYING
    each candidate subset (§7 fallback: LP solution, pruned, re-verified)."""
    budget = emission_budget(char_limit)
    if budget.j_max < 2:
        return None
    order = np.argsort(-np.abs(w), kind="stable")
    keep = min(int(budget.j_max), int(w.size))
    for _ in range(6):
        idx = np.sort(order[:keep])
        w2 = np.zeros_like(w)
        w2[idx] = w[idx]  # zeroed in place: full length, centers stay aligned
        a2, b2 = _affine_from_endpoints(w2, centers, sigma, u_T, dy_T)
        ok, _binding = _recheck_corridor(w2, a2, b2, centers, sigma, us, Lrel, Hrel, exempt, mode)
        if not ok:
            keep += 1  # pruning moved the curve out of the corridor: keep more
            if keep > int(w.size):
                return None
            continue
        expr = _emit(w2, centers, sigma, a2, b2, mx)
        if len(expr) <= char_limit:
            return w2, a2, b2
        step = max(1, (len(expr) - char_limit) // max(1, budget.per_term_cost))
        keep = max(2, keep - step)
    return None


# --- Per-(sigma, branch) finalization -------------------------------------------


@dataclass
class _Finalized:
    """Result of emitting one LP/MILP solution."""

    candidate: CCFCandidate | None
    emitted_length: int | None
    milp_fired: bool
    binding: str | None


def _finalize_solution(
    lp: _LPSolution,
    *,
    centers: np.ndarray,
    sigma: float,
    us: np.ndarray,
    Lrel: np.ndarray,
    Hrel: np.ndarray,
    exempt: np.ndarray,
    u_T: float,
    dy_T: float,
    my: float,
    mx: float,
    mode: str,
    branch_index: int,
    branch_kind: str,
    target_index: int,
    char_limit: int,
    milp_time_budget: float,
) -> _Finalized:
    """Prune -> re-verify -> emit -> overflow path (MILP §7, prune-to-fit) ->
    tight-mode gate. Returns the candidate or None (with diagnostics)."""
    phi = _phi(us, centers, sigma)
    w, a, b = _prune_and_reverify(
        lp.w, lp.a, lp.b, centers, sigma, us, Lrel, Hrel, exempt, u_T, dy_T, mode
    )
    expr = _emit(w, centers, sigma, a, b, mx)
    milp_fired = False

    if len(expr) > char_limit:
        milp_fired = True
        milp = _solve_milp(
            centers,
            phi,
            us,
            Lrel,
            Hrel,
            exempt,
            sigma,
            u_T,
            dy_T,
            mode,
            j_max=emission_budget(char_limit).j_max,
            time_budget=milp_time_budget,
        )
        if milp is not None and milp.status == "optimal":
            w, a, b = _prune_and_reverify(
                milp.w,
                milp.a,
                milp.b,
                centers,
                sigma,
                us,
                Lrel,
                Hrel,
                exempt,
                u_T,
                dy_T,
                mode,
            )
            expr = _emit(w, centers, sigma, a, b, mx)
        if len(expr) > char_limit:
            # MILP timeout / infeasible / still overflowing (§7): prune the LP
            # solution to fit and RE-VERIFY; never emit unverified.
            fit = _prune_to_fit(
                lp.w,
                lp.a,
                lp.b,
                centers,
                sigma,
                us,
                Lrel,
                Hrel,
                exempt,
                u_T,
                dy_T,
                mode,
                mx,
                char_limit,
            )
            if fit is None:
                return _Finalized(None, len(expr), milp_fired, "emit_overflow")
            w, a, b = fit
            expr = _emit(w, centers, sigma, a, b, mx)
            if len(expr) > char_limit:
                return _Finalized(None, len(expr), milp_fired, "emit_overflow")

    certified = lp.slack_total <= _SLACK_TOLERANCE
    binding: str | None
    if mode == "tight":
        g = _fired_curve(expr, mx, my)
        if g is None:
            return _Finalized(None, len(expr), milp_fired, "unparseable")
        m_prime = (_DU * _DU / (8.0 * sigma * sigma) + _TAIL) * float(np.sum(np.abs(w)))
        gate_ok, binding = _posthoc_gate(lambda u: g(u) - my, us, Lrel, Hrel, exempt, m_prime)
        if not gate_ok:
            return _Finalized(None, len(expr), milp_fired, binding)
    else:
        # Allowance mode: the certificate is analytic (LP + the nudge bound);
        # the corridor re-check reproduces it numerically and yields the
        # tightest binding for the escalation report. Violations within the
        # LP's own feasibility tolerance are not real; anything larger
        # downgrades to UNCERTIFIED rather than reject.
        recheck_ok, binding = _recheck_corridor(
            w, a, b, centers, sigma, us, Lrel, Hrel, exempt, mode, tol=_LP_ROW_TOL
        )
        certified = certified and recheck_ok

    cand = CCFCandidate(
        expression=expr,
        target_index=target_index,
        sigma=sigma,
        branch_index=branch_index,
        branch_kind=branch_kind,
        certified=certified,
        slack_total=lp.slack_total,
        m_bound=float(np.sum(np.abs(w))) / (sigma * sigma),
        mode=mode,
        nonzero_weights=int(np.count_nonzero(w)),
        binding=binding,
    )
    return _Finalized(cand, len(expr), milp_fired, None)


# --- Per-target driver (sigma ladder x branches) --------------------------------


def solve_target(
    mx: float,
    my: float,
    targets: Sequence[tuple[float, float]],
    teammates: Sequence[tuple[float, float]],
    circles: Sequence[tuple[int, int, int]],
    inverted: bool,
    target_index: int,
    *,
    sigma_ladder: Sequence[float] | None = None,
    max_branches: int | None = None,
    char_limit: int | None = None,
    milp_time_budget: float | None = None,
) -> CCFSolution:
    """Run the full CCF search for ONE target (5.2.md §8-§9).

    Sweep (CCF teammate radius) -> K best branches by the §8 layered-DAG DP ->
    sigma ladder coarse->fine (first feasible sigma wins) -> per branch the
    allowance-mode LP, falling back to the tight-mode LP for the same branch.
    """
    t0 = time.perf_counter()
    limit = config.MAX_EXPR_CHARS if char_limit is None else char_limit
    mtb = _MILP_TIME_BUDGET if milp_time_budget is None else milp_time_budget
    k_branches = _MAX_BRANCHES if max_branches is None else max_branches
    ladder = tuple(sigma_ladder) if sigma_ladder is not None else _SIGMA_LADDER

    tx, ty = targets[target_index]
    u_T = tx - mx
    dy_T = ty - my
    trace: list[str] = []
    cert = CCFCertificate(
        outcome=CCFOutcome.BASIS_INFEASIBLE, target_index=target_index, char_limit=limit
    )

    if u_T <= _DU:
        cert.outcome = CCFOutcome.UNREACHABLE
        cert.binding = "target_at_or_behind_muzzle"
        return CCFSolution(target_index, [], cert)

    reach = corridor.sweep_target(
        mx,
        my,
        target_index,
        targets,
        teammates,
        circles,
        inverted,
        du=_DU,
        max_branches=_SWEEP_CHAINS,
        teammate_radius=corridor.CCF_TEAMMATE_RADIUS,
    )
    if not reach.reachable:
        cert.outcome = CCFOutcome.UNREACHABLE
        cert.binding = f"corridor_empty@k={reach.first_blocked_column}"
        return CCFSolution(target_index, [], cert)

    exclusions = corridor.build_exclusions(
        target_index, targets, teammates, corridor.CCF_TEAMMATE_RADIUS
    )
    budget = emission_budget(limit)
    if budget.j_max < 2:
        cert.outcome = CCFOutcome.EMIT_OVERFLOW
        cert.binding = f"j_max={budget.j_max}<2"
        return CCFSolution(target_index, [], cert)

    # Branch selection (§8): the dedicated layered-DAG DP (_branch_paths)
    # yields K shortest branch paths that each stay inside one free-component
    # family — no top-on-bottom parent splice, so no vertical plunge to die
    # cell-wise at a chord-uncertifiable jump. DP order is already
    # cheapest-first (K is passed into _branch_paths), so no ranking pass of
    # the converted branches and no slicing are needed. Cell geometry does
    # not depend on sigma, so this runs ONCE. Only chains that SURVIVE the
    # cell-wise envelope are kept.
    branches: list[tuple[Chain, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    dead_branches = 0
    for chain in _branch_paths(
        mx, my, tx, ty, circles, exclusions, inverted, target_index, k_branches
    ):
        cells = corridor.chain_cells(chain, mx, circles, exclusions, _DU)
        if cells is None:
            dead_branches += 1
            continue
        n_s = len(chain.L)
        us = np.arange(n_s, dtype=float) * _DU
        Lcell, Hcell = np.asarray(cells[0], dtype=float), np.asarray(cells[1], dtype=float)
        if n_s == 1:
            Lsamp, Hsamp = Lcell, Hcell
        else:
            Lsamp = np.array(
                [Lcell[0]] + [max(Lcell[k - 1], Lcell[k]) for k in range(1, n_s - 1)] + [Lcell[-1]]
            )
            Hsamp = np.array(
                [Hcell[0]] + [min(Hcell[k - 1], Hcell[k]) for k in range(1, n_s - 1)] + [Hcell[-1]]
            )
        exempt = np.abs(us - u_T) < 0.5 * _DU
        branches.append(
            (
                chain,
                _branch_kind(chain, u_T, dy_T),
                us,
                Lsamp - my,
                Hsamp - my,
                exempt,
            )
        )
    if dead_branches:
        trace.append(f"cell_wise_dead_branches={dead_branches}")
    cert.branch_count = len(branches)
    if not branches:
        cert.outcome = CCFOutcome.BASIS_INFEASIBLE
        cert.binding = "all_branches_fail_the_cell_wise_envelope"
        cert.solve_seconds = time.perf_counter() - t0
        return CCFSolution(target_index, [], cert)

    had_lp = False
    milp_any = False
    best_cert: tuple[float, CCFCandidate] | None = None
    best_uncert: tuple[float, CCFCandidate] | None = None
    solves = 0
    winner_depth: int | None = None

    for depth, sigma in enumerate(ladder):
        n_centers = max(2, min(int(math.ceil(u_T / sigma)) + 1, budget.j_max))
        centers = np.linspace(0.0, u_T, n_centers)
        sigma_cert: CCFCandidate | None = None
        for bi, (_chain, kind, us, Lrel, Hrel, exempt) in enumerate(branches):
            if solves >= _MAX_SOLVES_PER_TARGET:
                trace.append("solve_cap_reached")
                break
            phi = _phi(us, centers, sigma)

            for mode in ("allowance", "tight"):
                if solves >= _MAX_SOLVES_PER_TARGET:
                    break
                lp = _solve_lp(centers, phi, us, Lrel, Hrel, exempt, sigma, u_T, dy_T, mode)
                solves += 1
                if lp is None:
                    trace.append(f"sigma={sigma:g} branch={bi} {mode} infeasible")
                    continue  # fall through to the tight-mode LP
                had_lp = True
                fin = _finalize_solution(
                    lp,
                    centers=centers,
                    sigma=sigma,
                    us=us,
                    Lrel=Lrel,
                    Hrel=Hrel,
                    exempt=exempt,
                    u_T=u_T,
                    dy_T=dy_T,
                    my=my,
                    mx=mx,
                    mode=mode,
                    branch_index=bi,
                    branch_kind=kind,
                    target_index=target_index,
                    char_limit=limit,
                    milp_time_budget=mtb,
                )
                milp_any = milp_any or fin.milp_fired
                cand = fin.candidate
                if cand is None:
                    trace.append(
                        f"sigma={sigma:g} branch={bi} {mode} rejected binding={fin.binding}"
                    )
                    if mode == "allowance":
                        continue  # try the tight-mode LP for this branch
                    break
                if cand.certified:
                    if best_cert is None or cand.m_bound < best_cert[0]:
                        best_cert = (cand.m_bound, cand)
                    sigma_cert = cand
                    trace.append(
                        f"sigma={sigma:g} branch={bi} {mode} CERTIFIED s={cand.m_bound * sigma * sigma:.3g}"
                    )
                    break  # first certified mode for this branch
                if best_uncert is None or cand.slack_total < best_uncert[0]:
                    best_uncert = (cand.slack_total, cand)
                trace.append(
                    f"sigma={sigma:g} branch={bi} {mode} UNCERTIFIED slack={cand.slack_total:.3g}"
                )
                if mode == "allowance":
                    continue  # a tight LP might still certify this branch
                break
            if sigma_cert is not None:
                break  # certified: no finer search within this sigma
        if sigma_cert is not None:
            winner_depth = depth
            break  # first feasible sigma wins (§9)

    cert.milp_fired = milp_any
    cert.solve_seconds = time.perf_counter() - t0
    cert.trace = tuple(trace[-12:])

    candidates: list[CCFCandidate] = []
    if best_cert is not None:
        candidates.append(best_cert[1])
    if best_uncert is not None:
        candidates.append(best_uncert[1])

    if best_cert is not None:
        win = best_cert[1]
        cert.outcome = CCFOutcome.CERTIFIED
        cert.sigma = win.sigma
        cert.ladder_depth = winner_depth
        cert.branch_index = win.branch_index
        cert.branch_kind = win.branch_kind
        cert.slack_total = win.slack_total
        cert.nonzero_weights = win.nonzero_weights
        cert.emitted_length = len(win.expression)
        cert.binding = win.binding
        return CCFSolution(target_index, candidates, cert)
    if best_uncert is not None:
        win = best_uncert[1]
        cert.outcome = CCFOutcome.UNCERTIFIED
        cert.sigma = win.sigma
        cert.branch_index = win.branch_index
        cert.branch_kind = win.branch_kind
        cert.slack_total = win.slack_total
        cert.nonzero_weights = win.nonzero_weights
        cert.emitted_length = len(win.expression)
        cert.binding = win.binding
        return CCFSolution(target_index, candidates, cert)
    if had_lp:
        cert.outcome = CCFOutcome.EMIT_OVERFLOW
        cert.binding = "emit_overflow_after_milp"
        return CCFSolution(target_index, [], cert)
    cert.outcome = CCFOutcome.BASIS_INFEASIBLE
    cert.binding = "infeasible_at_every_sigma_and_branch"
    return CCFSolution(target_index, [], cert)


def solve_for_frame(
    mx: float,
    my: float,
    targets: Sequence[tuple[float, float]],
    teammates: Sequence[tuple[float, float]],
    circles: Sequence[tuple[int, int, int]],
    inverted: bool,
    *,
    max_targets: int | None = None,
    sigma_ladder: Sequence[float] | None = None,
    max_branches: int | None = None,
    char_limit: int | None = None,
    milp_time_budget: float | None = None,
) -> CCFResult:
    """Run CCF for the nearest reachable targets (5.2.md §8-§9).

    Targets are attempted nearest-first (the ladder's aim convention), capped
    at ``max_targets``. The top-level outcome is the best target outcome by
    the §10 precedence (CERTIFIED > UNCERTIFIED > EMIT_OVERFLOW >
    BASIS_INFEASIBLE > UNREACHABLE); candidates are merged best-first.
    """
    k_targets = _MAX_TARGETS if max_targets is None else max_targets
    order = sorted(range(len(targets)), key=lambda j: targets[j][0] - mx)
    certificates: list[CCFCertificate] = []
    all_candidates: list[CCFCandidate] = []
    for target_index in order[:k_targets]:
        sol = solve_target(
            mx,
            my,
            targets,
            teammates,
            circles,
            inverted,
            target_index,
            sigma_ladder=sigma_ladder,
            max_branches=max_branches,
            char_limit=char_limit,
            milp_time_budget=milp_time_budget,
        )
        certificates.append(sol.certificate)
        all_candidates.extend(sol.candidates)

    certified = sorted((c for c in all_candidates if c.certified), key=lambda c: c.m_bound)
    uncertified = sorted(
        (c for c in all_candidates if not c.certified), key=lambda c: c.slack_total
    )
    merged = (certified + uncertified)[:_MAX_CANDIDATES]

    if certificates:
        top = min(
            (cert.outcome for cert in certificates),
            key=lambda o: _OUTCOME_PRECEDENCE[o],
        )
    else:
        top = CCFOutcome.UNREACHABLE  # no targets
    return CCFResult(outcome=top, candidates=merged, certificates=tuple(certificates))


def solve_for_game(game: object) -> CCFResult:
    """CCF for the current turn of a :class:`~graphwar_sim.state.Game` — the
    seam :func:`graphwar_sim.solver.solve` and the M5.4/M5.5 agents consume.
    """
    fr = corridor.shooter_frame(game)  # type: ignore[arg-type]
    return solve_for_frame(fr.mx, fr.my, fr.targets, fr.teammates, fr.circles, fr.inverted)
