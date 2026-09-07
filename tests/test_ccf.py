"""M5.2 section-11 unit/regression tests for CCF (graphwar_sim/ccf.py).

What this file pins (each assertion cites the spec line in its comments):

- §4 curvature bound: the emitted Gaussian sum's ``max|f''|`` is bounded by
  ``sigma^-2 * ||w||_1`` (the analytic bound of the module is validated
  numerically through the REAL parser and formatter).
- §2 emission budget: ``J_max`` is derived from ``config.MAX_EXPR_CHARS`` and
  ``config.MAX_AST_DEPTH`` (M5.3) through the REAL formatter, and an actual
  ``ccf._emit`` with ``J_max`` Gaussian weights + nonzero affine fits under
  the caps.
- §11 degenerate inputs: muzzle-at-or-behind target, single-sample span,
  corridor empty at k=0, target inside terrain, sigma ladder exhausted
  (BASIS_INFEASIBLE), and the MILP timeout path (no crash + the ``timeout``
  zero-weight sentinel contract).
- §11 no-scientific-notation regression on candidate expressions.
- §11 grep regression: no ``eval``/``exec``/dynamic import in ccf.py or
  corridor.py, and no ``cvxpy`` import anywhere in graphwar_sim/.

EXCLUSIONS honoured (per the task): NO test asserts the tight-mode posthoc
gate's shooter-relative frame contract, and NO test depends on the tight-mode
certificate RATE (no outcome-distribution assertions on real maps). The
real-map test below only asserts the notation property, which is
fix-independent.

Construction policy: every synthetic frame is seeded/exact, and where the
assignment's illustrative construction turned out not to produce the intended
verdict, the construction (not the source) was adjusted and the deviation is
documented inline.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from graphwar_sim import ccf, config, emission
from graphwar_sim.parser import PolishNotationFunction
from graphwar_sim.solver import RUNG_CCF, solve
from graphwar_sim.state import Game


def _du() -> float:
    """The corridor's column spacing (Phase 0 STEP_SIZE)."""
    return config.STEP_SIZE


def _ppn(expr: str) -> PolishNotationFunction:
    """Parse an expression with the REAL parser, raising on malformed input."""
    return PolishNotationFunction(expr)


def _curvature_bound_check(w, sigma, centers) -> None:
    """Assert max|f''| (numeric, central second differences through the real
    parser) <= sigma^-2 * ||w||_1 + a justified tolerance (5.2.md §4).

    Emitted with affine terms identically zero (a=0, b=0): only the Gaussian
    sum, so the bound is exactly the module's ``M <= sigma^-2 * s``.

    Grid: ``[c_min - 5 sigma, c_max + 5 sigma]`` at step ``h = sigma / 20``,
    i.e. >= 20 grid points per sigma so every bump and its curvature peak are
    resolved. The central-second-difference estimator's truncation error is
    ``(h^2/12) * max|f''''| ~ (sigma^2/400/12) * const/sigma^4 = O(||w||_1/sigma^2
    / 4800)`` per bump, so 2% of the bound is a generous slack; the small
    absolute term keeps the case ``||w||_1 ~ 0`` deterministic.
    """
    b_coeff = 1.0 / (2.0 * sigma * sigma)
    terms = [
        emission.gauss_term(float(wi), float(cj), b_coeff)
        for wi, cj in zip(w, centers, strict=True)
    ]
    expr = emission.balanced_sum(terms)
    f = _ppn(expr)

    lo = min(centers) - 5.0 * sigma
    hi = max(centers) + 5.0 * sigma
    h = sigma / 20.0  # TUNABLE grid step: >= 20 pts/sigma
    n = int(math.ceil((hi - lo) / h)) + 1
    fv = np.asarray([f.evaluate(float(u)) for u in np.linspace(lo, hi, n)], dtype=float)
    d2 = (fv[2:] - 2.0 * fv[1:-1] + fv[:-2]) / (h * h)
    est = float(np.max(np.abs(d2)))

    bound = float(sum(abs(float(wi)) for wi in w)) / (sigma * sigma)
    tol = 0.02 * bound + 1e-9  # TUNABLE estimator slack; see helper docstring
    assert est <= bound + tol, (
        f"sigma={sigma}: numeric max|f''|={est:.6e} exceeds analytic "
        f"sigma^-2 * ||w||_1 = {bound:.6e}"
    )


# --- §4 curvature bound, numeric vs analytic ---------------------------------


def test_curvature_bound_numeric_matches_analytic() -> None:
    """max|f''| of a parsed Gaussian sum never exceeds sigma^-2 * ||w||_1.

    Four seeded (w, sigma) combos: two short ones with mixed signs, a many-term
    case, and a coarse-sigma negative-weight pair (5.2.md §4).
    """
    # Short, mixed signs (5.2.md §4: |w_j| enter ||w||_1 regardless of sign).
    _curvature_bound_check([1.2, -0.7, 0.9, -1.1], 0.5, [1.0, 3.0, 5.0, 7.0])
    _curvature_bound_check([2.0, -1.5, 0.6], 2.0, [0.5, 4.5, 8.5])
    # Many-term case: 24 centers, seeded alternating-sign weights (dense basis).
    rng = np.random.default_rng(3)  # seeded; deterministic
    centers_many = np.linspace(0.0, 6.0, 24)
    w_many = 1.5 * rng.uniform(-1.0, 1.0, centers_many.size)
    _curvature_bound_check(w_many, 0.25, centers_many)
    # Coarse sigma, negative-weight pair.
    _curvature_bound_check([-1.3, 0.9], 4.0, [2.0, 6.0])


# --- §2 emission budget ------------------------------------------------------


def test_emission_budget_reproduced() -> None:
    """J_max is derived from config.MAX_EXPR_CHARS and config.MAX_AST_DEPTH
    through the REAL formatter.

    (a) reproduce the arithmetic on BOTH dimensions (5.2.md §2 step 2: the
    test reads the limits from config so it breaks if either changes);
    (b) sanity floor; (c) an actual ccf._emit with J_max nonzero Gaussian
    weights + nonzero affine fits under the char cap AND parses under the
    depth cap (breaks if either limit or the formatter changes).
    """
    budget = ccf.emission_budget()
    char_limit = config.MAX_EXPR_CHARS
    depth_limit = config.MAX_AST_DEPTH
    sigma_rep = ccf._SIGMA_LADDER[0]
    per = len(emission.gauss_term(-1.0, -12.3456, ccf._sigma_b(sigma_rep))) + 3
    affine = (
        len(f"({emission.format_literal(-1.234567890123)})*x")
        + len(f"({emission.format_literal(-12.345678901234)})")
        + 6
    )
    # (a) the derived quantities equal the arithmetic the module performs.
    assert budget.char_limit == char_limit
    assert budget.depth_limit == depth_limit
    assert budget.per_term_cost == per
    assert budget.affine_cost == affine
    assert budget.safety == ccf._EXPR_CHARS_SAFETY
    j_chars = max(0, (char_limit - affine - ccf._EXPR_CHARS_SAFETY) // per)
    j_depth = 1 << max(0, depth_limit - 3) if depth_limit >= 3 else 0
    assert budget.j_max_chars == j_chars
    assert budget.j_max_depth == j_depth
    assert budget.j_max == min(j_chars, j_depth)
    # At the shipped values the char limit binds (the depth dimension is
    # astronomically looser); the consumers below rely on the binding min.
    assert j_depth >= j_chars
    # (b) sanity floor on how many Gaussian terms the budget admits.
    assert budget.j_max >= 8, f"J_max={budget.j_max} implausibly small"  # TUNABLE sanity

    # (c) an actual emission at the budgeted cardinality fits under the cap.
    # Magnitudes mirror the budget's representative term (negative weight,
    # negative centre, coarsest-ladder sigma) so the per-term cost equals the
    # measured one; balanced-tree/affine wrappers are covered by the +3/+6
    # accounts. This breaks if MAX_EXPR_CHARS, MAX_AST_DEPTH, or the formatter
    # changes.
    centers = np.asarray([-12.3456 + 0.01 * i for i in range(budget.j_max)], dtype=float)
    w = np.full(budget.j_max, -1.0)
    expr = ccf._emit(w, centers, sigma_rep, -1.234567890123, -12.345678901234, mx=0.0)
    assert len(expr) <= char_limit, (
        f"J_max={budget.j_max} emission length {len(expr)} exceeds MAX_EXPR_CHARS={char_limit}"
    )
    _ppn(expr)  # the emitted expression must parse (incl. the depth cap)


# --- M5.3: solver/CCF emissions sit inside the AST depth cap with margin ------


def test_solver_emissions_within_ast_depth_cap() -> None:
    """Every expression the ladder emits parses AND evaluates at a tree depth
    well under ``config.MAX_AST_DEPTH``; CCF output specifically stays <= 16
    (real emissions peak at depth ~9 = ceil(log2 J_max) + 3, so 16 leaves
    ~7x headroom under the 64 cap). One seed per final rung of the M5.2
    per-seed battery (REPORT.md): 1 ccf, 2 SOLVER_FAILED, 3 fixed_grid,
    4 arc, 5 parabola, 9 per_target_gaussian, 38 line.
    """
    for seed in (1, 2, 3, 4, 5, 9, 38):  # TUNABLE battery: one seed per rung
        res = solve(Game.create(seed, num_soldiers=2))
        f = _ppn(res.expression)  # the parser's own guard accepts it
        assert f.depth() <= config.MAX_AST_DEPTH, f"seed={seed}: depth {f.depth()}"
        if res.rung == RUNG_CCF:
            assert f.depth() <= 16, f"seed={seed}: CCF emission at depth {f.depth()}"


def test_ccf_candidates_within_depth_margin() -> None:
    """Every candidate CCF produces (not only the certified one) parses and
    stays <= 16 deep — the whole candidate list is surfaced to the oracle."""
    for seed in (1, 10):
        res = ccf.solve_for_game(Game.create(seed, num_soldiers=2))
        assert res.candidates, f"seed={seed}: no candidates to check"
        for cand in res.candidates:
            assert _ppn(cand.expression).depth() <= 16, f"seed={seed}"


# --- §11 degenerate inputs ---------------------------------------------------


def test_target_at_or_behind_muzzle_unreachable() -> None:
    """Target not strictly ahead of the muzzle: UNREACHABLE, binding names the
    muzzle, no candidates (solve_target's u_T <= du guard)."""
    sol = ccf.solve_target(0.0, 0.0, [(0.0, 0.0)], [], [], False, 0)
    cert = sol.certificate
    assert cert.outcome == ccf.CCFOutcome.UNREACHABLE
    assert "target_at_or_behind_muzzle" in cert.binding
    assert sol.candidates == []


def test_single_sample_span_is_legal() -> None:
    """Target barely beyond the muzzle (+du): no crash, a legal CCFOutcome, and
    every candidate parses."""
    mx, my = -20.0, 0.0
    tx = mx + 1.5 * _du()  # just over one column
    sol = ccf.solve_target(mx, my, [(tx, my)], [], [], False, 0)
    assert sol.certificate.outcome in (
        ccf.CCFOutcome.CERTIFIED,
        ccf.CCFOutcome.UNCERTIFIED,
        ccf.CCFOutcome.EMIT_OVERFLOW,
        ccf.CCFOutcome.BASIS_INFEASIBLE,
        ccf.CCFOutcome.UNREACHABLE,
    )
    for cand in sol.candidates:
        _ppn(cand.expression)  # any candidate must parse


def test_corridor_empty_at_k0_is_unreachable() -> None:
    """A terrain circle whose crisp cells block the whole muzzle column ->
    sweep fails at column 0 -> UNREACHABLE with corridor_empty@k=0."""
    # A circle (plane px) centred on the muzzle column (world x=-20 -> px=77)
    # with radius reaching every row, so the muzzle column holds no free point
    # (same construction family as test_muzzle_column_blocked_at_k0).
    circles = [(77, 225, 225)]
    sol = ccf.solve_target(-20.0, 0.0, [(10.0, 0.0)], [], circles, False, 0)
    assert sol.certificate.outcome == ccf.CCFOutcome.UNREACHABLE
    assert "corridor_empty@k=0" in sol.certificate.binding
    assert sol.candidates == []


def test_target_inside_terrain_is_unreachable() -> None:
    """Target inside a terrain circle that fully blocks its column corridor ->
    UNREACHABLE (no free column near the goal)."""
    # Big circle centred at world x=10 (px 539), radius spanning the whole
    # band; muzzle column (world x=-20) stays free, so the sweep dies only at
    # the blocked columns around the target.
    circles = [(539, 225, 225)]
    sol = ccf.solve_target(-20.0, 0.0, [(10.0, 0.0)], [], circles, False, 0)
    assert sol.certificate.outcome == ccf.CCFOutcome.UNREACHABLE
    assert sol.candidates == []


def test_sigma_ladder_exhausted_is_basis_infeasible() -> None:
    """A corridor that is pointwise reachable but admits no branch: BASIS_INFEASIBLE
    deterministically, fix-independent of the tight-mode gate.

    Deviation from the assignment's illustrative "two thin slabs" construction,
    documented: a sub-pixel horizontal gap collapses in the crisp pixel-cell
    model to `corridor_empty` (UNREACHABLE, not BASIS_INFEASIBLE) — verified
    empirically. Thin reachable corridors are instead always affine-fit
    (s == 0) at sigma=(0.25,), so no margin/curvature mechanism makes the LP
    infeasible. The deterministic synthetic case below instead drives CCF to
    `all_branches_fail_the_cell_wise_envelope` (ccf.py:1267-1271): the cell-wise
    envelope rejects every branch before any LP runs, so the outcome is
    BASIS_INFEASIBLE regardless of the concurrent tight-mode gate.
    """
    mx = -20.0
    my = -1.6341220574798339
    tx = -12.259210677456277  # u_T = 7.74 ahead of the muzzle
    ty = 0.943753265492449
    circles = [(114, 154, 94), (107, 279, 37), (455, 170, 74), (264, 176, 42)]
    sol = ccf.solve_target(mx, my, [(tx, ty)], [], circles, False, 0, sigma_ladder=(0.25,))
    assert sol.certificate.outcome == ccf.CCFOutcome.BASIS_INFEASIBLE
    assert "all_branches_fail_the_cell_wise_envelope" in sol.certificate.binding
    assert sol.candidates == []


def test_milp_timeout_no_crash() -> None:
    """A near-zero MILP budget (monkeypatched) never crashes solve_target and
    yields a legal non-UNREACHABLE verdict (§7)."""
    old = ccf._MILP_TIME_BUDGET
    ccf._MILP_TIME_BUDGET = 1e-6  # TUNABLE: force the timeout window
    try:
        sol = ccf.solve_target(-20.0, 0.0, [(10.0, 0.0)], [], [], False, 0, char_limit=300)
    finally:
        ccf._MILP_TIME_BUDGET = old
    assert sol.certificate.outcome in (
        ccf.CCFOutcome.CERTIFIED,
        ccf.CCFOutcome.UNCERTIFIED,
        ccf.CCFOutcome.EMIT_OVERFLOW,
        ccf.CCFOutcome.BASIS_INFEASIBLE,
    )


def test_solve_milp_timeout_sentinel_has_zero_weights() -> None:
    """ccf._solve_milp maps a HiGHS time-limit status to the zero-weight
    'timeout' sentinel (§7).

    With ``time_budget=0.0`` HiGHS errors (status 2, "model error") rather than
    reporting a time limit on this platform, so we drive the mapping directly by
    mocking the underlying ``milp`` to report a time-limit status. This tests
    the sentinel's field contract deterministically without depending on
    solver timing.
    """
    centers = np.linspace(0.0, 2.0, 3)
    phi = np.zeros((3, 3))
    us = np.asarray([0.0, 1.0, 2.0])
    L = np.asarray([0.0, 0.0, 0.0])
    H = np.asarray([0.0, 0.0, 0.0])
    exempt = np.zeros(3, dtype=bool)

    original_milp = ccf.milp

    class _TimedOut:
        status = 1  # HiGHS time/iteration limit reached
        x = None

    ccf.milp = lambda **_: _TimedOut()  # type: ignore[assignment]
    try:
        sol = ccf._solve_milp(
            centers,
            phi,
            us,
            L,
            H,
            exempt,
            sigma=0.5,
            u_T=2.0,
            dy_T=0.0,
            mode="tight",
            j_max=7,
            time_budget=0.0,
        )
    finally:
        ccf.milp = original_milp
    assert sol is not None
    assert sol.status == "timeout"
    assert np.all(sol.w == 0.0)
    assert sol.s == 0.0 and sol.a == 0.0 and sol.b == 0.0 and sol.slack_total == 0.0


# --- §11 no scientific notation (regression; SLOW: ~15-30 s; kept LAST) ------

_EXPONENT_RE = re.compile(r"[0-9](?:\.[0-9]*)?[eE][+-]?[0-9]")
_NUM_MAPS = 10  # TUNABLE: real maps to sweep (cost ~1.4 s/map)


def test_no_scientific_notation_in_emissions() -> None:
    """No candidate expression from real maps contains an exponent.

    The parser rewrites every ``-`` to ``+-`` (unary negation), which corrupts
    scientific notation (``1e-06`` -> ``1e+-06``), so a CCF emission must never
    contain an exponent (5.2.md §2/§11). Slow: runs the full CCF resolve on
    ``_NUM_MAPS`` seeded real maps.
    """
    for seed in range(1, _NUM_MAPS + 1):
        game = Game.create(seed, num_soldiers=2)
        res = ccf.solve_for_game(game)
        for cand in res.candidates:
            assert not _EXPONENT_RE.search(cand.expression), (
                f"seed={seed}: exponent in emitted expression {cand.expression!r}"
            )
    # The formatter itself is plain-decimal even for tiny literals.
    assert "e" not in emission.format_literal(1e-9).lower()
    assert "e" not in emission.format_literal(-1.7e-8).lower()


# --- §11 grep regressions -----------------------------------------------------


def _graph_war_sim_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "graphwar_sim"


def test_grep_no_eval_exec_dynamic_import() -> None:
    """The corridor/CCF source never calls eval/exec or dynamic import (5.2.md
    §11; the module docstring's clean-room security rule)."""
    import graphwar_sim.ccf as ccf_mod
    import graphwar_sim.corridor as corr_mod

    for mod in (ccf_mod, corr_mod):
        src = Path(mod.__file__).read_text()
        for pat in (r"\beval\s*\(", r"\bexec\s*\(", r"importlib", r"__import__"):
            assert re.search(pat, src) is None, f"{mod.__name__} matches {pat!r}"


def test_grep_no_cvxpy_import() -> None:
    """No cvxpy dependency anywhere in graphwar_sim/ (§11; 5.2.md: "no cvxpy
    import anywhere in graphwar_sim/").

    Note: the literal token 'cvxpy' DOES appear in ccf.py/solver.py docstrings
    as part of clean-room citations ("No cvxpy — ..."), so asserting the bare
    token never appears would be false against the current source. We assert
    the spec's actual requirement — cvxpy is never imported or used as a
    package anywhere in the package source.
    """
    files = sorted(p for p in _graph_war_sim_dir().glob("*.py"))
    assert files, "no graphwar_sim sources found"
    for p in files:
        src = p.read_text()
        for m in re.finditer(r"(?m)^\s*(?:import|from)\s+cvxpy\b", src):
            raise AssertionError(f"{p.name}:{m.start()}: cvxpy imported/used (forbidden by §11)")
