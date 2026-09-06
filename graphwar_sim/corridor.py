"""Corridor sweep (M5.1) + cell-wise envelope (M5.2): reachability and
certification geometry for NORMAL_FUNC shots, shared by ``solver.py`` (the M5.1
pre-check) and ``ccf.py`` (the M5.2 certified corridor fit).

For each enemy target we decide whether **any** legal trajectory can reach it:
a single-valued ``y = f(x)`` curve, monotone in ``x`` (the physics advances
``x`` strictly; physics.py:216-228), whose fired form passes through the
muzzle via the auto vertical offset (GROUND_TRUTH.md §2.4). An empty corridor
is a PROOF that no such curve exists — a generic verdict, not a solver
failure.

M5.1 sweep (pointwise)
----------------------
Working in the shooter-facing world frame (muzzle at ``(mx, my)``, enemies at
larger ``x``), sample columns ``u_k = mx + k*du`` (``du = STEP_SIZE``). At
each column the **free set** ``F_k ⊂ R`` is the union of y-intervals clearing:

- the map bounds (the plane's vertical band, GROUND_TRUTH.md §1.2),
- the terrain (the simulator's crisp circle model, ``make_circle_obstacle``;
  the antialiased Java ovals' edge divergence is documented in state.py),
- the exclusion disks of every same-side soldier except the current shooter
  (the physics skips only the shooter itself, physics.py:252-255, so a shell
  may legally pass through the muzzle's own disk), and
- the disks of *every other enemy* (a clean shot must hit the chosen target
  before any other enemy).

The target's own disk is **not** excluded — entering it is the goal (strict
``<`` hit test, GROUND_TRUTH.md §2.6; a terrain overlap of the disk is
acceptable because the hit test precedes the terrain test, GROUND_TRUTH.md
§2.7, and 5.2.md §4 exempts the target endpoint).

Reachability propagates with a slope cap ``S`` (exact interval arithmetic
over unions of intervals, ``O(K·|F|)``):

    R_0 = {my}
    R_k = (R_{k-1} ⊕ [-S·du, +S·du]) ∩ F_k

The target is reachable iff for some column k inside the target's disk
``R_k`` intersects it.

**Soundness of the slope cap.** ``S·du = 31.62`` game units and the world band
is only ~29.16 tall, so the cap never binds inside the map: the sweep is a
pure geometric threading test. ``S`` is derived from Phase 0 constants — it is
the max |dy/dx| a projectile can exhibit before the trajectory dies. The
reference's adaptive integrator halves each outer step (``STEP_SIZE = 0.01``)
while ``dx² + dy² > FUNC_MAX_STEP_DISTANCE_SQUARED`` and stops the trajectory
when the halving floor ``dx ≤ FUNC_MIN_X_STEP_DISTANCE`` is reached
(GROUND_TRUTH.md §2.5, Constants.java:85-87). A **surviving** trajectory has
every sub-step with ``dx ≥ 1e-5`` and ``|dy| ≤ sqrt(0.001)``, i.e.
``|dy/dx| ≤ sqrt(FUNC_MAX_STEP_DISTANCE_SQUARED) / FUNC_MIN_X_STEP_DISTANCE
≈ 3162.28`` (a slope exceeding that terminates the shot). By the triangle
inequality the same bound holds for the total ``|Δy|`` over any interval of
length ``du``, so the propagation cap is exactly what any surviving shot
satisfies. ``S`` is ``# TUNABLE``: lowering it models a calm-shot envelope.

**What the sweep does NOT certify** (documented, per the ground rules — not a
silent margin):

- The sweep constrains the curve's values **at the sample columns**; between
  samples the physics only tests its own adaptive points, so a curve could in
  principle clip a disk between tested points and survive. The unreachable
  verdict is a proof for *clean* monotone trajectories; the solver only emits
  shots the physics self-verifies, so this gap cannot cause a wrong emission
  (5.2.md §9's property test guards the certified direction at 10x density).
- The launch nudge (GROUND_TRUTH.md §2.3) moves the first tested point by up
  to one game radius (``≈0.4545``) around the muzzle; the sweep starts at the
  muzzle column and treats the muzzle band as free. That band is inside the
  shooter's own disk, which the physics never hit-tests, so the approximation
  is conservative by less than one disk width.
- The M5.1 sweep is pointwise; the M5.2 CCF sharpens it to the cell-wise
  envelope below before any certificate is issued.

M5.2 cell-wise envelope (5.2.md §5)
-----------------------------------
The interpolation bound controls deviation of f from the CHORD between
consecutive samples, so a corridor defined only at sample points certifies
nothing between them. Per cell ``[u_k, u_{k+1}]`` we compute a conservative
envelope ``[Lcell_k, Hcell_k]`` with **exact** obstacle extrema (analytic
circle bands over the cell — no sampling; exclusion disks are constant), then
per sample ``L_k = max(Lcell_{k-1}, Lcell_k)``, ``H_k = min(Hcell_{k-1},
Hcell_k)`` (endpoints use their single adjacent cell). Both chord endpoints
lie in the cell's interval; the interval is convex so the chord lies inside
it; f deviates from the chord by at most ``M·du²/8``; margin at both endpoints
therefore certifies the CONTINUOUS curve across the cell.

Exclusion-disk radii
--------------------
Teammate disks use the Phase 0 hit radius ``SOLDIER_RADIUS`` for the M5.1
sweep (per the ground rule "use the Phase 0 collision radii — do not invent
margins") and the blast-radius-inflated radius ``SOLDIER_RADIUS +
EXPLOSION_RADIUS`` for the CCF cells (5.2.md §5 — a conservative "never graze
a teammate" envelope matching the Shared_core persona rule that teammate
corridors are walls). Other-enemy disks always use ``SOLDIER_RADIUS`` (they
must be *hit*, never grazed). Consequence, recorded in docs/OPEN_QUESTIONS.md:
the inflated radius can in principle exclude a legal safe shot in a corridor
between two teammates (20px Chebyshev spacing vs a 19px exclusion disk); the
solver's self-verification is the final authority either way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    from .state import Game

# --- Tunables (all # TUNABLE — the slope cap default is derived, see header) --

# Slope cap S: max |dy/dx| any surviving trajectory can exhibit, derived from
# the Phase 0 constants (see module docstring). TUNABLE.
_SLOPE_CAP: float = (
    math.sqrt(config.FUNC_MAX_STEP_DISTANCE_SQUARED) / config.FUNC_MIN_X_STEP_DISTANCE
)
# Sample spacing over the shooter->target span. Phase 0: STEP_SIZE.
_DU: float = config.STEP_SIZE
# Cap on the number of distinct corridor chains extracted for the LP (M5.2).
_MAX_BRANCHES: int = 8
# Teammate exclusion radius (world units): Phase 0 SOLDIER_RADIUS (M5.1).
_TEAMMATE_RADIUS: float = config.PLANE_GAME_LENGTH * config.SOLDIER_RADIUS / config.PLANE_LENGTH
# Teammate exclusion radius for the CCF cell envelope (world units): the
# blast-radius-inflated disk of 5.2.md §5 (SOLDIER_RADIUS + EXPLOSION_RADIUS).
_CCF_TEAMMATE_RADIUS: float = (
    config.PLANE_GAME_LENGTH
    * (config.SOLDIER_RADIUS + config.EXPLOSION_RADIUS)
    / config.PLANE_LENGTH
)
# Target hit radius (world units): the Phase 0 SOLDIER_RADIUS hit test.
WORLD_RADIUS: float = config.PLANE_GAME_LENGTH * config.SOLDIER_RADIUS / config.PLANE_LENGTH


# --- Frame helpers (mirror physics.py:154-160 and GROUND_TRUTH.md §1.3/1.4) ---


def plane_x_of(wx: float) -> float:
    """World x -> plane pixel x (GROUND_TRUTH.md §1.4; physics ``_to_plane_x``)."""
    return config.PLANE_LENGTH * wx / config.PLANE_GAME_LENGTH + config.PLANE_LENGTH / 2.0


def world_y_range_for_plane_rows(py_lo: int, py_hi: int) -> tuple[float, float]:
    """Vertical world band spanning plane pixel rows [py_lo, py_hi] inclusive.

    ``y = PLANE_GAME_LENGTH * (-py + PLANE_HEIGHT/2) / PLANE_LENGTH``
    (GROUND_TRUTH.md §1.3, Function.java:193-194).
    """
    y_lo = config.PLANE_GAME_LENGTH * (-py_hi + config.PLANE_HEIGHT / 2.0) / config.PLANE_LENGTH
    y_hi = config.PLANE_GAME_LENGTH * (-py_lo + config.PLANE_HEIGHT / 2.0) / config.PLANE_LENGTH
    return y_lo, y_hi


def map_y_bounds() -> tuple[float, float]:
    """The vertical world band of the plane's pixel rows [0, PLANE_HEIGHT)."""
    return world_y_range_for_plane_rows(0, config.PLANE_HEIGHT - 1)


def _circle_world_y(cy: int) -> float:
    return (config.PLANE_GAME_LENGTH * (-cy + config.PLANE_HEIGHT / 2.0)) / config.PLANE_LENGTH


# An exclusion disk: world (x, y) centre plus world radius.
_Exclusion = tuple[float, float, float]


# --- Exact interval arithmetic over unions -----------------------------------


@dataclass(frozen=True)
class _Iv:
    """One interval of the reachable set at a column, with its chain parent."""

    lo: float
    hi: float
    parent: int | None = None  # index into the previous column's intervals


def _subtract(
    sorted_ivs: Sequence[tuple[float, float]],
    blocks: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """``sorted_ivs`` minus the union of ``blocks`` (both sorted/disjoint)."""
    if not sorted_ivs or not blocks:
        return list(sorted_ivs)
    out: list[tuple[float, float]] = []
    for lo, hi in sorted_ivs:
        cur, end = lo, hi
        for blo, bhi in blocks:
            if bhi < cur:
                continue
            if blo > end:
                break
            if blo > cur:
                out.append((cur, blo))
            cur = max(cur, bhi)
            if cur >= end:
                break
        if cur < end:
            out.append((cur, end))
    return out


def _merge(ints: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping/touching intervals into sorted disjoint intervals."""
    if not ints:
        return []
    out: list[tuple[float, float]] = []
    for lo, hi in sorted(ints):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _dilate_and_intersect(
    prev: Sequence[_Iv], cap: float, free: Sequence[tuple[float, float]]
) -> list[_Iv]:
    """``(prev ⊕ [-cap, +cap]) ∩ free``, keeping parent pointers.

    Each output interval's parent is the first overlapping dilated interval of
    the previous column (ties resolved deterministically). Adjacent pieces
    with the same parent are merged.
    """
    dilated = [(iv.lo - cap, iv.hi + cap, idx) for idx, iv in enumerate(prev)]
    out: list[_Iv] = []
    for flo, fhi in free:
        for dlo, dhi, p in dilated:
            if dhi < flo:
                continue
            if dlo >= fhi:
                break
            lo = max(flo, dlo)
            hi = min(fhi, dhi)
            if lo < hi:
                # Merge geometrically-adjacent fragments regardless of parent.
                # The reachable set is the UNION of these intervals, so the
                # minimal union is the correct geometry. The old condition
                # (same-parent only) let geometrically-identical fragments from
                # different parents accumulate, which fragments the reachable
                # set exponentially (2x per column) whenever the dilation cap
                # spans the whole band — the default cap S*du = 31.62 > band
                # height 29.16, so this hit on any map with a tall terrain
                # blob. Merging to the minimal union (first parent kept as the
                # backtrace representative) is sound and keeps the sweep O(K).
                if out and lo <= out[-1].hi:
                    out[-1] = _Iv(out[-1].lo, max(out[-1].hi, hi), out[-1].parent)
                else:
                    out.append(_Iv(lo, hi, p))
    return out


# --- M5.1 pointwise free sets ------------------------------------------------


def _column_free(
    wx: float,
    circles: Sequence[tuple[int, int, int]],
    exclusions: Sequence[_Exclusion],
) -> list[tuple[float, float]]:
    """The free set F_k at world column ``wx`` (see module docstring)."""
    px = plane_x_of(wx)
    if not (0 <= px < config.PLANE_LENGTH):
        return []  # out of bounds: collidePoint returns True (Obstacle.java:99-103)

    y_min, y_max = map_y_bounds()
    blocks: list[tuple[float, float]] = []
    for cx, cy, r in circles:
        dx = px - cx
        if abs(dx) > r:
            continue
        s = math.sqrt(r * r - dx * dx)
        # Blocked plane rows: integer y with (y-cy)^2 <= s^2  <=>
        # ceil(cy - s) .. floor(cy + s) (crisp model, inclusive).
        py_lo = math.ceil(cy - s)
        py_hi = math.floor(cy + s)
        blo, bhi = world_y_range_for_plane_rows(py_lo, py_hi)
        if bhi < y_min or blo > y_max:
            continue
        blocks.append((max(blo, y_min), min(bhi, y_max)))
    blocked = _merge(blocks)

    free: list[tuple[float, float]] = [(y_min, y_max)]
    free = _subtract(free, blocked)
    for ex, ey, er in exclusions:
        if abs(wx - ex) <= er:
            free = _subtract(free, [(ey - er, ey + er)])
    return _merge(free)


def build_exclusions(
    target_index: int,
    targets: Sequence[tuple[float, float]],
    teammates: Sequence[tuple[float, float]],
    teammate_radius: float,
) -> list[_Exclusion]:
    """Exclusion disks for one target's sweep/cells: teammates (excluding the
    shooter — the caller already removed it) with ``teammate_radius``, every
    other enemy with the Phase 0 hit radius (they must be hit, not grazed)."""
    return [(ex, ey, teammate_radius) for ex, ey in teammates] + [
        (ex, ey, WORLD_RADIUS) for j, (ex, ey) in enumerate(targets) if j != target_index
    ]


# --- M5.2 cell-wise envelope (5.2.md §5) -------------------------------------


def _cell_obstacle_band(
    wx_lo: float,
    wx_hi: float,
    cx: int,
    cy: int,
    r: int,
) -> tuple[float, float] | None:
    """World blocked band ``[lo, hi]`` of a circle over cell ``[wx_lo, wx_hi]``
    — *exact* extrema, no sampling (5.2.md §5) — or None if the circle's pixel
    x-range does not intersect the cell.

    The band half-height ``s = sqrt(r² - (px - cx)²)`` is maximized at the
    pixel x closest to the centre, so ``band = [cy_w - s_max, cy_w + s_max]``
    with ``s_max`` computed analytically from that closest point. The crisp
    terrain model (``make_circle_obstacle``) matches this exactly.
    """
    px_lo, px_hi = plane_x_of(wx_lo), plane_x_of(wx_hi)
    lo, hi = min(px_lo, px_hi), max(px_lo, px_hi)
    if hi < cx - r or lo > cx + r:
        return None
    d = 0.0 if lo <= cx <= hi else min(abs(cx - lo), abs(cx - hi))
    s_max = math.sqrt(max(0.0, r * r - d * d))
    # s_max is in plane pixels; convert to world units to match cy_w.
    s_max_world = s_max * config.PLANE_GAME_LENGTH / config.PLANE_LENGTH
    cy_w = _circle_world_y(cy)
    return (cy_w - s_max_world, cy_w + s_max_world)


def _side_at(band: tuple[float, float], blo: float, bhi: float) -> str | None:
    """Side of ``band`` relative to the branch interval ``[blo, bhi]``:
    ``"below"`` (band under the branch), ``"above"``, or None when the band
    overlaps the branch (cannot happen for a valid chain)."""
    if band[1] <= blo:
        return "below"
    if band[0] >= bhi:
        return "above"
    return None


def _combine_sides(a: str | None, b: str | None) -> str | None:
    """Combine side verdicts at the two cell boundaries. An obstacle below
    (above) the branch at both boundaries is a floor (ceiling); a missing
    boundary verdict (obstacle not present there) inherits the other; any
    disagreement is ``"both"`` (conservative)."""
    if a is None:
        return b
    if b is None:
        return a
    return a if a == b else "both"


def cell_conservative_bounds(
    u_lo: float,
    u_hi: float,
    branch_lo_lo: float,
    branch_lo_hi: float,
    branch_hi_lo: float,
    branch_hi_hi: float,
    circles: Sequence[tuple[int, int, int]],
    exclusions: Sequence[_Exclusion],
) -> tuple[float, float] | None:
    """Conservative free envelope ``[Lcell, Hcell]`` of one corridor cell
    (5.2.md §5): the chord between the branch's endpoints must stay inside the
    envelope across the WHOLE cell.

    ``branch_lo_*``/``branch_hi_*`` are the branch's interval at ``u_lo`` /
    ``u_hi`` (used to classify each obstacle as a floor/ceiling of the
    branch). For every obstacle intersecting the cell we take the exact
    extremal band; below at both boundaries -> floor (raises ``Lcell``), above
    at both -> ceiling (lowers ``Hcell``), undetermined/swapping -> the
    envelope may not contain the band at all (returns None). Returns None when
    no chord can thread the cell.
    """
    y_min, y_max = map_y_bounds()
    bands: list[tuple[float, float, str | None]] = []  # (lo, hi, side)

    for cx, cy, r in circles:
        band = _cell_obstacle_band(u_lo, u_hi, cx, cy, r)
        if band is None:
            continue
        side = _combine_sides(
            _side_at(band, branch_lo_lo, branch_lo_hi),
            _side_at(band, branch_hi_lo, branch_hi_hi),
        )
        bands.append((band[0], band[1], side))
    for ex, ey, er in exclusions:
        if u_hi < ex - er or u_lo > ex + er:
            continue
        band = (ey - er, ey + er)
        side = _combine_sides(
            _side_at(band, branch_lo_lo, branch_lo_hi),
            _side_at(band, branch_hi_lo, branch_hi_hi),
        )
        bands.append((band[0], band[1], side))

    lcell, hcell = y_min, y_max
    for blo, bhi, side in bands:
        if side == "below":
            lcell = max(lcell, bhi)  # floor: chord must stay above its top
        elif side == "above":
            hcell = min(hcell, blo)  # ceiling: chord must stay below its bottom
        elif not (bhi <= lcell or blo >= hcell):
            return None  # side undetermined and the band crosses the envelope
        if lcell >= hcell:
            return None
    return (lcell, hcell)


def chain_cell_bounds(
    chain: Chain,
    mx: float,
    circles: Sequence[tuple[int, int, int]],
    exclusions: Sequence[_Exclusion],
    du: float = _DU,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Per-sample cell-wise corridor bounds from an M5.1 chain (5.2.md §5):
    ``L_k = max(Lcell_{k-1}, Lcell_k)``, ``H_k = min(Hcell_{k-1}, Hcell_k)``
    (endpoints use their single adjacent cell). None if any cell envelope is
    empty (the chain does not survive the cell-wise sharpening).
    """
    L, H = list(chain.L), list(chain.H)
    n = len(L)
    if n == 1:
        return ((L[0],), (H[0],))
    cells: list[tuple[float, float]] = []
    for k in range(n - 1):
        env = cell_conservative_bounds(
            mx + k * du,
            mx + (k + 1) * du,
            L[k],
            H[k],
            L[k + 1],
            H[k + 1],
            circles,
            exclusions,
        )
        if env is None:
            return None
        cells.append(env)
    out_l: list[float] = [cells[0][0]]
    out_h: list[float] = [cells[0][1]]
    for k in range(1, n - 1):
        out_l.append(max(cells[k - 1][0], cells[k][0]))
        out_h.append(min(cells[k - 1][1], cells[k][1]))
    out_l.append(cells[-1][0])
    out_h.append(cells[-1][1])
    return (tuple(out_l), tuple(out_h))


# --- The M5.1 sweep ----------------------------------------------------------


@dataclass(frozen=True)
class Chain:
    """One corridor branch: a contiguous chain of free intervals per column.

    ``L``/``H`` give the interval ``[L_k, H_k]`` selected at column ``u_k`` by
    the sweep for ``k = 0..K`` (``K + 1 == len(L)``). Consumed by the M5.2 CCF
    LP as the corridor constraints.
    """

    target_index: int
    L: tuple[float, ...]
    H: tuple[float, ...]
    goal_column: int  # first column where this chain enters the target disk


@dataclass(frozen=True)
class TargetReachability:
    """Per-target outcome of the corridor sweep."""

    target_index: int
    tx: float
    ty: float
    reachable: bool
    chains: tuple[Chain, ...] = ()
    first_blocked_column: int | None = None  # first column where R_k = ∅


def sweep_target(
    mx: float,
    my: float,
    target_index: int,
    targets: Sequence[tuple[float, float]],
    teammates: Sequence[tuple[float, float]],
    circles: Sequence[tuple[int, int, int]],
    inverted: bool,
    slope_cap: float = _SLOPE_CAP,
    du: float = _DU,
    max_branches: int = _MAX_BRANCHES,
    teammate_radius: float = _TEAMMATE_RADIUS,
) -> TargetReachability:
    """Corridor sweep for one target. See :func:`sweep_targets`.

    ``teammate_radius`` (world units) sizes the teammate exclusion disks
    (M5.1: Phase 0 SOLDIER_RADIUS; CCF: blast-radius-inflated).
    """
    tx, ty = targets[target_index]
    radius = WORLD_RADIUS

    # Columns cover the muzzle -> far edge of the target's disk.
    span = tx + radius - mx
    k_max = int(math.ceil(span / du))
    columns = [mx + k * du for k in range(k_max + 1)]

    exclusions = build_exclusions(target_index, targets, teammates, teammate_radius)
    # Terrain is mirrored for a TEAM2 shooter (physics.py:247-250); mirroring
    # the circle centres once is equivalent.
    circ = [(config.PLANE_LENGTH - cx, cy, r) for cx, cy, r in circles] if inverted else circles

    # Muzzle column must hold the muzzle (soldier placement guarantees it).
    if not any(lo <= my <= hi for lo, hi in _column_free(columns[0], circ, exclusions)):
        return TargetReachability(
            target_index=target_index, tx=tx, ty=ty, reachable=False, first_blocked_column=0
        )

    cols: list[list[_Iv]] = [[_Iv(my, my)]]
    reachable = False
    goal_column = -1
    first_blocked: int | None = None

    for k in range(1, k_max + 1):
        free = _column_free(columns[k], circ, exclusions)
        if not free:
            first_blocked = k
            break
        row = _dilate_and_intersect(cols[k - 1], slope_cap * du, free)
        if not row:
            first_blocked = k
            break
        cols.append(row)
        if not reachable and abs(columns[k] - tx) <= radius:
            for iv in row:
                if iv.lo <= ty + radius and iv.hi >= ty - radius:
                    reachable = True
                    goal_column = k
                    break

    if not reachable:
        return TargetReachability(
            target_index=target_index,
            tx=tx,
            ty=ty,
            reachable=False,
            first_blocked_column=first_blocked,
        )

    # --- Extract chains (contiguous parent paths) up to the cap --------------
    def backtrace(k: int, iv: _Iv) -> Chain:
        stack: list[tuple[int, _Iv]] = []
        cur_k, cur = k, iv
        while True:
            stack.append((cur_k, cur))
            if cur.parent is None:
                break
            cur = cols[cur_k - 1][cur.parent]
            cur_k -= 1
        stack.reverse()
        return Chain(
            target_index=target_index,
            L=tuple(civ.lo for _ck, civ in stack),
            H=tuple(civ.hi for _ck, civ in stack),
            goal_column=k,
        )

    chains: list[Chain] = []
    seen: set[int] = set()
    for k in range(goal_column, k_max + 1):
        if len(chains) >= max_branches:
            break
        if abs(columns[k] - tx) > radius:
            continue
        for iv in cols[k]:
            if iv.lo <= ty + radius and iv.hi >= ty - radius:
                if id(iv) in seen:
                    continue
                seen.add(id(iv))
                chains.append(backtrace(k, iv))

    return TargetReachability(
        target_index=target_index, tx=tx, ty=ty, reachable=True, chains=tuple(chains)
    )


def sweep_targets(
    mx: float,
    my: float,
    targets: Sequence[tuple[float, float]],
    teammates: Sequence[tuple[float, float]],
    circles: Sequence[tuple[int, int, int]],
    inverted: bool,
    slope_cap: float = _SLOPE_CAP,
    du: float = _DU,
    max_branches: int = _MAX_BRANCHES,
    teammate_radius: float = _TEAMMATE_RADIUS,
) -> list[TargetReachability]:
    """Run the corridor sweep for every target in the shooter-facing frame.

    Parameters
    ----------
    mx, my:
        Muzzle in world coords (shooter-facing frame, mirror applied).
    targets:
        Enemy positions in world coords (all with ``x > mx``).
    teammates:
        World coords of same-side soldiers **excluding the current shooter**
        (the physics never hit-tests the shooter itself, physics.py:252-255).
    circles:
        Terrain circles in **plane** pixel coords (the mirror, if any, is
        applied here).
    inverted:
        True for a TEAM2 shooter (mirror about the vertical centre).
    """
    return [
        sweep_target(
            mx,
            my,
            j,
            targets,
            teammates,
            circles,
            inverted,
            slope_cap,
            du,
            max_branches,
            teammate_radius,
        )
        for j in range(len(targets))
    ]


def reachability(game: Game) -> list[TargetReachability]:
    """Corridor sweep for the current turn of a :class:`~graphwar_sim.state.Game`.

    Builds the shooter-facing frame exactly like ``solver._build_frame``
    (mirror + plane->world transform) and runs :func:`sweep_targets`.
    """
    team = game.state.current_team()
    inverted = team.team == config.TEAM2
    shooter = team.current_soldier()

    def w(px: float, py: float) -> tuple[float, float]:
        if inverted:
            px = config.PLANE_LENGTH - px
        return (
            config.PLANE_GAME_LENGTH * (px - config.PLANE_LENGTH / 2.0) / config.PLANE_LENGTH,
            config.PLANE_GAME_LENGTH * (-py + config.PLANE_HEIGHT / 2.0) / config.PLANE_LENGTH,
        )

    mx, my = w(shooter.x, shooter.y)
    targets: list[tuple[float, float]] = []
    teammates: list[tuple[float, float]] = []
    for t in game.state.teams:
        for s in t.soldiers:
            if not s.alive:
                continue
            wx, wy = w(s.x, s.y)
            if t.team == team.team:
                if s is not shooter:
                    teammates.append((wx, wy))
            else:
                targets.append((wx, wy))
    circles: Sequence[tuple[int, int, int]] = getattr(game, "circles", ())
    return sweep_targets(mx, my, targets, teammates, circles, inverted)


__all__ = [
    "Chain",
    "TargetReachability",
    "WORLD_RADIUS",
    "build_exclusions",
    "cell_conservative_bounds",
    "chain_cell_bounds",
    "map_y_bounds",
    "reachability",
    "sweep_target",
    "sweep_targets",
]
