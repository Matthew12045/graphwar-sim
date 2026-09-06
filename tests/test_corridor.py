"""Tests for the M5.1 corridor sweep + M5.2 cell-wise envelope.

The corridor sweep is a *proof* of unreachability: if R_k is empty no clean
monotone-x trajectory can reach the target (see graphwar_sim/corridor.py for
the model and the soundness argument). These tests pin the geometry: walls,
channels, arcs, teammate disks, slope-cap semantics, the goal-disk rule, and
the cell-wise sharpening that the M5.2 certificate builds on.

Geometry helpers: circles are (cx, cy, r) in plane pixels; world
``y = (225 - py) * 50/770``, world ``x = (px - 385) * 50/770``.
"""

from __future__ import annotations

import math

from graphwar_sim import config
from graphwar_sim.corridor import (
    WORLD_RADIUS,
    build_exclusions,
    cell_conservative_bounds,
    chain_cell_bounds,
    map_y_bounds,
    sweep_target,
    sweep_targets,
)

DU = config.STEP_SIZE  # default column spacing (0.01)
DEFAULT_CAP = math.sqrt(config.FUNC_MAX_STEP_DISTANCE_SQUARED) / config.FUNC_MIN_X_STEP_DISTANCE


def _py(wy: float) -> int:
    """Plane row for a world y (inverse of the y transform)."""
    return round(225 - wy * 770 / 50)


def _reach(
    mx: float,
    my: float,
    targets: list[tuple[float, float]],
    teammates: list[tuple[float, float]],
    circles: list[tuple[int, int, int]] | None = None,
    inverted: bool = False,
    slope_cap: float = DEFAULT_CAP,
    du: float = DU,
) -> list[bool]:
    return [
        r.reachable
        for r in sweep_targets(
            mx,
            my,
            targets,
            teammates,
            circles if circles is not None else [],
            inverted,
            slope_cap,
            du,
            8,
            WORLD_RADIUS,
        )
    ]


def _cell_du_bounds(
    mx: float,
    my: float,
    targets: list[tuple[float, float]],
    teammates: list[tuple[float, float]],
    circles: list[tuple[int, int, int]],
    du: float,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Pointwise sweep at coarse du, then the cell-wise (L, H) of the first
    reachable chain — or None if no chain survives the sharpening."""
    results = sweep_targets(
        mx, my, targets, teammates, circles, False, DEFAULT_CAP, du, 8, WORLD_RADIUS
    )
    reachable = [r for r in results if r.reachable]
    if not reachable:
        return None
    chain = reachable[0].chains[0]
    excl = build_exclusions(0, targets, teammates, WORLD_RADIUS)
    return chain_cell_bounds(chain, mx, circles, excl, du)


# --- Basic geometry ----------------------------------------------------------


def test_empty_field_is_reachable() -> None:
    """No terrain, no teammates: the straight line of sight exists."""
    assert _reach(-20.0, 0.0, [(10.0, 0.0)], []) == [True]


def test_full_wall_spanning_the_band_is_unreachable() -> None:
    """A terrain circle spanning the whole vertical band blocks every column
    it covers -> no trajectory exists (a proof, not a fit gap)."""
    y_min, y_max = map_y_bounds()
    half = max(abs(y_min), abs(y_max)) + 1.0
    r_px = int(math.ceil(half * 770 / 50))
    circles = [(385, 225, r_px)]  # huge circle across the whole map
    assert _reach(-20.0, 0.0, [(10.0, 0.0)], [], circles) == [False]


def test_arc_over_a_circle_is_reachable() -> None:
    """A circle between muzzle and target blocks the straight line but the
    sweep must find the over-the-top path (the M2 arc rung's raison d'etre)."""
    # Circle spanning px 300..470 (world x -5.5..+5.5), centre row 225 (y=0);
    # its top is at world y ~ +5.52. Muzzle left below, target right above.
    circles = [(385, 225, 85)]
    assert _reach(-8.0, -10.0, [(12.0, 8.0)], [], circles) == [True]


def test_multiple_circles_merged_exactly() -> None:
    """Overlapping circles merge into one blocked band (exact union)."""
    y_min, y_max = map_y_bounds()
    half = max(abs(y_min), abs(y_max)) + 1.0
    r_px = int(math.ceil(half * 770 / 50))
    circles = [(300, 225, r_px), (400, 225, r_px)]
    assert _reach(-20.0, 0.0, [(10.0, 0.0)], [], circles) == [False]


def test_teammate_disk_blocking_the_only_gap_is_unreachable() -> None:
    """Terrain leaves a narrow gap; a teammate's disk (SOLDIER_RADIUS) covers
    it -> the corridor is empty even though terrain alone would allow it."""
    # A floor circle fills the band below world y = -0.15 and a ceiling circle
    # fills it above +0.15, leaving exactly the narrow gap [-0.15, +0.15] over
    # world x in [-7.2, +7.2] (cx=385, r such that the disk's rim lands on the
    # gap edges).
    pf = 225 - (-0.15) * 770 / 50  # row of world y = -0.15
    pb = 225 - 0.15 * 770 / 50  # row of world y = +0.15
    floor_c = (385, int(round((449 + pf) / 2)), int(round((449 - pf) / 2)))
    ceil_c = (385, int(round(pb / 2)), int(round(pb / 2)))
    circles = [floor_c, ceil_c]
    assert _reach(-20.0, 0.0, [(0.0, 0.0)], [], circles) == [True]
    # A teammate's disk at the target blocks the whole gap.
    assert _reach(-20.0, 0.0, [(0.0, 0.0)], [(0.0, 0.0)], circles) == [False]


def test_target_own_disk_is_the_goal_not_an_obstacle() -> None:
    """A path ending inside the target's disk is a hit: the sweep must count
    it reachable even though the disk sits right at a terrain edge."""
    circles = [(385, 225, 130)]  # large central mound (top at world +8.44)
    top_world = 130 * 50 / 770
    ty = top_world + 0.1
    assert _reach(-20.0, ty - 4.0, [(15.0, ty)], [], circles) == [True]


def test_muzzle_column_blocked_at_k0() -> None:
    """Degenerate: the muzzle column itself holds no free point -> unreachable
    at k=0 (cannot happen for game placement, must still be classified)."""
    assert _reach(-20.0, 0.0, [(10.0, 0.0)], [(-20.0, 0.0)], []) == [False]


def test_slope_cap_semantics() -> None:
    """The slope cap binds only near the muzzle: R expands +/-cap*du per
    column, so after band_height/(cap*du) columns it covers the whole band
    and obstacle geometry alone decides. A tiny cap with a high nearby goal
    must be unreachable; the derived default cap makes the same geometry
    reachable. (With the derived default, cap*du = 31.6 > band height, so the
    cap never binds in real games — enemies are >= 1.3 world units away; the
    sweep degenerates to a geometric threading test, as documented.)"""
    targets = [(-24.9, 12.0)]  # 0.1 world beyond the muzzle, high above it
    r = sweep_target(-25.0, 0.0, 0, targets, [], [], False, 5.0, DU, 8, WORLD_RADIUS)
    assert not r.reachable
    r = sweep_target(-25.0, 0.0, 0, targets, [], [], False, DEFAULT_CAP, DU, 8, WORLD_RADIUS)
    assert r.reachable


def test_default_slope_cap_derived_from_phase0() -> None:
    """S = sqrt(FUNC_MAX_STEP_DISTANCE_SQUARED) / FUNC_MIN_X_STEP_DISTANCE:
    the max |dy/dx| a surviving trajectory can exhibit (physics dies beyond
    it — see corridor docstring)."""
    expected = math.sqrt(config.FUNC_MAX_STEP_DISTANCE_SQUARED) / config.FUNC_MIN_X_STEP_DISTANCE
    assert math.isclose(expected, DEFAULT_CAP, rel_tol=1e-12)


def test_inverted_geometry_mirrors_terrain() -> None:
    """The TEAM2 mirror is folded into the terrain: the same world-frame
    geometry is reachable from either side."""
    circles = [(385, 225, 85)]
    assert _reach(-8.0, -10.0, [(12.0, 8.0)], [], circles, inverted=True) == [True]


# --- M5.2 cell-wise envelope -------------------------------------------------


def test_cell_wise_envelope_catches_a_synthetic_spike() -> None:
    """At a coarse du a pointwise corridor sees free columns on both sides of
    a small circle, but the cell-wise envelope must catch the circle rising
    between the samples. This test fails against a pointwise-only
    implementation (5.2.md §11)."""
    circles = [(300, 225, 3)]  # tiny 3px circle (world r ~ 0.195)
    targets = [(15.0, 0.0)]
    teammates: list[tuple[float, float]] = []
    mx, my = -20.0, 0.0
    # du = 0.5 world (~7.7px): the 3px circle sits strictly between columns.
    du = 0.5
    r = sweep_target(
        mx, my, 0, targets, teammates, circles, False, DEFAULT_CAP, du, 8, WORLD_RADIUS
    )
    assert r.reachable, "pointwise sweep must see the columns free"
    bounds = _cell_du_bounds(mx, my, targets, teammates, circles, du)
    assert bounds is None, "cell-wise envelope must reject the spike"


def test_cell_wise_envelope_matches_pointwise_on_clean_field() -> None:
    """No obstacles: the cell envelope equals the map band (L=Y_MIN)."""
    bounds = _cell_du_bounds(-20.0, 0.0, [(10.0, 0.0)], [], [], DU)
    assert bounds is not None
    y_min, y_max = map_y_bounds()
    L, H = bounds
    assert all(math.isclose(lo, y_min, abs_tol=1e-9) for lo in L)
    assert all(math.isclose(hi, y_max, abs_tol=1e-9) for hi in H)
    assert len(L) == len(H) > 1


def test_cell_envelope_obstacle_side_classification() -> None:
    """A circle below the chord is a floor (raises Lcell), never a ceiling."""
    # Circle (cx=370) with the cell spanning its centre column: the exact
    # band over the cell is [cy_w - r_w, cy_w + r_w] with r_w = 60*50/770.
    r_w = 60 * 50 / 770
    circles = [(370, 225, 60)]
    u_lo, u_hi = -1.0, -0.95  # px 369.6..370.4: contains cx=370
    env = cell_conservative_bounds(u_lo, u_hi, 5.0, 5.0, 5.0, 5.0, circles, [])
    assert env is not None
    y_min, y_max = map_y_bounds()
    lcell, hcell = env
    assert math.isclose(lcell, r_w, abs_tol=1e-9)  # floor at the circle's top
    assert math.isclose(hcell, y_max, abs_tol=1e-9)  # no ceiling in the way
    assert lcell < hcell
    del y_min


def test_cell_envelope_empty_when_obstacle_blocks_every_chord() -> None:
    """An obstacle whose band crosses every possible chord empties the cell."""
    y_min, y_max = map_y_bounds()
    half = max(abs(y_min), abs(y_max)) + 1.0
    r_px = int(math.ceil(half * 770 / 50))
    env = cell_conservative_bounds(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, [(385, 225, r_px)], [])
    assert env is None


def test_cell_envelope_teammate_disk_is_a_floor_or_ceiling() -> None:
    """A teammate disk below the branch is a floor at its exact radius."""
    band_hi = WORLD_RADIUS
    env = cell_conservative_bounds(-1.0, 0.0, 2.0, 3.0, 2.0, 3.0, [], [(-0.5, 0.0, WORLD_RADIUS)])
    assert env is not None
    lcell, _hcell = env
    assert math.isclose(lcell, band_hi, abs_tol=1e-9)


def test_chain_cell_bounds_endpoint_rule() -> None:
    """Per-sample L_k = max(Lcell_{k-1}, Lcell_k) (endpoints use one cell)."""
    from graphwar_sim.corridor import Chain

    chain = Chain(target_index=0, L=(-1.0, 0.0, 1.0), H=(0.0, 1.0, 2.0), goal_column=2)
    # No obstacles: every cell envelope is the full band, so the per-sample
    # bounds equal the band.
    bounds = chain_cell_bounds(chain, -20.0, [], [], DU)
    assert bounds is not None
    y_min, y_max = map_y_bounds()
    L, H = bounds
    assert (y_min, y_min, y_min) == L
    assert (y_max, y_max, y_max) == H
