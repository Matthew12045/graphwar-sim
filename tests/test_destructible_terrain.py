"""Destructible terrain (M-terrain): blast craters + carve-aware corridor/CCF.

Every shot ends in a blast that carves an EXPLOSION_RADIUS crater out of the
terrain grid (GameData.java:1020-1032, Obstacle.explodePoint); craters persist
for the match. Corridor/CCF subtract carve disks from circle-blocked rows.
"""

from __future__ import annotations

import math

from graphwar_sim import config
from graphwar_sim.corridor import (
    _column_free,
    build_exclusions,
    chain_cells,
    plane_x_of,
    reachability,
    shooter_frame,
)
from graphwar_sim.physics import Obstacle, Soldier, _java_int_cast
from graphwar_sim.state import Game, GameState, Team, make_circle_obstacle


def _flat_game(
    circles: list[tuple[int, int, int]],
    shooter: tuple[float, float] = (50.0, 225.0),
    target: tuple[float, float] = (720.0, 225.0),
    turn: int = 0,
) -> Game:
    teams = [
        Team(name="t1", team=1, soldiers=[Soldier(x=shooter[0], y=shooter[1], alive=True)]),
        Team(name="t2", team=2, soldiers=[Soldier(x=target[0], y=target[1], alive=True)]),
    ]
    return Game(
        state=GameState(teams=teams, current_turn=turn),
        terrain=make_circle_obstacle(circles),
        circles=list(circles),
    )


def test_fire_carves_crater_at_terrain_impact() -> None:
    wall = [(385, 225, 5)]
    g = _flat_game(wall)
    assert g.carves == []
    r = g.fire("0*x")
    assert r.hits == []
    ex, ey = _java_int_cast(r.last_x), _java_int_cast(r.last_y)
    assert g.carves == [(ex, ey, config.EXPLOSION_RADIUS)]
    assert config.EXPLOSION_RADIUS == 12
    # Carve center is always cleared; the wall itself is breached at its core.
    assert g.terrain.collide_point(ex, ey) is False
    assert g.terrain.collide_point(385, 225) is False


def test_team2_carve_mirrors_x() -> None:
    wall = [(385, 225, 5)]
    g = _flat_game(wall, turn=1)
    r = g.fire("0*x")
    expected_x = config.PLANE_LENGTH - _java_int_cast(r.last_x)
    expected_y = _java_int_cast(r.last_y)
    assert g.carves[0][0] == expected_x
    assert g.carves[0][1] == expected_y
    assert g.terrain.collide_point(expected_x, expected_y) is False


def test_crater_opens_a_corridor() -> None:
    # Radius-5 wall (11px thick): one sacrificial flat shot digs it open.
    wall = [(385, 225, 5)]
    g = _flat_game(wall)
    first = g.fire("0*x")
    assert first.hits == []
    second = g.fire("0*x")
    assert second.num_steps > first.num_steps
    assert len(second.hits) == 1  # passes the wall and kills
    assert len(g.carves) == 2


def test_carve_happens_on_soldier_hit() -> None:
    g = _flat_game([])
    r = g.fire("0*x")
    assert len(r.hits) == 1
    assert len(g.carves) == 1
    ex, ey, er = g.carves[0]
    assert (ex, ey) == (_java_int_cast(r.last_x), _java_int_cast(r.last_y))
    assert er == config.EXPLOSION_RADIUS


def test_off_field_shot_carves_clipped() -> None:
    g = _flat_game([])
    r = g.fire("10*x")  # steep: exits the top band
    assert len(g.carves) == 1
    # No crash; interior grid unchanged (empty field stays empty inside).
    assert g.terrain.collide_point(385, 225) is False
    assert g.terrain.collide_point(100, 100) is False
    _ = r


def test_carve_raises_on_gridless_obstacle() -> None:
    def collide_point(x: int, y: int) -> bool:
        return x < 0 or x >= 770 or y < 0 or y >= 450

    obs = Obstacle(collide_point=collide_point)
    assert obs.grid is None
    try:
        obs.carve(100, 100, 12)
    except ValueError:
        pass
    else:
        raise AssertionError("carve on grid-less obstacle must raise")


def test_column_free_agrees_with_pixel_grid() -> None:
    circles = [(200, 225, 40), (500, 150, 30)]
    g = _flat_game(circles)
    # Hand-dug craters (no firing needed for the geometry cross-check).
    carve_list = [(200, 225, 12), (500, 150, 12), (300, 300, 12)]
    for ex, ey, er in carve_list:
        g.terrain.carve(ex, ey, er)
        g.carves.append((ex, ey, er))
    # Sampled world columns across the plane.
    for px in range(0, 770, 37):
        wx = config.PLANE_GAME_LENGTH * (px - config.PLANE_LENGTH / 2.0) / config.PLANE_GAME_LENGTH
        # NOTE: plane_x_of inverts this exactly; use it directly.
        wx = (px - config.PLANE_LENGTH / 2.0) * config.PLANE_GAME_LENGTH / config.PLANE_LENGTH
        free = _column_free(wx, circles, [], carve_list)
        pc = int(plane_x_of(wx))
        if not (0 <= pc < config.PLANE_LENGTH):
            assert free == []
            continue
        for py in range(0, config.PLANE_HEIGHT, 7):
            blocked = g.terrain.collide_point(pc, py)
            # World midpoint of row py's pixel cell.
            y_top = (
                config.PLANE_GAME_LENGTH * (-py + config.PLANE_HEIGHT / 2.0) / config.PLANE_LENGTH
            )
            y_bot = (
                config.PLANE_GAME_LENGTH
                * (-(py + 1) + config.PLANE_HEIGHT / 2.0)
                / config.PLANE_LENGTH
            )
            mid = (y_top + y_bot) / 2.0
            inside = any(lo < mid < hi for lo, hi in free)
            # Row-cell midpoints never sit exactly on a free boundary (boundaries
            # align to row edges), so blocked <=> not inside, exactly.
            assert inside != blocked, f"px={px} pc={pc} py={py} blocked={blocked} free={free}"


def _stacked_wall() -> list[tuple[int, int, int]]:
    return [(385, y, 12) for y in range(0, 450, 5)]


def test_sweep_reaches_through_crater() -> None:
    wall = _stacked_wall()
    teams = [
        Team(name="t1", team=1, soldiers=[Soldier(x=350.0, y=225.0, alive=True)]),
        Team(name="t2", team=2, soldiers=[Soldier(x=420.0, y=225.0, alive=True)]),
    ]
    g = Game(
        state=GameState(teams=teams, current_turn=0),
        terrain=make_circle_obstacle(wall),
        circles=list(wall),
    )
    assert not any(r.reachable for r in reachability(g))
    g.fire("0*x")
    g.fire("0*x")
    assert any(r.reachable for r in reachability(g))


def test_cell_envelope_sound_with_carve() -> None:
    wall = _stacked_wall()
    teams = [
        Team(name="t1", team=1, soldiers=[Soldier(x=350.0, y=225.0, alive=True)]),
        Team(name="t2", team=2, soldiers=[Soldier(x=420.0, y=225.0, alive=True)]),
    ]
    g = Game(
        state=GameState(teams=teams, current_turn=0),
        terrain=make_circle_obstacle(wall),
        circles=list(wall),
    )
    g.fire("0*x")
    g.fire("0*x")
    g.fire("0*x")
    from graphwar_sim import ccf as _ccf

    fr = shooter_frame(g)
    reach = reachability(g)
    reachable = [r for r in reach if r.reachable]
    assert reachable, "expected the dug wall to be reachable"
    excl = build_exclusions(0, fr.targets, fr.teammates, 0.45)
    # The M5.1 sweep's backtraced chains splice branches (documented plunge)
    # and die cell-wise here; the CCF branch DP is the certifiable geometry
    # source — use its first surviving branch for the soundness cross-check.
    tx, ty = fr.targets[0]
    chain = None
    cells = None
    for branch in _ccf._branch_paths(
        fr.mx, fr.my, tx, ty, fr.circles, excl, fr.inverted, 0, 4, fr.carves
    ):
        cells = chain_cells(branch, fr.mx, fr.circles, excl, 0.01, fr.carves)
        if cells is not None:
            chain = branch
            break
    assert chain is not None and cells is not None, "expected a surviving CCF branch"
    Lcells, Hcells = cells
    du = 0.01
    for k, (lc, hc) in enumerate(zip(Lcells, Hcells, strict=True)):
        u_lo, u_hi = fr.mx + k * du, fr.mx + (k + 1) * du
        px_lo, px_hi = plane_x_of(u_lo), plane_x_of(u_hi)
        c_lo, c_hi = math.floor(min(px_lo, px_hi)), math.floor(max(px_lo, px_hi))
        for pc in range(c_lo, c_hi + 1):
            if not (0 <= pc < config.PLANE_LENGTH):
                continue
            for py in range(config.PLANE_HEIGHT):
                if not g.terrain.collide_point(pc, py):
                    continue
                y_top = (
                    config.PLANE_GAME_LENGTH
                    * (-py + config.PLANE_HEIGHT / 2.0)
                    / config.PLANE_LENGTH
                )
                y_bot = (
                    config.PLANE_GAME_LENGTH
                    * (-(py + 1) + config.PLANE_HEIGHT / 2.0)
                    / config.PLANE_LENGTH
                )
                mid = (y_top + y_bot) / 2.0
                assert not (lc < mid < hc), f"cell {k} pc={pc} py={py} blocked but inside envelope"


def test_ccf_certifies_shot_through_crater() -> None:
    from graphwar_sim import ccf
    from graphwar_sim.parser import PolishNotationFunction
    from graphwar_sim.physics import process_function_range

    wall = _stacked_wall()
    teams = [
        Team(name="t1", team=1, soldiers=[Soldier(x=350.0, y=225.0, alive=True)]),
        Team(name="t2", team=2, soldiers=[Soldier(x=420.0, y=225.0, alive=True)]),
    ]
    g = Game(
        state=GameState(teams=teams, current_turn=0),
        terrain=make_circle_obstacle(wall),
        circles=list(wall),
    )
    # Three sacrificial flats widen the breach (all miss); the enemy stands.
    for _ in range(3):
        r = g.fire("0*x")
        assert r.hits == []
    sol = ccf.solve_for_game(g)
    assert sol.outcome.value == "CERTIFIED", f"expected CERTIFIED, got {sol.outcome}"
    assert sol.candidates, "expected a certified candidate"
    expr = sol.candidates[0].expression
    team = g.state.current_team()
    f = PolishNotationFunction(expr)
    resim = process_function_range(
        f, team.current_soldier(), g.all_soldiers(), g.terrain, team.team == config.TEAM2
    )
    assert resim.hits, "re-simulation must hit through the carved wall"
