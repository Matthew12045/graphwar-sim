"""M5.2 sec.11 property test: every CERTIFIED CCF shot clears terrain at 10x density.

Requirement (5.2.md sec.11): "every CERTIFIED result re-simulated through
``physics.py`` at 10x sample density, zero collisions, >= 200 randomized maps.
Use the real integrator with its adaptive step-halving -- not a
reimplementation."

What the certificate claims (mod :mod:`graphwar_sim.ccf`): the emitted
expression keeps the *fired* trajectory inside the free geometry from the
muzzle to the goal column (the certificate is about CLEARANCE -- whether the
shot also HITS its target is the simulator's call). This test therefore
re-fires every CERTIFIED candidate two independent ways and asserts both hold:

Real integrator (``physics.process_function_range``)
-----------------------------------------------------
We rebuild the map and fire the candidate through the REAL integrator, exactly
as ``solver._verify`` does (``solver.py:458-480``) -- ``PolishNotationFunction``
on the expression, then ``process_function_range(f, shooter, game.all_soldiers(),
game.terrain, frame.inverted)``. From the :class:`ShotResult` we assert:

- **No terrain collision** = the trajectory REACHED the goal column. The
  integrator advances ``x`` monotonically (physics.py:198-220) and STOPS at the
  first terrain pixel, truncating the trajectory (physics.py:245-247: it sets
  ``num_steps`` and breaks when ``obstacle.collide_point(int(x), int(y))`` is
  true). If the shot had collided anywhere before the goal, the trajectory would
  end early; asserting the trajectory's farthest world-``u`` reached the
  target's disk (``u_T - WORLD_RADIUS``) proves it cleared every terrain pixel
  on the way. We reconstruct that ``u`` from ``ShotResult.points`` (plane px)
  by undoing the plane transform and TEAM2 mirror of physics.py:222-226.
- **No teammate among ``result.hits``** -- the certificate excludes teammate
  disks (corridor.py:86-97); a friendly hit would violate it. The shooter's own
  disk is skipped by the physics (physics.py:229-233), so it is not a teammate.

10x-dense fired-curve sampling
------------------------------
``ccf._fired_curve`` (ccf.py:649-667) reconstructs the exact curve the game
fires through the REAL parser, REAL launch-angle fixed point and REAL nudge --
it mirrors the muzzle setup of ``process_function_range`` (physics.py:154-206:
``game_coordinate_radius`` == ``ccf._NUDGE_RADIUS``, same ``_get_start_angle``,
same auto-offset). Sampling this curve at 10x the corridor density
(``du = config.STEP_SIZE``, so step ``STEP_SIZE / 10``) and testing points with
the physics' OWN collision path (``terrain.collide_point(int(x), int(y))`` --
the identical helper the integrator calls at physics.py:245) catches byte-level
collisions between the corridor's STEP-spaced samples that the real integrator's
coarser grid can slide past. Sample points are converted to plane pixels
exactly as the integrator does (physics.py:222-226, incl. the TEAM2 mirror at
225-226), and soldier disks are tested with the physics' hit test
(``dist^2 < SOLDIER_RADIUS^2``, physics.py:235-238) against every soldier except
the intended target -- which is exempt because entering it is the goal (its disk
may only be entered within ``2 * WORLD_RADIUS`` of the goal column).

If a CERTIFIED candidate fails either check it is a genuine certificate defect
(a "CERTIFIED-but-collided" case, the sec.13 kill-criterion), reported by the
assertion.
"""

from __future__ import annotations

import collections

from graphwar_sim import ccf, config
from graphwar_sim.corridor import WORLD_RADIUS, shooter_frame
from graphwar_sim.parser import PolishNotationFunction
from graphwar_sim.physics import process_function_range
from graphwar_sim.state import Game

# Soldiers per map.
_NUM_SOLDIERS = 2  # the acceptance battery's inner soldier count.

# Number of randomized maps. # TUNABLE -- spec minimum is 200; keep 200.
_NUM_MAPS = 200

# 10x the corridor's sample density (corridor du == config.STEP_SIZE).
_SAMPLE_STEP = config.STEP_SIZE / 10.0


def _reached_u(result, inverted, mx):
    """Farthest world-``u`` reached by the real trajectory (mirror-aware).

    ``result.points`` gives the integrated trajectory in plane px with the TEAM2
    mirror already applied (physics.py:258-265); we undo the mirror and the
    plane->world transform (inverse of physics.py:222-226) to recover world-x,
    then subtract the muzzle ``mx`` to get ``u``.
    """
    px = result.points[-1][0]
    if inverted:
        px = config.PLANE_LENGTH - px
    world_x = (
        config.PLANE_GAME_LENGTH * (px - config.PLANE_LENGTH / 2.0) / config.PLANE_LENGTH
    )
    return world_x - mx


def _plane_of(wx, wy, inverted):
    """World ``(wx, wy)`` -> plane px, exactly as physics.py:222-226 (incl. the
    TEAM2 mirror at 225-226)."""
    px = config.PLANE_LENGTH * wx / config.PLANE_GAME_LENGTH + config.PLANE_LENGTH / 2.0
    py = (
        -config.PLANE_LENGTH * wy / config.PLANE_GAME_LENGTH + config.PLANE_HEIGHT / 2.0
    )
    if inverted:
        px = config.PLANE_LENGTH - px
    return px, py


def _check_certified_candidate(seed, cand, game):
    """Assert the real-integrator + 10x-dense clearance for one CERTIFIED shot.

    Returns a short diagnostic summary ``(tag, n_samples)``.
    """
    fr = shooter_frame(game)
    target_index = cand.target_index
    tx, ty = fr.targets[target_index]
    u_T = tx - fr.mx

    # --- (a) Real-integrator check on a FRESH map ---------------------------
    fresh = Game.create(seed, num_soldiers=_NUM_SOLDIERS)
    fresh_fr = shooter_frame(fresh)
    shooter = fresh.state.current_team().current_soldier()
    f = PolishNotationFunction(cand.expression)
    result = process_function_range(
        f, shooter, fresh.all_soldiers(), fresh.terrain, fresh_fr.inverted
    )
    reached = _reached_u(result, fresh_fr.inverted, fresh_fr.mx)

    # No terrain collision == the trajectory reached the goal column
    # (physics.py:245-247 truncates the trajectory early on terrain contact).
    assert reached >= u_T - WORLD_RADIUS, (
        f"seed={seed} CERTIFIED candidate for target {target_index} stopped at "
        f"u={reached:.3f} (goal u_T={u_T:.3f}): terrain collision before the goal; "
        f"certificate claims clearance."
    )

    # No teammate among hits (physics skip rule excludes only the shooter,
    # physics.py:229-233; corridor excludes teammate disks, corridor.py:86-97).
    team_id = fresh.state.current_team().team
    mate_ids = {
        (j, k)
        for j, team in enumerate(fresh.state.teams)
        if team.team == team_id
        for k, s in enumerate(team.soldiers)
        if s is not shooter
    }
    mate_hits = [h for h in result.hits if (h[0], h[1]) in mate_ids]
    assert not mate_hits, f"seed={seed} CERTIFIED candidate: friendly fire {mate_hits}"


    # --- (b) 10x-dense fired-curve sampling on the real geometry ------------
    curve = ccf._fired_curve(cand.expression, fr.mx, fr.my)
    assert curve is not None, f"seed={seed} CERTIFIED candidate did not parse"

    # Map the intended target to its plane soldier (enemy teams, alive).
    shooter_id = (shooter.player_index, shooter.soldier_index)
    target_soldier = None
    for j, team in enumerate(fresh.state.teams):
        if team.team == team_id:
            continue
        for k, s in enumerate(team.soldiers):
            if not s.alive:
                continue
            if fresh_fr.inverted:
                swx = config.PLANE_GAME_LENGTH * (
                    (config.PLANE_LENGTH - s.x) - config.PLANE_LENGTH / 2.0
                ) / config.PLANE_LENGTH
            else:
                swx = config.PLANE_GAME_LENGTH * (
                    s.x - config.PLANE_LENGTH / 2.0
                ) / config.PLANE_LENGTH
            swy = (
                config.PLANE_GAME_LENGTH * (-s.y + config.PLANE_HEIGHT / 2.0)
                / config.PLANE_LENGTH
            )
            if abs(swx - tx) < 1e-6 and abs(swy - ty) < 1e-6:
                target_soldier = s
                break

    soldiers = fresh.all_soldiers()  # sets player_index/soldier_index
    radius2 = config.SOLDIER_RADIUS * config.SOLDIER_RADIUS
    n_samples = 0
    u = 0.0
    while u <= u_T + 1e-9:
        px, py = _plane_of(fr.mx + u, curve(u), fr.inverted)
        # physics' own collision path (physics.py:245).
        assert not fresh.terrain.collide_point(int(px), int(py)), (
            f"seed={seed} CERTIFIED candidate for target {target_index}: fired "
            f"curve enters terrain at u={u:.4f} (plane {int(px)},{int(py)}); "
            f"certificate claims clearance."
        )
        for s in soldiers:
            if s is target_soldier:
                # The intended target may be hit -- but only near the goal column.
                if u >= u_T - 2 * WORLD_RADIUS:
                    continue
                if (s.x - int(px)) ** 2 + (s.y - int(py)) ** 2 < radius2:
                    raise AssertionError(
                        f"seed={seed} CERTIFIED candidate: enters target disk "
                        f"before the goal at u={u:.4f}"
                    )
                continue
            if (s.player_index, s.soldier_index) == shooter_id:
                continue  # shooter's disk is never hit-tested (physics.py:229-233)
            if (s.x - int(px)) ** 2 + (s.y - int(py)) ** 2 < radius2:
                raise AssertionError(
                    f"seed={seed} CERTIFIED candidate: fired curve enters a "
                    f"soldier disk at u={u:.4f} (plane {int(px)},{int(py)})"
                )
        n_samples += 1
        u += _SAMPLE_STEP
    return "ok", n_samples


def test_certified_no_collisions_over_200_maps():
    """Fire every CERTIFIED candidate on seeds 1..200 and assert zero collisions
    (terrain at 10x density + soldier disks), no friendly fire, and that each
    trajectory reaches its goal column."""
    summary = collections.Counter()
    failures = []
    total_dense = 0

    for seed in range(1, _NUM_MAPS + 1):
        game = Game.create(seed, num_soldiers=_NUM_SOLDIERS)
        res = ccf.solve_for_game(game)
        summary[f"outcome:{res.outcome.value}"] += 1

        certified = [c for c in res.candidates if c.certified]
        summary["certified_candidates"] += len(certified)

        if res.outcome == "CERTIFIED":
            assert certified, (
                f"seed={seed}: OUTCOME=CERTIFIED but no certified candidate returned"
            )
        if not certified:
            continue

        n_checked = 0
        for cand in certified:
            try:
                _tag, n = _check_certified_candidate(seed, cand, game)
                total_dense += n
                n_checked += 1
            except AssertionError as exc:  # a genuine CERTIFIED-but-collided case
                failures.append(str(exc))
        summary["candidates_checked"] += n_checked

    # Always print the distribution (for the battery REPORT); do NOT assert it.
    print(f"\n[ccf property]  {dict(summary)}")
    print(f"[ccf property]  total 10x-dense samples checked: {total_dense}")

    assert not failures, (
        f"{len(failures)} CERTIFIED-but-collided case(s) across "
        f"{_NUM_MAPS} maps:\n" + "\n".join(failures)
    )
