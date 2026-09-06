# Evaluation report

Distributions over the seeded batteries, per the M5 instrumentation rule
(5.2.md §12): report **distributions, not means**, and diff rung histograms
against the previous milestone's board.

- Battery source: `eval/results/seeds.json` (root seed 1000, 3-agent round
  robin, 5 matches/pair × 2 sides = 30 matches). Reproducible from the seed
  file alone: `python3 -m eval --from-seeds eval/results/seeds.json`.
- The per-seed first-shot batteries below run `graphwar_sim.solver.solve` once
  per seed on fresh `Game.create(seed, num_soldiers=…)` maps.

---

## M5.1 — corridor pre-check + honest outcome taxonomy

**Change under test:** the M5.1 milestone added (a) the corridor reachability
pre-check in `solve()` (`graphwar_sim/corridor.py`) and (b) the outcome
taxonomy that replaces the single `dud` bucket with
`PASS_UNREACHABLE | SOLVER_FAILED`, plus attempt dedupe and the
stalemate/turn-cap draw split in the runner.

### Match board (root seed 1000, 2×2 soldiers, before vs after M5.1)

| Metric (solver agent)      | M4 board | M5.1 board |
|----------------------------|:--------:|:----------:|
| Matches                    |   20     |     20     |
| Wins                       |   14     |     14     |
| Win rate                   |  0.700   |   0.700    |
| Shots recorded             |  322     |    226     |
| Hit rate                   |  0.078   |   0.111    |
| Kills                      |   31     |     31     |
| Friendly fire              |    0     |      0     |
| `dud`-bucket turns         |  297     |      0     |
| `SOLVER_FAILED` turns      |   —      |    297     |
| `PASS_UNREACHABLE` turns   |   —      |      0     |
| Repeat-suppressed attempts |   —      |     96     |

The win/kill columns are identical, as they must be: M5.1 changed
*classification and bookkeeping*, not shot selection. The 322 → 226 shot-count
drop is the dedupe rule (96 duplicate attempts suppressed: on unreachable or
stuck maps the deterministic solver re-emits the identical expression for an
unchanged board). Hit rate rises mechanically because suppressed repeats no
longer dilute it.

### Rung histogram (per-shot, matches; M4 → M5.1)

| Rung                 | M4 | M5.1 |
|----------------------|:--:|:----:|
| `dud`                | 297 |  —  |
| `SOLVER_FAILED`      |  —  | 297 |
| `PASS_UNREACHABLE`   |  —  |   0 |
| `per_target_gaussian`|   7 |   7 |
| `parabola`           |   2 |   2 |
| `arc`                |  12 |  12 |
| `line`               |   3 |   3 |
| `fixed_grid_gaussian`|   1 |   1 |

### Per-seed first-shot battery (seeds 1–40, the M2/M5.1 measurement unit)

- 2×2 soldiers: `SOLVER_FAILED` = 13/40 seeds, `PASS_UNREACHABLE` = 0/40;
  the corridor sweep found every seed reachable.
- **Finding (recorded in `docs/OPEN_QUESTIONS.md` (f)):** the corridor refutes
  the M2 report's "genuine physical limit" classification of seeds
  2, 18, 20, 23, 27, 32. They are *fit gaps*: the sweep (whose slope cap
  provably contains every surviving trajectory) finds a clear monotone-x path
  for all six. `PASS_UNREACHABLE` structurally requires a full-band terrain
  wall, which the seeded generator (mean circle radius 40 px) never produced
  in 300 seeds (1/2/4 soldiers); the pass machinery is therefore tested with
  a synthetic full-wall board
  (`tests/test_solver.py::test_pass_unreachable_on_full_wall`).

### Slope-cap semantics (corridor soundness, OPEN_QUESTIONS (g))

The derived default cap `S = sqrt(FUNC_MAX_STEP_DISTANCE_SQUARED) /
FUNC_MIN_X_STEP_DISTANCE ≈ 3162` gives `S·du = 31.6` game units per column
against a 29.16-unit band: the cap never binds away from the muzzle, so the
sweep is a pure geometric threading test — exactly the intended proof that
only full walls block. Covered by `test_slope_cap_semantics`.

### CCF teammate-disk inflation consequence (OPEN_QUESTIONS (h))

The M5.2 cell envelope inflates teammate exclusion disks to
`SOLDIER_RADIUS + EXPLOSION_RADIUS = 19 px` (5.2.md §5) while the M5.1 sweep
keeps the Phase 0 hit radius (7 px). In a 20 px Chebyshev corridor between
two teammates the 19 px disk can exclude a legal safe shot: a false
UNREACHABLE costs a turn, never a wrong shot (the solver's self-verification
stays the final authority). Recorded, accepted.

---

## M5.2 — CCF (Certified Corridor Fit)

*(filled at the end of the M5.2 milestone: outcome distributions, rung
histogram diffed against the M5.1 board above, emitted-length distributions,
sigma/branch usage, MILP firing rate, and the §13 acceptance/kill decision.)*
