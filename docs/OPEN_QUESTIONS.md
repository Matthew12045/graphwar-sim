# Open Questions

This file tracks questions that could not be resolved from the reference source
alone. Per the plan's ground rules, divergence is **reported, not tuned away**,
and any assumption we carry forward is recorded here with its status.

**Status as of M0: all three original open questions are RESOLVED** with
`file:line` citations. Details and citations live in `GROUND_TRUTH.md`.

**Status as of M2:** the three original questions remain resolved. M2 adds one
recorded divergence (fixed-grid vs per-target-centre basis ordering) — see the
M2 section below.

---

## Original open questions (all resolved)

### (a) Hit-test coordinate space — RESOLVED

**Question:** Is the trajectory-vs-soldier hit test performed in game/math
coordinates or in plane/pixel coordinates?

**Answer:** **Plane/pixel coordinates.** Both the integrated trajectory point and
the soldier positions are in plane space when the distance is computed. The
transform to plane happens at `Function.java:243-244`, and the distance check
uses those plane values at `Function.java:264-284`. Units are consistent.
See `GROUND_TRUTH.md` §2.6.

### (b) `soldierCollides` 5-point sampling scope — RESOLVED

**Question:** What is `Obstacle.soldierCollides` (the center + 4-cardinal-point
check) used for? Is it the trajectory hit test?

**Answer:** It is **placement validation only**, not the trajectory hit test.
`Obstacle.soldierCollides` (`Obstacle.java:123-161`) is a 5-point white-pixel
check used when deciding where a soldier may be *placed*. The trajectory hit
test is the separate strict-`<` radius check in `Function.java:272`.
See `GROUND_TRUTH.md` §2.6 and §6.8.

### (c) Multi-hit / win rule — RESOLVED

**Question:** Can a single shot kill multiple soldiers? When does the game end?

**Answer:** **Yes, a single shot can multi-kill** — the hit loop
(`Function.java:252-284`) has no `break`, so the trajectory continues past a hit
and can record further hits on later points. **The game ends when either team
reaches zero alive soldiers** — `GameData.checkGameFinished`
(`GameData.java:539`): `if (team1Alive == false || team2Alive == false) return
true;`. See `GROUND_TRUTH.md` §2.6 and §5.

---

## Carried-forward assumptions (not yet a question, but flagged)

These are source-faithful behaviors that look like latent quirks. We reproduce
them exactly (fidelity to source wins) and flag them here so a future reviewer
knows they are deliberate, not bugs in our port.

- **Inverse transform uses `PLANE_LENGTH` for both axes**
  (`Function.java:243-244`), not `PLANE_HEIGHT` for `y`. We match the source.
  See `GROUND_TRUTH.md` §6.7.
- **`Obstacle.getNumCircles` (client) diverges from `GraphServer.generateCircles`
  (server).** We use the server version as authoritative. See `GROUND_TRUTH.md`
  §6.6.

---

## M2: fixed-grid vs per-target-centre basis — divergence from the plan

**Question:** The plan names a **fixed-grid** least-squares Gaussian basis as
the *primary* basis for the M2 solver. On the seeded battery, does it outperform
the simpler **per-target-centre** variant (one Gaussian centre per enemy)?

**Answer:** **No — the plan's named primary underperforms, and is the only rung
that produced friendly fire.** We therefore order the degradation ladder with
per-target-centre Gaussians first and the fixed-grid rung near the bottom. This
is a divergence from the plan's ordering, recorded here (not tuned away).

Measured on the seeded battery (24 seeds, 2 teams × 4 soldiers, isolated per
rung — each rung fired alone, no ladder), after the plain-decimal emission fix
(see below):

| Rung (isolated)       | tried | hit | friendly fire |
|-----------------------|:-----:|:---:|:-------------:|
| `parabola`            |  24   |  8  |      0        |
| `line`                |  24   |  7  |      0        |
| `per_target_gaussian` |  24   |  5  |      0        |
| `fixed_grid_gaussian` |  24   |  4  |      **1**    |

Notes on why this is expected, not a bug:

- **Terrain occlusion is a real constraint, but not always the binding one.**
  A single-valued `y = f(x)` curve can only reach a target along a
  monotone-`x` path, and it may *arc over or under* terrain. Far-centred
  Gaussians underflow to ~0 near the muzzle (a flat approach) and die at
  terrain before reaching far enemies. Some of those duds were a **fit gap** —
  the ladder never tried a terrain-clearing arc — and are now recovered by the
  terrain-aware `arc` rung (see below). A smaller remainder is a genuine
  physical limit of NORMAL_FUNC mode (no monotone-`x` path exists), which is
  why the degradation ladder and the safe-dud rung exist at all.
- **The fixed grid is the *worst* fit for the occlusion problem.** Its centres
  are spread uniformly over the enemy span, so the near-muzzle region is
  under-resolved relative to per-target centres; the resulting flat approach
  dies at terrain more often, and on one seed the least-squares weights swing
  the curve into a teammate. Per-target centres place a lobe exactly on each
  enemy, which is the better use of the basis under occlusion.
- **The multi-target Gaussian rungs still lead the ladder** despite the lower
  isolated hit rate, because they are the plan's intended primary basis and the
  only rung that can multi-kill (relevant once more than one enemy is in play).

**Emission fix that changed these numbers:** the parser rewrites every `-` to
`+-` (unary negation), so a scientific-notation literal such as `1e-06` is
corrupted into `1e+-06` and misparses as `1*e - 6 ≈ -3.28`. The solver now
emits **plain decimals only** (`solver._num`), never an exponent. Before this
fix, tiny Gaussian weights silently corrupted the curve (a round-trip test
caught it: intended `1e-6` at a centre, parsed value `-3.28`).

**Status:** recorded. The ladder order in `graphwar_sim/solver.py` reflects the
measured ordering above; the fixed-grid rung is retained (the plan's named
basis) but demoted. The M2 acceptance bar is *parseable + certified + logged
hit rate / rung distribution*, **not** a high hit rate. (The "low absolute hit
rate is the terrain-occlusion limit" claim below this table was later revised:
a terrain-aware `arc` rung recovered most of the duds — see the next section.)

---

## M2: terrain-aware `arc` rung — recovering the "terrain duds"

**Question:** On the seeded battery, ~half the maps degraded to the safe-dud
rung because the fitted curve hit a rock before reaching the enemy. Is that a
genuine physical limit (no clear path exists) or a fit gap (the ladder never
tried a curve that clears the terrain)?

**Answer:** **Mostly a fit gap.** The closed-form `parabola` rung is one
specific member of the family of quadratics through the muzzle and an enemy;
it fixes the curvature. A single-valued `y = f(x)` can only reach a target
along a monotone-`x` path, but it may *arc over or under* terrain — and the
curvature is exactly the degree of freedom that decides whether the arc clears
the rocks. The new `arc` rung sweeps that curvature per enemy and lets the
simulator (the oracle) pick the first arc that clears terrain and lands.

**Model.** For an enemy `(tx, ty)` and muzzle `(mx, my)`, the auto-offset
constraint is `f(tx) - f(mx) = ty - my`. Writing `f(x) = A·x² + B·x` (the
constant term is absorbed by the auto-offset), the secant slope
`s = (ty - my)/(tx - mx)` pins `B = s - A·(tx + mx)` once the quadratic
coefficient `A` is chosen. Sweeping `A` over `[-_ARC_A_MIN, _ARC_A_MAX]`
step `_ARC_A_STEP` (all `# TUNABLE`) yields arcs that all pass through the
muzzle and the enemy but bulge to different extents. Candidates are ordered
nearest-enemy-first, then by `|A|` (flattest first).

**Measured effect** (40 seeds, 2 teams × 2 soldiers, full ladder):

| Metric            | before | after |
|-------------------|:------:|:-----:|
| hit rate          | 18/40 (45%) | **33/40 (82%)** |
| `arc` rung        |   —    |  15   |
| `dud` rung        |  22    |   7   |
| friendly fire     |   0    |   0   |
| parse failures    |   0    |   0   |

The 15 recovered seeds are exactly the ones a wider offline sweep (quadratics
through muzzle+enemy, `A ∈ [-0.3, 0.3]`, 0.002 step) confirmed are reachable by
*some* clear arc — i.e. the `arc` rung is not missing them, it is finding them.

**What the `arc` rung does *not* fix — the genuine limit.** Six of the seven
remaining duds (seeds 2, 18, 20, 23, 27, 32 at the default 2-soldier config)
are **not** reachable by any quadratic arc in `[-0.3, 0.3]` in the offline
sweep: no monotone-`x` path clears the terrain, so a safe dud is the correct
degradation. (Seed 31 is a knife-edge: it clears only at exactly `A = 0.024`,
one 0.002-grid point, and was not recovered by the shipped 0.005 grid. We
deliberately did not refine the grid to chase this single fragile seed — a
shot that clears only at one 4-decimal curvature is numerically unreliable,
and tuning the grid to it would overfit the seeded battery. Recorded, not
tuned away.)

**Grid choice.** A narrow fine grid (`[-0.1, 0.1]`, step 0.005) recovers the
most duds at the lowest cost; widening the range to `[-0.3, 0.3]` recovers
nothing extra (the extra curvature just misses the targets) and costs ~3× more
integrations. The arc rung sits just above the duds in the ladder, so its
per-candidate integration cost is paid only on the maps where every cheaper
rung already failed.

**Status:** implemented and recorded. `graphwar_sim/solver.py` adds the
`RUNG_ARC` rung (`_arc_candidates` / `_arc_grid`) between the fixed-grid rung
and the duds. `tests/test_solver.py::test_arc_rung_clears_terrain_the_cheaper_
rungs_cannot` guards the terrain-awareness property.

---

## Genuinely open (deferred to later milestones)

- **Token-cost / ablation metrics (M4).** The minimal viable slice skips
  token-cost tracking and ablations. If added later, the LLM agent's prompt
  budget and the cost of `simulate_tool` calls will need a defined accounting
  method. Deferred by the plan, not a source question.
- **Determinism of the RNG across JVM/Python.** The server uses
  `java.util.Random`/`nextGaussian`. For reproducible golden tests we seed a
  Python `random.Random`; the *distribution* must match, but exact
  Gaussian-stream parity with the JVM is not required (golden tests compare
  simulator behavior on fixed inputs, not RNG stream identity). Confirmed
  acceptable for the M1 golden-test strategy.
