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

## M3 / M4: agents and the evaluation harness — notes (not source questions)

These are design/practice notes from the M3 (agents) and M4 (eval) milestones.
None of them is a divergence from the reference source; they are recorded so
a future reviewer does not misread the numbers or the choices.

### Observation frame (M3) — resolved by test, not a question

The agent observation presents the board in the **centered world frame**
(``x`` in ``[-25, 25]``, ``y`` up) with the TEAM2 mirror applied so the
shooter always faces right — the same frame the solver's fit works in and the
physics integrates. This was the plan's highest-risk integration point; it is
locked by the frame round-trip test (``tests/test_agents.py``), which fires a
world-frame expression derived from the observation through the real physics
and asserts the intended geometry is hit (both TEAM1 and TEAM2/mirror cases).

The observation's terrain summary (coarse world-frame blocked cells, step
15px) and the ASCII board size are ``# TUNABLE`` (not source constants).

### Rung distribution in matches (M4) — a measurement-context difference

``eval/results/leaderboard.md`` reports the solver's per-**shot** rung counts:
``dud=297`` of 322 shots. This is *not* a regression vs the M2 battery
(33/40 first-shot hits): most of those duds come from (a) the handful of
seeded maps where no monotone-x path exists (the genuine occlusion limit, see
the M2 ``arc`` rung section) and (b) matches that reach the 100-turn cap — on
an unreachable map the solver duds every turn for the whole match,
multiplying the dud count linearly while the M2 battery measured exactly one
solve per seed. A per-*seed* first-shot rung distribution remains the M2
battery's number.

### Baseline matches mostly draw (M4) — expected

``random`` vs ``straight`` matches end in draws at the turn cap on most maps:
neither baseline clears terrain (that is precisely what the solver's ``arc``
rung was built for), so they trade duds until the cap. Win rate between
baselines is still meaningful on the few maps a straight line happens to
thread through.

### Plot reproducibility (M4) — data parity, not PNG byte parity

Re-running from the seed file reproduces ``leaderboard.md`` **byte-for-byte**
(asserted in ``tests/test_eval.py``) and the same per-match plot set. PNG
*bytes* are not asserted: matplotlib output can vary with library version and
font cache while carrying identical data.

## M5: corridor sweep, taxonomy split — findings (reported before M5.2)

Status: all three items below were answered from the reference source and
from measurements; none is a placeholder.

### (d) Pre-fire movement — RESOLVED: soldiers never move during a match

**Question (M5.1):** does the real game let a soldier MOVE before firing? If
yes and the sim omits it, some PASS_UNREACHABLE verdicts would be simulator
artifacts.

**Answer: No. A turn is type-a-function-and-fire; there is no movement step.**
Citations:
- ``Player.java`` has no movement API — only ``startSoldier(x, y)`` (pre-game
  placement) and ``nextTurn()`` soldier cycling.
- ``GameScreen.java:570-610`` — all mouse handlers are ``TODO`` stubs;
  ``RoomBoard.mouseDragged`` (RoomBoard.java:204) is a stub too.
- ``NetworkProtocol.java:31-38`` — the protocol has ``ADD_SOLDIER`` /
  ``SET_SOLDIER`` (pre-game) and ``FIRE_FUNC``, but no move message.
- The README describes the only in-game action: type ``y = f(x)``.

Consequence: the sim's no-movement model is faithful; PASS_UNREACHABLE is
not a movement artifact. (The real game lets players CHOOSE placement
pre-game; our seeded random placement is a ``# TUNABLE`` stand-in and can
create positions a human would not pick — a map-generation modeling choice,
recorded, not a physics gap.)

### (e) Expression character limit — RESOLVED: the reference has none

**Question (5.2.md §0):** the CCF emission budget needs the game's character
limit.

**Answer: the reference imposes no expression length limit.** The function
input is a plain ``JTextField`` with no ``DocumentFilter`` (GameScreen.java:101,
GraphUtil.makeTextField — the config "columns" value is a width hint, not a
limit); the function travels URL-encoded over a line-based socket protocol
(GameData.java:323, ``readMessage``) with no length check anywhere. The
harness therefore uses a defensive ``# TUNABLE`` cap (``MAX_EXPR_CHARS``,
not from source) for the CCF emission budget; the test asserts it reads the
cap from config so the test breaks if the cap changes.

### (f) The corridor refutes the M2 "genuine limit" seeds — a measurement, not a tune

**Finding:** the M2 report classified seeds 2, 18, 20, 23, 27, 32 (2×2 config)
as a "genuine physical limit: no monotone-x path clears the terrain." The
M5.1 corridor sweep (with the Phase-0-derived slope cap, which provably
contains every surviving trajectory) finds all of them **reachable**: the
sweep is empty only when some column of the free set is entirely blocked
(a terrain/disk wall spanning the whole band), and the seeded generator
(mean circle radius 40px) essentially never produces such a wall. The M2
judgment was correct only for the *quadratic-arc family* (the arc rung's
sweep range): the six seeds are **fit gaps**, not physical limits. The dud
bucket therefore splits mostly into `SOLVER_FAILED` (corridor-reachable,
ladder failed) with `PASS_UNREACHABLE` structurally rare on seeded maps
(observed: 0/300 seeds at 1/2/4 soldiers; the taxonomy machinery is tested
with a synthetic full-wall board). Reported, not tuned away.

### (g) Slope cap: the derived default never binds in real games

``S = sqrt(FUNC_MAX_STEP_DISTANCE_SQUARED) / FUNC_MIN_X_STEP_DISTANCE ≈ 3162``
(derivation in ``corridor.py``: the max |dy/dx| a surviving trajectory can
exhibit before the integrator's halving floor kills it). With
``S·du = 31.6 > band height 29.16``, R fills the entire band within ~1
column: the cap binds only within the first ~`band_height/(S·du)` columns of
the muzzle, and enemies are ≥ ~1.3 world units away. So the sweep degenerates
to a geometric threading test, which is exactly the intended proof: any
monotone-x curve is slope-feasible; only full walls block. Tested directly
(`test_slope_cap_semantics`)

### (h) CCF teammate-disk inflation (5.2.md §5) — recorded consequence

The CCF cell envelope excludes teammate disks with ``SOLDIER_RADIUS +
EXPLOSION_RADIUS = 19px`` per 5.2.md §5 (M5.1's sweep keeps the Phase 0 hit
radius 7px). Consequence: the inflated disk can in principle exclude a legal
safe shot in a corridor between two teammates (20px Chebyshev spacing vs a
19px exclusion disk). The solver's self-verification remains the final
authority; a false-unreachable costs a turn, never a wrong shot.

---

## M5.2: CCF — findings

### (i) Corridor pixel-cell collision fidelity — the free set was half a pixel too generous

**Finding:** the physics tests `collide_point(int(x), int(y))` — a plane point in
row `py` spans the full pixel cell `[py, py+1)`, and `make_circle_obstacle`
blocks integer rows whose CENTER is inside the circle (`(py - cy)^2 <= s^2`).
The corridor's blocked band therefore covers the FULL pixel cells of the blocked
rows `[py_lo, py_hi]`: `[y(py_hi + 1), y(py_lo)]` — not just up to `y(py_hi)`
(the row's top edge). The old band admitted curve-edge points that COLLIDE in
the physics. For the M5.1 sweep (oracle-gated) that was merely optimistic; for
an M5.2 CCF CERTIFICATE it is fatal, so `_column_free` and
`_cell_obstacle_band` now use the full-cell band, plus an empty-rows guard when
a circle covers no integer row center at a column. Consequence: M5.1 verdicts
stay sound (the sweep is now slightly more conservative). Recorded, not tuned
away. Same fidelity pass: `_side_at` classifies a cell band that only PARTIALLY
overlaps a branch (top/bottom sliver) as ceiling/floor — only a band strictly
CONTAINING the branch is unclassifiable;
`tests/test_corridor.py::test_cell_wise_envelope_catches_a_synthetic_spike`
asserts the spike's cells raise the floor (narrowing), not chain death.

**Addendum — second fidelity fix, same theme:** the blocked-row ranges must come
from the physics' DISCRETE test column `int(px)` (the trajectory point is tested
at `collide_point(int(x), int(y))`, not at the continuous `px`). The continuous
`px` computation was up to one pixel row OVER-OPTIMISTIC on a circle's right
flank (`px > cx`), where `int(px) <= px` puts the tested column closer to the
centre and admits one more blocked row. This was found by the M5.2 property test
once the tight-mode CCF gate frame fix (`ccf.py`: gate on `g(u) − my`, not the
absolute fired curve) stopped masking it — the old broken gate's shifted window
had rejected most ceiling-hugging tight candidates. `_column_free` now evaluates
each circle at `pc = int(px)`; `_cell_obstacle_band` unions the blocked row
ranges over every discrete column `[floor(px_lo), floor(px_hi)]` of the cell
(the rows nest, so the exact union is the row range at the integer column
closest to the centre). M5.1 verdicts stay sound — the sweep is now slightly
more conservative on right flanks and unchanged elsewhere. The soldier
hit-tests are CONTINUOUS `dist² < r²` (no `int` rounding), so exclusion disks
were deliberately left continuous. Recorded, not tuned away.

### (j) §8 K branch paths — diversity-penalty DP re-runs, not exact K-best

**Question (5.2.md §8):** "Take the K best paths."

**Answer (recorded deviation):** exact K-best needs a (path, node) state space
that is not worth its cost at K=4. `graphwar_sim/ccf.py::_branch_paths` instead
re-runs the exact shortest-path DP with `+_PATH_PENALTY` added to every node
(component) used by an earlier path, which picks the K most diverse cheap
branches — what the cell-wise envelope actually consumes. The DP itself stays
exact (layered DAG; the column sweep is equivalent to Dijkstra). The choice and
its rationale live in the `_branch_paths` docstring; this entry records the
deviation from literal K-best.

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
