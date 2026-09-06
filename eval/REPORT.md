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

**Change under test:** (a) the CCF rung (`graphwar_sim/ccf.py`, §3–§7) inserted
into the degradation ladder between `line` and `fixed_grid_gaussian`
(`solver.py`, lazy `_ordered_candidates`); (b) §8 layered-DAG branch selection
(`_branch_paths`: over/under corridor branches per target); (c) two soundness
fixes the §11 property test forced once CERTIFIED shots were actually
re-simulated: the tight-mode gate now checks the FIRED RELATIVE curve
(`g(u) − my`; the old gate compared the absolute curve against
shooter-relative bounds — a window shifted by `2·my`, leaving the launch
nudge unmeasured in tight corridors), and the corridor's terrain blocking now
evaluates the physics' discrete test column (`OPEN_QUESTIONS` (i) addendum;
87 CERTIFIED-but-collided cases → 0).

### Property test (§11 — the soundness gate)

200 maps (`Game.create(seed, num_soldiers=2)`): every CERTIFIED candidate
re-fired through the real integrator and re-sampled at 10x corridor density
with the physics' own collision path — **zero collisions, zero friendly fire**
across 240 certified candidates and 7,742,781 dense samples. Outcome
distribution over the same maps: 152 CERTIFIED / 41 UNCERTIFIED / 7
BASIS_INFEASIBLE. Verification now runs across CPU cores (~54 s wall;
sequential vs 10-way parallel runs are outcome-identical on seeds 1–24, so
the MILP wall-clock budget is not load-sensitive in practice).

Measurement caveat (recorded): a stale editable install
(`pip install -e` from the `.kilo/worktrees/ui` worktree) shadowed the working
tree in every context that resolved `graphwar_sim` off `sys.path` rather than
CWD — script-file runs measured pre-§8 code and produced the earlier
8/15/1 smoke numbers. Reinstalled from the repo root; all M5.2 numbers below
are measured on the current tree. Playbook rule: after any `pip install -e`,
re-point it at the repo root, not a worktree.

### Match board (root seed 1000, 2×2 soldiers; M5.1 → M5.2)

| Metric (solver agent)      | M5.1 board | M5.2 board |
|----------------------------|:----------:|:----------:|
| Matches                    |     20     |     20     |
| Wins                       |     14     |  **15**    |
| Win rate                   |   0.700    |  **0.750** |
| Shots                      |    226     |  **178**   |
| Hit rate                   |   0.111    |  **0.152** |
| Kills                      |     31     |  **33**    |
| Friendly fire              |     0      |     0      |
| `SOLVER_FAILED` turns      |    297     |  **247**   |
| `PASS_UNREACHABLE` turns   |     0      |     0      |
| Repeat-suppressed attempts |     96     |     96     |
| Draws                      |     12     |     11     |

Nineteen of twenty matches are unchanged. The one flip is seed 2008:
a 100-turn draw (50 solver attempts, 1 hit, 0 kills) became a 3-turn win on a
CCF-certified shot — that match's 48 leaving attempts are the whole
226→178 shot-count drop (and straight's 127→78 repeats). No match regressed.

### Rung histogram (per-turn, matches; M5.1 → M5.2)

| Rung                 | M5.1 | M5.2 |
|----------------------|:----:|:----:|
| `SOLVER_FAILED`      | 297  | 247  |
| `ccf`                |  —   | **6**|
| `per_target_gaussian`|   7  |   7  |
| `parabola`           |   2  |   2  |
| `arc`                |  12  |   8  |
| `line`               |   3  |   3  |
| `fixed_grid_gaussian`|   1  |   1  |

262 of 274 solver turns paid the CCF stage (every non-closed-form rung sits
behind it); CCF converted 6 of those into certified shots, arc lost 4 turns
to those conversions, and 46 of the SOLVER_FAILED drop is the ended seed-2008
draw. Battery source: `eval/run_ccf_battery.py` →
`eval/results/ccf_battery.json` (20 solver matches, 357 s wall).

### Per-seed first-shot battery (seeds 1–40, the M2/M5.1 measurement unit)

| Rung                 | M5.2 |
|----------------------|:----:|
| `ccf`                |**17**|
| `per_target_gaussian`|  10  |
| `parabola`           |   4  |
| `arc`                |   3  |
| `line`               |   1  |
| `fixed_grid_gaussian`|   1  |
| `SOLVER_FAILED`      | **4**|

(The M5.1 battery recorded only the per-seed SOLVER_FAILED count, 13/40 —
no per-seed rung split exists to diff against.) `SOLVER_FAILED` seeds are now
2, 18, 23, 27 — and two of the six documented fit-gap seeds
(`OPEN_QUESTIONS` (f): 2, 18, 20, 23, 27, 32) are rescued by CCF
certification (20, 32). The CCF stage is reached on 25 of 40 seeds and
certifies 17 of them (68%); fixed_grid certifies 1 of 25 (4%).
σ usage (17 certifications): 4.0×9, 2.0×4, 1.0×2, 0.5×1, 0.25×1 — the full
ladder earns its keep. Branch kinds: under×11, over×6. MILP fired on 2 of 17
(seeds 29, 33; both certified). Solve wall-clock distribution (match battery,
n=6): min 0.135 s, median 0.226 s, p90 1.23 s, max 1.23 s; per-seed max
3.6 s (seed 33, the MILP conversion).

### Emitted length

CCF certifications: median 147 chars (89–300, n=17 per-seed; 37–206, n=6 in
matches) vs fixed_grid's 543 chars on the one shot it produced. The
certification cost is real but the emission is 3.7x shorter on the same
char-limit budget.

### `solve_cap_reached` (§9 cap: 24 solves vs 40 ladder slots)

5 certificate walks hit the cap (seeds 14, 17, 24, 27×2 — the last two are
seed 27's two targets). Only seed 27 actually reaches the CCF stage in the
solver ladder. Raised to 40 in memory for the probe: seed 27 completes the
full walk and stays UNCERTIFIED. **No ladder starvation — the cap is not
hiding certifications; left unchanged.**

### §13 acceptance decision

- CERTIFIED rate at the CCF stage (68%, 17/25 first-shot seeds) > fixed_grid
  (4%, 1/25) ✓
- Median emitted length 147 vs 543 chars ✓
- Zero CERTIFIED-but-collided in the property test (240 candidates, 7.7M
  samples) ✓
- `dud`/`SOLVER_FAILED` bucket materially shrunk: 13/40 → 4/40 seeds
  first-shot, 297 → 247 turns in matches ✓
- Kill clause "CCF < 10% of rungs": on the per-seed measurement unit CCF is
  17/40 = 42.5% of rungs — well above the kill line. The per-turn match
  reading (6/274 = 2.2%) is an artifact of stuck maps re-attempting the
  identical failed shot every turn and is not the histogram the criterion
  targets. EMIT_OVERFLOW: 0 anywhere (matches, per-seed, 200-map property
  test) ✓

**Decision: SHIP.** CCF stays as a ladder rung.

### Fixed-grid death row

`fixed_grid_gaussian` fired on first-shot seed 3 and on match seed 2005 — in
both cases only AFTER CCF had failed on the same turn, recovering a shot CCF
could not produce. CCF does not dominate it on the same seeds, so the rung
survives and the ladder stays at 5 rungs. Recorded for the next milestone:
both of fixed_grid's firings produce 500+ character emissions where every CCF
shot is ≤300; if §13 ever re-opens, the question is whether one extra
converted shot is worth the payload length.

## M5.3 — agent budgets (AST depth cap + simulate budget + accounting seams)

What shipped — three seams, zero behavioral change to any measured surface:

1. **AST depth cap** (`config.MAX_AST_DEPTH = 64`, `# TUNABLE`): the parser
   now rejects expressions whose evaluation tree exceeds 64 levels —
   `_reorder_rec` raises `MalformedFunction` past the recursion cap, and
   `PolishNotationFunction.__init__` computes the prefix list's tree depth
   iteratively (explicit stack) and rejects past the cap. Before this, a
   1000-term chain `1+1+…+1` (1999 chars, INSIDE `MAX_EXPR_CHARS`) raised
   `RecursionError` in `_reorder_rec` (probe: 990 terms parse, 1000 die at
   Python's 1000-frame limit), which sailed past every
   `except MalformedFunction` — `Game.fire` (state.py:242 is unguarded), the
   match runner, the UI server — and killed the match/page. Depth is the
   same missing-limit family as the char limit (OPEN_QUESTIONS (e), (k);
   the JVM dies with StackOverflowError on the same input). The guard
   inherits into every consumer at once: `Game.fire`, `agents.simulate_tool`,
   `solver._verify`, `ccf._fired_curve`, the UI server. Deep input is now a
   classified `PARSE_ERROR` turn (safe dud fired), a UI 400, and an oracle
   `error="MalformedFunction"` — never a crash. Behavior at depth ≤ 64 is
   unchanged; full-suite green and the M5.2 battery numbers below are
   reproduced unchanged (checked at close via the scratch `--out` rerun from
   `seeds.json`).
2. **Simulate budget wrapper** (`agents/simulate_budget.py`):
   `BudgetedSimulator` wraps the byte-identical pure oracle
   (`agents/simulate_tool.py` untouched — M5.4.2 depends on it). Per-turn
   count budget (default 3, `# TUNABLE`, mid-grid of the planned ablation),
   budget check FIRST, denied calls recorded without delegating, delegated
   calls count regardless of parseability, `new_turn()` ledger
   (`{"turn", "calls", "denied"}`), `remaining` for the M5.5.6 feedback
   format. Count-based only: NO wall-clock budget anywhere in M5.3.
3. **Accounting seams**: `AgentStats`/`AgentMatchStats` gain
   `simulate_calls`/`simulate_denied` (defaults 0; merged by the runner's
   existing `internal:` block). The accounting definition (units, per-turn
   granularity, distributions-not-means aggregation) is recorded in
   `docs/OPEN_QUESTIONS.md` ("M5.3: agent budgets"), resolving the
   "Genuinely open" token-cost item.

### Determinism / outcome-neutrality

- The budget is a call COUNT, so determinism holds by construction; the
  wrapper's ledger is a pure function of the (game, call-sequence) — tested.
- The depth cap is outcome-neutral at the shipped value: every expression the
  ladder emits parses at depth ≤ 16 (CCF peak ≈ 9 = ceil(log2 J_max) + 3,
  J_max = 55), asserted per rung on the M5.2 per-seed battery rungs
  (`tests/test_ccf.py`), and the `emission_budget` now reads BOTH limits
  (char 2000 binds; depth dimension admits J ≤ 2^61).
- **M5.2 battery numbers unchanged** — the close-out rerun
  (`python3 -m eval --from-seeds eval/results/seeds.json --out <scratch>`)
  reproduces the M5.2 section's match board and rung histogram exactly (15
  wins / 0.750, 178 shots, 247 SOLVER_FAILED, ccf=6, arc=8, the seed-2008
  3-turn flip), so the depth cap is outcome-neutral. The committed
  `eval/results/leaderboard.md` artifact was NOT regenerated (it is the
  M5.1-era baseline, pre-dating M5.2's solver changes — unchanged by M5.3),
  and the byte-parity leaderboard test stays green WITHOUT any artifact
  regeneration.

### Deliberately not run in M5.3

The simulate-budget ablation grid (N ∈ {0, 3, 10} × ASCII on/off): no
simulate-consuming agent exists (M3 skipped LLMAgent/HybridAgent), so every
wrapper integration number would be synthetic. The grid is the planned
M5.4/M5.5 protocol (OPEN_QUESTIONS, "M5.3: agent budgets"); the wrapper is
its infrastructure, with the `new_turn()` consuming-loop contract documented
in the class docstring.
