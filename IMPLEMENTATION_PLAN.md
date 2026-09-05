# Graphwar Solver + LLM Agent Arena — Implementation Plan

## Context

`graph_war.md` specifies a Python project where (a) a deterministic optimizer
and (b) LLM agents play Graphwar — an artillery game where a shot's trajectory
is a function `y = f(x)` typed by the player. The directory already contains a
solver sketch (`Implementation.py`), a simulate-tool sketch
(`3._Give_it_a_simulate_tool.py`), a shared LLM prompt, and seven style prompts.

The plan sequences the spec's five phases (0 ground truth → 1 simulator →
2 solver → 3 agents → 4 eval). **Ground truth was pulled from the real Java
source (`catabriga/graphwar`, master) and it contradicts assumptions baked into
the existing prompt files and the solver sketch.** Those corrections are
load-bearing and must propagate before Phases 2 and 3 are built.

### Three source-confirmed corrections (from `catabriga/graphwar`)

1. **The frame is NOT shooter-relative.** It is a **centered world frame** with
   an **automatic vertical offset**. `Function.java` sets `valuesX[0]` to the
   map *center* (x=0 is the middle of the map, not the shooter), steps forward
   by `STEP_SIZE`, and adds `offSet = -f(x0) + y0` so the curve is forced
   through the soldier. Left-facing is a **mirror** (`x = PLANE_LENGTH - x`).
   README confirms: "the fired graph is that function translated by a constant
   so it passes through your soldier: `y = f(x) + c`."
   → The core prompt's "soldier sits at (0,0)", "x from 0 to MAX_LENGTH", and
   "f(0) within 0.5 of 0" are **wrong**; every style prompt's closed form
   (Sniper `f(x)=(h/d)x`, Howitzer `-a·x(x-L)`, …) is in the wrong frame.
   **All prompts must be rewritten from source.**

2. **Firing angle has NO effect in plain function mode.** It is only used in
   Second-Order DE mode. `getStartAngle` is a display/ODE finite-difference.
   → The sketch's **Stage 0 angle scan and all `rotate()` calls are dead code**
   for function mode. The solver is a *single* fit, not an angle loop.

3. **Hit radius is `SOLDIER_RADIUS = 7`**, not "BLAST_RADIUS".
   `EXPLOSION_RADIUS = 12` is the *visual* explosion, not the hit test. The hit
   test is `distSquared < SOLDIER_RADIUS²` (strict `<`), and the trajectory
   **continues past a hit** (no `break`) — so one shot can hit multiple
   soldiers. This validates the Master Magician multi-kill premise; "BLAST_RADIUS"
   in the prompts is a misnomer.

### Other source facts that change the design
- **Parser is shunting-yard → Polish notation**, not recursive descent.
  Tokenizer regex (verbatim): `[0-9]*\.?[0-9]+|\(|\)|x|y'|y|\+|\*|/|\^…|sqrt|log|abs|sin|sen|cos|tan|tg|-|ln|e|pi`.
  Pre-substitutions: lowercase, `-`→`+`+`-`, `exp`→`e^`, `,`→`.`. Implicit
  multiplication (`2x`→`2*x`). `e`=Math.E, `pi`=Math.PI. `MalformedFunction.java`
  holds the exact error strings (Phase 3 feeds these back to the LLM).
- **Integration is adaptive, not fixed-dx.** `STEP_SIZE=0.01`; the step **halves**
  when a step's distance² > `FUNC_MAX_STEP_DISTANCE_SQUARED=0.001`, floor
  `FUNC_MIN_X_STEP_DISTANCE=1e-5`; `FUNC_MAX_STEPS=20000`. Termination:
  non-finite y, obstacle collision, or step-halving exhausted.
- **Real constants** (all in `src/GraphServer/Constants.java`): `PLANE_LENGTH=770`,
  `PLANE_HEIGHT=450`, `PLANE_GAME_LENGTH=50` (world x∈[−25,25], y∈[−15,15]),
  `SOLDIER_RADIUS=7`, `INITIAL_NUM_SOLDIERS=2`, `MAX_SOLDIERS_PER_PLAYER=4`,
  `TURN_TIME=60000`.

---

## Phase 0 — Ground truth (confirm + correct)

Read, in this order (skip all UI/server files beyond `Constants.java`):
1. `src/GraphServer/Constants.java` — all magic numbers in one place.
2. `src/Graphwar/Function.java` — frame remap, `offSet`, mirror, adaptive
   integration, soldier/terrain collision. Resolves facts 1, 2, 2b, 3, 4, 5.
3. `src/Graphwar/PolishNotationFunction.java` + `FunctionToken.java` +
   `MalformedFunction.java` — parser grammar + exact error strings (fact 6).
4. `src/Graphwar/GameData.java` + `RoomBoard.java` — turn structure, win
   condition, per-turn randomness (fact 7).

**Write:**
- `docs/GROUND_TRUTH.md` — the 8 spec facts with `file:line` citations, **plus a
  "Corrections to draft assumptions" section** recording the three findings above.
- `docs/OPEN_QUESTIONS.md` — (a) coordinate space/units of the `distSquared`
  hit test (game-space doubles vs plane pixels); (b) whether `soldierCollides`
  5-point sampling applies to projectile hits or only placement; (c) exact win
  condition and whether a multi-hit shot kills all hit soldiers or just the first.

**Gate:** report the three corrections to the user **before** Phase 1 — they
invalidate the prompt files and the sketch's angle loop.

---

## Phase 1 — Headless simulator (`graphwar_sim/`)

```
graphwar_sim/
├── config.py     # real constants from Constants.java, cited; sketch-only
│                 # numbers labeled # TUNABLE — not from source
├── parser.py     # faithful port of the Java shunting-yard → Polish notation
├── physics.py    # adaptive-step integration, SOLDIER_RADIUS hit, terrain, no-wrap OOB
├── state.py      # deterministic seeded map, soldiers, teams, turn order
└── render.py     # matplotlib terrain + soldiers + trajectory
```

- **`parser.py`** — hand-write a faithful port of `PolishNotationFunction.java`
  (tokenizer + shunting-yard), **not** a library (sympy/asteval/numexpr). A
  library accepts/rejects different inputs and emits different errors, so the
  LLM would train against a grammar that isn't the game's. Build a small AST
  from the Polish notation and evaluate with numpy (vectorized over `valuesX`).
  No `eval`/`exec`/`sympify` in the hot path. Reproduce the exact parse-error
  strings from `MalformedFunction.java`.
- **`physics.py`** — model the **adaptive step-halving** loop explicitly (the
  biggest fidelity risk; fixed-dx diverges on steep curves). Hit test
  `distSquared < SOLDIER_RADIUS²`, continue past a hit, skip active shooter +
  dead. Terrain via the plane/terrain model. Shot dies (no wrap) on leaving the
  plane or non-finite y.
- **Reuse:** `3._Give_it_a_simulate_tool.py`'s `simulate()` loop is a good
  *driver* skeleton, but replace its sympy `compile_fn` with the ported parser
  and its `dt=0.05, max_len=60` with `STEP_SIZE=0.01` + adaptive halving.

**Golden tests (the gate):** `tests/golden/` ≥ 20 scenarios, hand-verified
against `graphwar.jar` (build via `make`). Must include the three correction
cases: (a) a left-facing shooter (mirror path), (b) a function with large `f(0)`
that still passes through the soldier via the auto-offset (proves the frame
model), (c) a steep curve that triggers step-halving (proves adaptive
integration). Plus a tokenizer accept/reject table from the Java source.
**Report divergence; do not tune Python to match.**

**Acceptance:** `pytest tests/golden/` green vs `graphwar.jar`.

---

## Phase 2 — Deterministic solver (`graphwar_sim/solver.py`)

Produces a legal expression string that hits targets, clears terrain, misses
teammates — with a proof of clearance. **Major simplification from the sketch:**

- **Delete the Stage 0 angle scan and all `rotate()` calls** (no firing angle
  in function mode). One fit, not an angle loop.
- **Work in centered world x** (x=0 at map center), not `u∈[0,L]`.
- **Fix the offset constraint.** The sketch's `exp(-b·C²)@w == 0` (i.e. `f(0)=0`)
  is wrong — it forces the curve through the map center. The game auto-adds
  `c = -f(x0) + y0`; the solver must evaluate the *translated* curve `f(x)+c`
  at obstacles/targets and is free to let `f(0)` be anything. Bake the offset
  into the objective/constraints.
- **Certification stays** (frame-independent): `|g−P| ≤ M·du²/8`,
  `M ≤ 2b·Σ|w_j|` for the Gaussian basis. On failure, halve `dx`, then fall
  through the degradation ladder.

**Basis approach (benchmark all three, pick a default):**
1. **Fixed-grid linear least-squares** — the natural *primary* once the angle
   loop is gone: `f = Σ w_j φ_j` is linear in `w`, hitting waypoints is plain
   `lstsq`, clearance/friendly constraints are linear inequalities. Cheap,
   robust, no cvxpy required → **the always-available floor**.
2. **Per-target-center curvature-regularized QP** (the sketch, via cvxpy/OSQP)
   — the high-precision path.
3. **Logistic basis** `σ(u)=1/(1+e^{−k(u−c)})` — prior-art baseline.

**Degradation ladder** (aligns with the Master Magician protocol): full
multi-target fit → fewer centers/wider σ → closed-form single-target rungs
(straight line → single arc `a·x(x−L)` → sigmoid step, **re-derived in the
centered world frame with the auto-offset**) → deliberate safe dud. Record which
rung fired (a Phase 4 metric).

**Emission:** `to_graphwar()` emits `e^(−b(x−c)²)` (valid — the game rewrites
`exp`→`e^`). Add a **round-trip test**: `parse(to_graphwar(w))` reproduces the
intended curve.

**Acceptance:** on a seeded-map battery, the solver emits parseable, certified
expressions; hit rate + degradation-rung distribution logged; round-trip green.

---

## Phase 3 — LLM agents (`agents/`)

```
agents/
├── base.py          # Agent Protocol, Observation dataclass (frame handling)
├── observation.py   # board → model-agnostic obs (world frame + ASCII render)
├── simulate_tool.py # Phase-1-backed simulate() (NOT sympy)
├── prompts/
│   ├── core.txt     # rewritten from source: strike f(0)≈0.5, |y|≤200,
│   │                #   BLAST_RADIUS, MAX_LENGTH; use y∈[−15,15],
│   │                #   SOLDIER_RADIUS=7, plane bounds, centered frame
│   └── styles/      # sniper, howitzer, serpent, bodyguard, professor,
│                    #   gremlin, master_magician — personas survive,
│                    #   closed forms re-derived in the correct frame
├── solver_agent.py  # wraps Phase 2, no LLM
├── llm_agent.py     # core + one style prompt; parse-error retry loop
├── hybrid_agent.py  # LLM waypoints (JSON) → Phase 2 certifies
└── baselines.py     # RandomAgent, StraightShotAgent
```

- **`Observation`** must present the board in the **centered world frame** (or
  explicitly convert to shooter-relative *and tell the model the frame*). If a
  shooter-relative frame is used for ergonomics, the `simulate()` tool and the
  committed expression must be in the *world* frame — the agent layer does the
  conversion. **This frame conversion is the highest-risk Phase 3 integration
  point; add a frame round-trip test before any LLM runs.**
- **`LLMAgent`** — one class parameterized by style file (six instances, not six
  code paths). Loop: model → expression → parse-validate → (on error: feed the
  exact game error back, retry) → simulate → commit. Cap retries; on exhaustion
  emit the safe dud.
- **Reuse:** the seven style files → `styles/*.txt` (post-audit);
  `The_Master_Magician.txt` → `styles/master_magician.txt` **and** its
  comb/Lagrange/staircase methods seed the closed-form degradation rungs in
  Phase 2 (cross-phase reuse).

**Acceptance:** every agent plays a full seeded match headless without crash;
frame round-trip test green; parse-failure/retry counts logged.

---

## Phase 4 — Evaluation harness (`eval/`)

- **Match runner** — seeded map generation (seed set saved to disk), round-robin
  across the agent roster, N matches per pair.
- **Metrics** — win rate, hit rate, mean turns-to-kill, friendly-fire rate,
  parse-failure rate, wall-clock per turn, token cost per turn; solver-only:
  degradation-rung distribution. **Ablations:** ASCII board on/off; simulate
  budget N ∈ {0, 3, 10}.
- **Output** — `eval/results/leaderboard.md` + per-match trajectory plots via
  `render.py`.

**Acceptance:** leaderboard + plots for the full roster; reproducible from the
seed file alone.

---

## Scaffolding (once, up front)

`pyproject.toml` (Python 3.11+, numpy, scipy, cvxpy, matplotlib, pytest, ruff;
full type hints), `docs/`, `graphwar_sim/`, `agents/`, `eval/`, `tests/golden/`,
`README.md` (GPLv3 clean-room license position). Commit at each phase boundary.

---

## Recommended first milestone — minimal viable slice

Get a **working `SolverAgent` beating baselines before any LLM is involved** —
fully deterministic and testable, no LLM dependency:

**M0** (ground truth + report corrections) → **M1** (simulator + golden tests)
→ **M2** (fixed-grid least-squares solver only; skip QP/logistic) → **M3**
(`SolverAgent` + `RandomAgent` + `StraightShotAgent` only; skip `LLMAgent`/
`HybridAgent`) → **M4** (match runner with win-rate + hit-rate only; skip
token-cost/ablations).

This delivers a certified, parseable, frame-correct solver playing real headless
matches against baselines — the core of the project. The LLM agents and full
eval metrics/ablations are then pure additions on top.

---

## Top risks & de-risking

| # | Risk | De-risk |
|---|------|---------|
| 1 | Frame model wrong (centered-world + auto-offset + mirror) | Confirmed from source; dedicated golden tests (left-facing + large `f(0)`) in M1; frame round-trip test in M3 |
| 2 | Adaptive step-halving not modeled | Model the halving loop in `physics.py`; steep-curve golden case that requires halving |
| 3 | Parser behavior mismatch (implicit mult, `e^`, aliases, error strings) | Port the shunting-yard (not a library); tokenizer accept/reject table golden test |
| 4 | Hit-test coordinate space/units ambiguous | Resolve at line level in M0; if ambiguous, OPEN_QUESTIONS + bracket with boundary golden tests (7.0 vs 7.1) |
| 5 | cvxpy/OSQP unavailable or slow | Least-squares is the always-available floor; QP is an enhancement; wall-clock budget in `config.py` falls back to LS |

## Milestones

1. **M0** — `GROUND_TRUTH.md` + `OPEN_QUESTIONS.md`; three corrections reported.
2. **M1** — simulator; ≥20 golden tests pass vs `graphwar.jar`.
3. **M2** — solver: certified multi-target shots + degradation ladder + basis benchmark.
4. **M3** — all agents play full seeded matches headless.
5. **M4** — leaderboard + per-match plots + ablations.
