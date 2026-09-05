# Project: Graphwar Solver + LLM Agent Arena

Build a Python project that lets both (a) a deterministic optimizer and
(b) LLM agents play Graphwar — an artillery game where a shot's trajectory
is a mathematical function y = f(x) typed by the player.

## Phase 0 — Ground truth (do this FIRST, do not skip)

Clone and read the original source before writing any game logic:
  git clone https://github.com/catabriga/graphwar   # Java, GPLv3

Extract the following and write them to `docs/GROUND_TRUTH.md`, each with a
file path + line number citation. Where the source is ambiguous, say so
explicitly — do NOT guess and do NOT fill gaps with plausible numbers.

1. Coordinate system: map extents, origin, y-axis direction.
2. Whether f(x) is evaluated in a SHOOTER-RELATIVE frame (i.e. is the
   trajectory x_world = x_shooter ± u, y_world = y_shooter + f(u)?).
   Confirm the sign convention for shooters facing left vs right.
3. FIRING ANGLE: the tutorial ties the angle to differential-equation mode
   (y'' = f(x, y, y')). Determine from the source whether the angle has ANY
   effect in plain function mode. This is a load-bearing question — the
   solver design branches on the answer. Report what the code actually does.
4. Trajectory integration: step size in x, whether the projectile advances
   in fixed dx increments, and the exact termination conditions.
5. Collision: soldier hit radius, terrain collision test, damage model,
   splash radius if any, and friendly-fire rules.
6. Out-of-bounds: what happens when |y| exceeds the map, and whether the
   shot dies or wraps.
7. Parser: the EXACT grammar — allowed functions, operators, constants,
   precedence, and how division-by-zero / NaN / discontinuities are handled.
   Transcribe the whitelist from the tokenizer; do not reconstruct it.
8. Turn structure, wind (if any), and any per-turn randomness.

Also review prior art for technique (not for constants):
  - https://github.com/nico-fb/Graphwar-Bot        (stacked logistic terms)
  - https://github.com/edleebinj/graphwar-generator (DCT-based)
  - https://github.com/ADEHJKNTV/GraphingSolver

## Phase 1 — Headless simulator

`graphwar_sim/` — a faithful Python reimplementation of the trajectory and
collision engine, driven ENTIRELY by constants sourced in Phase 0. Every
constant in `config.py` carries a comment citing its Java origin.

- `parser.py`   — expression parser matching the game's grammar exactly.
                  Reject anything outside the whitelist. Safe eval, no exec.
- `physics.py`  — trajectory integration, collision, damage.
- `state.py`    — map, terrain, soldiers, teams, turn order.
- `render.py`   — matplotlib/PIL trajectory plot for debugging.

Validation gate: build `tests/golden/` with at least 20 scenarios where you
hand-verify the Python result against the Java game (run graphwar.jar, or
instrument the Java source with print statements). The simulator is not
considered done until golden tests pass. Report any divergence rather than
tuning the Python until it matches.

## Phase 2 — Deterministic solver

Implement a curve-fitting solver that produces a legal expression string
hitting a target while clearing terrain and teammates.

Approach:
- Sample the shooter→target corridor into waypoints that clear obstacles.
  Obstacle avoidance is a path-planning problem in the (u, y) half-plane.
- Fit a radial basis expansion:  f(u) = Σ_j w_j · exp(-(u - c_j)² / (2σ²))
  with centers c_j on a fixed grid. Fitting waypoints is then LINEAR in w,
  so solve as least squares (or a QP with an L1/L∞ penalty on w to keep the
  emitted string short and the curve tame).
- Also implement a logistic-basis variant, σ(u) = 1/(1 + e^(-k(u-c))), and
  benchmark the two. Prior art suggests logistic works well in practice.

Certification (this part is proven, keep it):
  Between adjacent samples spaced du, linear interpolation error obeys
      |g - P| ≤ M · du² / 8,      M = max |f''|
  For the Gaussian basis, substituting t = (u - c_j)/σ makes each term of
  f'' equal to w_j(t² - 1)e^(-t²/2)/σ², and |(t² - 1)e^(-t²/2)| attains its
  maximum of exactly 1 at t = 0. Therefore
      M ≤ σ⁻² · Σ_j |w_j|
  Use this to certify clearance: if the sampled path clears every obstacle
  by more than M·du²/8, the continuous path provably clears it too.

- Degradation ladder: if the primary fit fails to certify, fall back through
  progressively simpler families (fewer centers → wider σ → straight shot),
  and report which rung fired.
- Emit expressions that satisfy the Phase 0 grammar and stay under any
  character limit the game imposes.

## Phase 3 — LLM agent interface

`agents/` with a common protocol so any policy can play:

    class Agent(Protocol):
        def act(self, obs: Observation) -> str:  # returns an expression
            ...

- `SolverAgent`  — wraps Phase 2.
- `LLMAgent`     — serializes the board into a text/JSON prompt, asks a model
                   for an expression, validates it against the parser, and
                   retries with the parse error on failure (cap the retries).
- `HybridAgent`  — LLM proposes strategy/waypoints in structured form, the
                   Phase 2 solver turns them into a certified expression.
- `RandomAgent`, `StraightShotAgent` — baselines.

Observation must be model-agnostic: soldier positions, terrain polyline,
team assignment, HP, turn index, and the legal grammar. Include an ASCII
rendering of the board alongside the numeric data — worth measuring whether
it helps.

## Phase 4 — Evaluation harness

`eval/` — run agents head-to-head over seeded maps.

Metrics: win rate, hit rate, mean turns-to-kill, friendly-fire rate,
parse-failure rate, wall-clock and token cost per turn, and for the solver,
the distribution of degradation rungs used.

Output a markdown leaderboard plus per-match trajectory plots.

## Ground rules

- **Do not invent constants.** Every magic number traces to Phase 0 or is
  labeled `# TUNABLE — not from source` in `config.py`.
- If Phase 0 can't answer something, write the open question into
  `docs/OPEN_QUESTIONS.md` and design around the uncertainty. Don't paper
  over it.
- Python 3.11+, numpy/scipy, pytest, ruff, full type hints.
- Note the original is GPLv3 — keep the Python reimplementation clean-room
  where practical, and document the license position in the README.
- Commit at each phase boundary. Report Phase 0 findings before starting
  Phase 1; if the angle question resolves differently than expected, flag it
  rather than proceeding on the assumption.