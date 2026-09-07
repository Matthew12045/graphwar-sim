# Graphwar — Python reimplementation

A headless, faithful reimplementation of the artillery game **Graphwar**, in
which a shot's trajectory is the graph of a function `y = f(x)` typed by the
player. The goal is to reproduce the behavior of the original Java client
(`ref/graphwar/`, GPL-licensed) closely enough to (a) run reproducible games
and (b) later plug in LLM agents and evaluate them.

This is **Milestone M1**: a headless simulator plus a golden-test suite that
checks the Python physics against the compiled Java reference, shot for shot.

## Status

| Milestone | Scope | Status |
|-----------|-------|--------|
| M0 | Ground truth from the Java source | ✅ done — see `docs/GROUND_TRUTH.md` |
| M1 | Headless simulator + golden tests | ✅ done — see below |
| M2 | Deterministic solver | ✅ done — degradation ladder, 33/40 hit rate on the seeded battery |
| M3 | Agents (minimal slice: solver + baselines; LLM agents deferred) | ✅ done — `agents/`, frame round-trip test |
| M4 | Eval harness (win rate + hit rate only) | ✅ done — `eval/`, see `eval/results/leaderboard.md` |
| UI | Interactive web client (NORMAL_FUNC) | ✅ done — `ui/`, visual spec in `docs/UI_GROUND_TRUTH.md` |
| M5.4 | LLMAgent (Anthropic tool-use) + spectator UI | ✅ done — `agents/llm_agent.py`; roster entry `llm:<model>`; UI team modes + autoplay/step |

## Scope

Only **NORMAL_FUNC** mode is in scope: the trajectory is `y = f(x)`. The two
ODE modes (FST_ODE / SND_ODE) are out of scope for M1.

## License & clean-room position

The reference implementation is licensed under **GPLv3** and lives in
`ref/graphwar/` (a separately-cloned git repo, excluded from this repository's
VCS via `.gitignore`).

This project is a **clean-room reimplementation**:

- We **cite** the Java source (`file:line`) to document *what* the behavior is,
  but we do **not copy** its code. Each Python module carries comments naming
  the Java method and line range it ports.
- No GPL-licensed code is included in this repository.
- The Python code in this repository is released under **MIT** (see
  `pyproject.toml`).

This keeps the two codebases legally separable: the reference is cited as a
specification, not vendored.

## Layout

```
graphwar_sim/
  config.py    # constants + token types, each cited to Java file:line
  parser.py    # prefix-notation function parser/evaluator (no eval/exec)
  physics.py   # process_function_range: the shot integration + hit test
  state.py     # Game / GameState / Team: turn order, win rule, seeded map
  render.py    # headless matplotlib rendering (terrain, soldiers, trajectory)
agents/
  base.py          # Agent protocol, Observation (centered world frame), stats
  observation.py   # observe(game): board -> world-frame obs + ASCII map
  simulate_tool.py # fire a candidate through the real physics, no kills applied
  simulate_budget.py # BudgetedSimulator: per-turn simulate budget + ledger
  baselines.py     # RandomAgent, StraightShotAgent, Bot67Agent (roster key "67")
  solver_agent.py  # wraps the deterministic M2 solver
  llm_agent.py     # LLMAgent: Anthropic tool-use over the budgeted simulate tool
  personas/        # M5.4 persona harness: manifest + adapted style texts + verifiers
  waypoints.py     # M5.5.3 waypoint plan schema (the only thing a hybrid LLM emits)
  hybrid_agent.py  # HybridAgent (M5.5): LLM plan -> certified CCF shot
  emission.py      # format_literal: plain-decimal emission (no exponents)
eval/
  runner.py        # seeded round-robin match runner + leaderboard writer
  metrics.py       # win rate / hit rate roll-up
  __main__.py      # cli: run a fresh leaderboard or reproduce from seeds.json
  results/         # committed leaderboard + per-match plots + seeds.json
ui/
  server.py        # FastAPI wrapper: /api/new_game, /api/state, /api/fire,
                   # /api/agent_turn (M5.4 spectator team modes)
  static/          # no-build-step frontend: canvas redraw of the Java game screen
                   # + match-setup panel / playback bar (play, pause, step, speed)
tools/
  golden/Graphwar/GoldenShot.java   # in-reference harness that dumps shots as JSON
  jar_probe/                        # one-off terrain probes
  time_tests.py
planning/          # milestone specs & plans (graph_war.md, IMPLEMENTATION_PLAN.md,
                   # 5.2.md, The_bridge_nobody_wrote_down.md, PROGRESS_REPORT.txt)
sketches/          # superseded design drafts — see sketches/README.md
personas/          # shared LLM prompt + style prompts for agent-based play
tests/
  golden/    # ≥20 scenarios compared against graphwar.jar output
  test_agents.py  # M3: frame round-trip, agent contract, full-match no-crash
  test_eval.py    # M4: determinism, reproducibility from the seed file
  test_ui_server.py # UI: API contract via FastAPI TestClient
docs/
  GROUND_TRUTH.md   # M0: the behavior spec with file:line citations
  UI_GROUND_TRUTH.md # UI: the game client's look & behavior, with citations
  OPEN_QUESTIONS.md # reported divergences & carried-forward assumptions
```

## Evaluating agents (M4)

Run a fresh leaderboard from a root seed (round-robin across the roster):

```
python3 -m eval --root 1000 --matches 5 --out eval/results
```

Reproduce the committed leaderboard **from its seed file alone**:

```
python3 -m eval --from-seeds eval/results/seeds.json --out eval/results
```

M5.4 adds an `llm:<model>` roster entry (opt-in — `DEFAULT_ROSTER` stays
network-free): pass e.g. `roster=["solver", "llm:gpt-x"]` to
`eval.runner.build_plan` and set the auth env vars (`ANTHROPIC_AUTH_TOKEN`
Bearer — the Claude Code / gateway convention — or `ANTHROPIC_API_KEY`
x-api-key, plus optional `ANTHROPIC_BASE_URL`). The `anthropic` SDK is the
optional `[llm]` extra. The `67` bot baseline is roster-selectable by key
`"67"` (a guaranteed dud, so it stays out of the default round-robin).

## Install

```bash
python3 -m pip install -e ".[dev]"
```

Requires Python 3.11+, `numpy`, `scipy`, `matplotlib`. Dev extras add
`pytest`, `ruff`, `mypy`; the `llm` extra adds `anthropic` (only needed by
`agents/llm_agent.py`, imported lazily).

## Play it (web UI)

```bash
python3 -m pip install -e ".[ui]"
python3 -m uvicorn ui.server:app --reload
```

Open http://127.0.0.1:8000. This is a clean-room **redraw** of the original
Java client's game screen (NORMAL_FUNC only): type `y = f(x)`, press Fire,
and watch the shot trace the curve. The `New Match` button takes an optional
seed (the input next to it) for reproducible maps. Two honest repurposings,
called out in `docs/UI_GROUND_TRUTH.md` §8: the reference's multiplayer chat
box is now a per-turn match log, and there is no turn countdown (the
simulator has no turn clock).

M5.4 adds spectator modes in the panel below the game box: pick a driver per
side (Human / Solver / Random / Straight Shot / LLM with a model name),
a max-turns draw cap, and a playback bar (Play/Pause, single-Step, and a
speed selector that scales only the inter-turn delay). Agent sides are
played one turn per request through `/api/agent_turn`; the log names the
agent behind every shot (`Player 1 (Solver): y = ... → HIT`).

## Running the golden tests

The golden tests compare the Python simulator's shot output against the Java
reference. Two parts:

1. **Reference capture** (needs a JVM + the reference build).
   `tools/golden/Graphwar/GoldenShot.java` is compiled into the reference's
   `bin/` and, for each scenario, fires one shot through the reference
   `Function.processFunctionRange`, printing the result as a JSON line (and,
   when terrain circles are present, the exact `collidePoint` grid).
   `tools/golden/generate_golden.py` drives it and writes the captured
   reference results to `tests/golden/data/golden.json`.

   ```bash
   python3 tools/golden/generate_golden.py
   ```

2. **Parity check** (pure Python, no JVM). The pytest suite in
   `tests/golden/` replays each scenario's inputs in Python and asserts parity
   against the checked-in `golden.json` on `numSteps`, `lastX`/`lastY`, the hit
   list, and the full trajectory.

   ```bash
   pytest tests/golden/
   ```

Because the reference output is checked in, `pytest tests/golden/` runs without
a JVM. Re-run step 1 only to regenerate the reference after the reference
changes.

**Divergence is reported, not tuned away.** If a golden test diverges from the
reference, the difference is recorded in `docs/OPEN_QUESTIONS.md` rather than
silently adjusted in the Python port.

## Security note (for M3)

The function parser is a security surface: LLM-supplied strings must never
reach `eval`/`exec`/`sympify`. The parser is a hand-written tokenizer +
recursive-descent evaluator over a fixed whitelist of operations
(`parser.py`). A security review is required before M3 ships any LLM agent.
