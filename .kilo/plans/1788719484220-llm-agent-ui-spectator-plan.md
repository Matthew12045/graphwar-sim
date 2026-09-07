# M5.4 (Phases 6-8): LLMAgent + eval registration + UI spectator modes

Repo `/Users/matthurindo/graph_war`, branch `main` at `5e085e4` (clean).
Three slices, one commit each, subjects prefixed `M5.4:` (per handoff
convention: one slice per commit, never mix unrelated files).

## Environment (run first — AGENTS.md rule)

1. `python3 -m pip show graphwar-sim` → Editable project location MUST be
   `/Users/matthurindo/graph_war`. If not: `python3 -m pip install -e . --no-deps`
   from the repo root (never from a worktree).
2. Gate order per slice: pip show check → `python3 -m mypy --strict graphwar_sim/ agents/ eval/`
   → `python3 -m ruff check graphwar_sim agents eval` + `python3 -m ruff format --check`
   on those dirs → `python3 -m pytest tests/ -q --ignore=tests/golden` (exit code
   is the authority). Baseline: 196 tests green.
3. No `rg` on shell; zsh; macOS spawn breaks stdin heredocs — write probe
   scripts to real files in `/var/folders/f0/j5cm67kx7jx6tt2yd7zs8_fw0000gn/T/kilo/`.

## User-locked decisions

- **Prompt frame**: adapt the shared core prepend
  (`Shared_core_prepend_to_every_bot_.txt`) to the engine's REAL frame; do not
  paste verbatim. Persona style files are OUT of this phase (they belong to the
  separate personas-migration workstream; `The_bridge_nobody_wrote_down.md`).
- **Simulate tool**: wrapped in `BudgetedSimulator` (M5.3) with
  `DEFAULT_SIMULATE_BUDGET` (3/turn). Denials feed a tool_result error and
  double as the loop's cost stop-signal. Counters merge into match stats
  automatically (runner's `internal:` block) — zero runner changes needed.
- **Safe dud**: `"0*x"` (repo convention: `eval/runner.py:227`,
  `agents/baselines.py:57,92`), NOT the handoff's literal `"x"` example — the
  handoff itself points at the existing safe-dud convention, and `0*x` is it.
- **Model/gateway** (user's `~/.claude-9arm.json`): the LLM is
  `qwen3.8-27b-fp8` served by the 9arm gateway (`https://gateway.9arm.co`),
  which speaks the Anthropic Messages API (Claude Code convention:
  `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN`). The `anthropic` SDK plan
  stands — the client must honor those env vars. NEVER copy the auth token
  value into the repo, docs, or code.

---

## Slice 1 — `agents/llm_agent.py` (commit `M5.4: LLMAgent ...`)

### New file `agents/llm_agent.py`

`LLMAgent(model: str, max_attempts: int = 4, client: Any | None = None)`
satisfying `agents/base.py::Agent` (name / act / stats).

- `self.name = f"llm:{model}"` — unique per model, so eval stats keys and the
  UI match log distinguish sides (known limitation: llm:X vs llm:X mirror
  collides in `play_match`'s stats dict; round-robin rosters can't produce it —
  document in the docstring, don't over-engineer).
- `client=None` → `_build_client()` lazily does `import anthropic` and
  constructs the client honoring the 9arm/Claude Code env convention
  (user-locked decision): `base_url=os.environ["ANTHROPIC_BASE_URL"]` when
  set; auth from `ANTHROPIC_AUTH_TOKEN` (Bearer, the 9arm gateway's var)
  falling back to `ANTHROPIC_API_KEY` (x-api-key). Raise a clear
  `RuntimeError` at construction if neither token var is set, naming BOTH
  env vars — fail at construction, never mid-match. Tests inject fakes via
  `client=`.
- anthropic is an OPTIONAL dependency: no module-level import. Add mypy
  override `[[tool.mypy.overrides]] module = "anthropic.*"
  ignore_missing_imports = true` so the strict gate passes with or without the
  extra installed.

**System prompt** (adapted core prepend; real frame, real SimResult, exact
grammar). Skeleton:

```
You are a Graphwar shot generator. You control one soldier.

FRAME (world coordinates — this is the exact space your expression is
evaluated in): x in [-25, 25], y up. Your soldier 'M' sits at (sx, sy),
given per turn; the projectile starts there and travels toward increasing
x. The game shifts your curve VERTICALLY so it passes through the muzzle:
effective curve = f(x) + (sy - f(sx)). Aim by choosing f so the shifted
curve passes within hit radius (~0.45 world units) of an enemy's (x, y).
Hitting a teammate is a critical failure. Terrain ('#' cells) ends the
shot harmlessly. The game nudges the launch point along the curve's own
tangent before the offset.

SYNTAX (the parser's exact tokenizer whitelist — nothing else exists):
numbers, ( ) x + - * / ^, functions sqrt log (base 10) ln abs sin sen cos
tan tg, constants e pi. Variable x ONLY (no y, no y'). Implicit
multiplication works (2x, x(x+1), 2sin(x)). '-' is always unary (a-b is
parsed as a+(-b)). Unknown characters are silently DROPPED by the
tokenizer. Max 2000 chars; deeper than 64 nested terms is rejected
(MalformedFunction → your shot is replaced by a safe dud).

TOOL: simulate(expr) fires a candidate through the real physics WITHOUT
applying kills. Returns {parseable, hit_enemy, hit_teammate, num_hits,
error}. You have N calls this turn. Use them. Revise. Then commit.

OUTPUT: after your final simulate call, emit ONLY the bare expression on
one line (no "y =", no prose, no code fence).
```

Grammar text must quote `docs/GROUND_TRUTH.md:264` content, not embellish.

**Per-turn user message**: `obs.ascii_board` + numeric serialization of
`obs.shooter`, `obs.own_soldiers`, `obs.enemy_soldiers`, `obs.terrain_blocks`
(world-frame tuples, 1-decimal rounding; terrain compressed — e.g. sorted
pairs, one per line or a compact run — it's the coarse 15px grid). Frame is
exactly what every other agent reads — do NOT re-derive or shift coordinates.
State the per-turn simulate budget ("simulate calls remaining: N" — M5.5.6
feedback format, fed by `BudgetedSimulator.remaining`).

**Tool schema** (Anthropic tool-use):

```python
{"name": "simulate",
 "description": "Fire a candidate y=f(x) through the real physics WITHOUT "
                "applying kills. Probe before committing.",
 "input_schema": {"type": "object",
                  "properties": {"expr": {"type": "string"}},
                  "required": ["expr"]}}
```

**act(game, obs) loop**:

1. `sim = BudgetedSimulator(game)`; call `sim.new_turn()` at turn start
   (contract: wrapper's docstring; first call closes an empty turn 0 —
   harmless, counters stay correct).
2. Fresh conversation per turn (no cross-turn memory).
3. One ATTEMPT = the model producing a final (non-tool_use) response whose
   last non-empty line is extracted as the candidate (strip whitespace,
   optional leading `y =`/`y=`, code-fence markers). Validate with
   `PolishNotationFunction(expr)`.
4. Within an attempt, while `response.stop_reason == "tool_use"`: for each
   tool_use block, call `sim.simulate(block.input["expr"])` → JSON-ish
   string of the `SimResult` fields; `SimulateBudgetExhausted` → tool_result
   error text "simulate budget exhausted — commit your best expression now".
   Cap total API round-trips per turn at `_MAX_TOOL_ROUNDS_PER_TURN = 8`
   (# TUNABLE) — breach ⇒ treat as a failed attempt.
5. Parse failure: `stats.parse_failures += 1`, `stats.retries += 1`
   (same counters RandomAgent exposes), append a user correction message
   quoting the reference-style error (`type(exc).__name__` — the reference
   exception is message-free), re-call. Cap at `max_attempts` total
   attempts; exhaustion ⇒ return `"0*x"` (safe dud).
6. `stats()` maps: parse_failures, retries, `simulate_calls = sim.calls_used`,
   `simulate_denied = sim.denied_used`.

### `agents/__init__.py`

Remove LLMAgent from the "deferred" note (keep HybridAgent deferred), import
+ export `LLMAgent` (safe: no module-level anthropic import).

### `pyproject.toml`

`[project.optional-dependencies] llm = ["anthropic>=0.40"]` (NOT base deps)
+ the mypy override above.

### `tests/test_llm_agent.py` (new; fake client, zero network)

Fake client mirrors ONLY the minimal surface used: `messages.create(...)`
returns an object with `.stop_reason` and `.content` blocks
(`.type == "text"` w/ `.text`; `.type == "tool_use"` w/ `.id`, `.name`,
`.input`). Scripted response sequences. Cover:

1. Well-formed first response → expression returned verbatim, 0 counters.
2. Malformed then corrected → retries=1, parse_failures=1, corrected expr
   fired; assert the correction message contained the exception name.
3. Exhaustion (all attempts malformed) → returns `"0*x"`, counters = attempts.
4. Tool-use round: simulate block routed through `BudgetedSimulator`
   (tool_result carries SimResult fields; `stats().simulate_calls == 1`).
5. Budget denial → denial error text returned; `simulate_denied == 1`;
   no delegation (assert the pure oracle was NOT called for the denied expr).
6. Constructor without `ANTHROPIC_AUTH_TOKEN`/`ANTHROPIC_API_KEY`
   (monkeypatch delenv both, no injected client) → clear error naming both.
7. System prompt contains the exact whitelist tokens (spot-check
   "sqrt", "log", "ln", "abs", "sin", "cos", "tan", "x ONLY").

---

## Slice 2 — eval registration (commit `M5.4: llm:<model> roster ...`)

### `eval/runner.py`

- Extract module-level `make_agent(name: str, seed: int) -> Agent`:
  exact-string factory lookup first (`_AGENT_FACTORIES`); else `name`
  starts with `"llm:"` → split on the FIRST `":"`, construct
  `LLMAgent(model=<rest>)` (seed unused; LLM agents are deterministic given
  the model). Unknown → `ValueError(f"unknown agent in roster: {name}")`.
- `_make_agents` delegates to `make_agent` twice. `DEFAULT_ROSTER` UNCHANGED
  (headless default stays network-free). Import LLMAgent lazily inside the
  llm branch OR top-level (top-level is safe — no anthropic import at module
  import).

### `tests/test_eval.py` additions

- `make_agent("llm:fake-model", seed)` with an injected fake client →
  instance name `"llm:fake-model"`, `play_match` runs it vs `"solver"` end
  to end with zero network (fake always returns a valid expr) and stats
  keys carry `llm:fake-model`.
- Unknown roster name → ValueError message.

---

## Slice 3 — UI game modes + spectator controls (commit `M5.4: UI team modes ...`)

### `ui/server.py`

- `NewGameBody` gains `team_modes: {team1: str, team2: str} | None` and
  `max_turns: int | None`. Modes validated against
  `{"human"} | set(_AGENT_FACTORIES) | llm:<model>` prefix — reuse
  `eval.runner.make_agent` directly (do NOT duplicate the mapping). Defaults
  `"human"`/unlimited when omitted (today's behavior preserved).
- Module state: `_team_modes: dict[int, str]`, `_team_agents: dict[int, Agent]`,
  `_turns_played: int`, `_max_turns: int | None`. Agents constructed EAGERLY
  in `new_game` (under the lock) so a missing `ANTHROPIC_API_KEY` or unknown
  mode fails fast as a 400 with detail.
- `POST /api/agent_turn` (no body): under lock; finished ⇒ 409 (same shape as
  fire); current side is human ⇒ 409 `{"error": "human_turn"}`. Otherwise:
  `obs = observe(game)`; `expr = agent.act(game, obs)`; classify with the
  runner's own logic (`hit_team_counts` + `_classify` via
  `_peek_solver_rung` — import, don't reimplement); fire via
  `game.play_turn(expr)`, defensive `MalformedFunction` → safe dud
  `game.fire("0*x")` + `advance_turn()` (mirror runner.py:223-228);
  `_turns_played += 1`. Response = EXACTLY the `/api/fire` payload shape
  (shot/board/game_over/winner/shooter/func_str/start_angle/seed) PLUS
  `agent` (name), `outcome` (`ShotOutcome.value`), `solver_rung` (nullable).
  One turn per request — never a whole match synchronously.
- Turn cap: when `not game.finished() and _turns_played >= _max_turns` →
  respond `game_over: true, winner: null, draw_reason: "TURN_CAP"` (both
  `/api/fire` and `/api/agent_turn`); board JSON gains `turns_played` and
  `max_turns`; `/api/state` and `/api/new_game` responses include
  `team_modes` so the frontend can resync.

### `ui/static/` (single animation pipeline — no second path)

- `index.html`: match-setup panel — mode `<select>` per side (Human / Solver
  / Random / Straight Shot / LLM), text input for the model shown when LLM
  is selected, defaulting to `qwen3.8-27b-fp8` (the user's 9arm model), `max_turns` numeric input (default 30 — conservative for live
  API loops; `eval`'s MatchConfig 100 is a batch number), visible not silent.
  Playback bar: Play/Pause toggle, Step (advance exactly one agent turn while
  paused), speed `<select>` (0.5x/1x/2x/4x) scaling the INTER-TURN delay
  only (animation durations stay constant).
- `app.js`: extract the post-response animation from `fire()` into
  `animateShotResponse(data)` (logTurn → dial → shotAnim → explosion →
  hitFlashes → waitExplosion → finishShot) reused by both `fire()` and
  `agentTurn()`. After `finishShot`: if game not over and current side's
  mode ≠ human and playing ⇒ schedule `/api/agent_turn` after
  `BASE_INTER_TURN_DELAY / speed`; if the side is human ⇒ enable input and
  wait. Pause stops auto-advancing after the in-flight animation; Play
  resumes the driver; Step (disabled while animating) requests exactly one
  agent turn while paused. Human-vs-human unchanged end to end.
- Log lines gain the agent name: `Player 1 (Solver): y = ... → HIT`
  (server's `agent` field; keep `textContent`-only — no HTML injection).
  Show `solver_rung` when present (only the solver records one). Draw at
  turn cap → overlay "Draw — turn cap reached".
- `style.css`: minimal styles matching the existing gbutton/panel look.

### `tests/test_ui_server.py` additions

- `team_modes` omitted → defaults human, old flow green (existing tests
  already pin this).
- `{"team_modes": {"team1": "solver", "team2": "random"}}` new_game → 200;
  `/api/agent_turn` plays a deterministic solver turn; response has
  `agent == "solver"`, `outcome` in the ShotOutcome values; log payload
  fields present; turn advanced.
- `/api/agent_turn` while a human side is up → 409 `human_turn`.
- `max_turns: 1` with two agent sides → second turn responds
  `game_over == true, winner == null, draw_reason == "TURN_CAP"`.
- `team_modes` with `llm:<model>` and neither auth env var set (monkeypatch
  delenv both) → new_game 400 with a clear detail (fail-fast constructor
  path).
- Unknown mode string → 400.

---

## Docs + commits

- `PROGRESS_REPORT.txt` + `README.md`: note LLMAgent exists and is playable
  headlessly (roster entry `llm:<model>`) and in the UI (spectator mode);
  keep the notes additive (the report body is otherwise stale from M5.2 —
  do not rewrite history).
- Three commits (one per slice), verify `git status`/`git log` before each
  (parallel sessions have committed here). Suggested subjects:
  1. `M5.4: LLMAgent — Anthropic tool-use with budgeted simulate + safe-dud fallback`
  2. `M5.4: llm:<model> roster registration in the eval runner`
  3. `M5.4: UI team modes + spectator autoplay/step controls`
- Never push; never weaken tests; `# TUNABLE — not from source` on every new
  constant (`_MAX_TOOL_ROUNDS_PER_TURN`, UI inter-turn delay, speed table,
  UI default max_turns=30).

## Risks

- **9arm gateway tool-use support (single open empirical question)**: the
  plan assumes `qwen3.8-27b-fp8` through `gateway.9arm.co` accepts Anthropic
  `tools`/`tool_result` blocks (Claude-Code-compatible gateways generally
  do, but the model's tool-call behavior is unverified). The fake-client
  test suite is unaffected either way. First validation step below is one
  live probe; if the gateway rejects tool use, STOP and return to the user
  with the error — the fallback (a no-tools LLMAgent mode) is a design
  change that needs a decision, not a silent workaround.
- anthropic SDK response surface: only rely on `stop_reason == "tool_use"`,
  `.content` block `.type/.text/.id/.name/.input`; `max_tokens` stop with no
  text = parse-failure path.
- LLM latency inside the server lock: single-user UI, acceptable; document
  in the endpoint docstring.
- llm:X vs llm:X mirror-match stats-key collision: documented limitation.
- Committed eval artifacts stay network-free and untouched (no roster
  changes to seeds.json; no regeneration).

## Validation

1. Gates per slice (order above); full pytest with `--ignore=tests/golden`.
2. Manual UI smoke (no API key needed): `python3 -m uvicorn ui.server:app`,
   solver-vs-random spectator match with autoplay/pause/step/speed; log
   lines show agent names; turn-cap draw fires.
3. FIRST live probe (user-locked model): export
   `ANTHROPIC_BASE_URL=https://gateway.9arm.co` and `ANTHROPIC_AUTH_TOKEN`
   from `~/.claude-9arm.json` (never copy the token into the repo), run the
   UI, set both sides to LLM `qwen3.8-27b-fp8`, play a few turns. This
   answers the open tool-use-support risk before any further LLM work.
