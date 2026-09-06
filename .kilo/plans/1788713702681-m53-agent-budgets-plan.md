# M5.3 — Agent budgets: AST depth cap + simulate_tool budget + accounting seams

Source of truth: `5.2.md` appendix (M5.3 patch, line 260) + `5.2.md` §2 (AST depth
cap) + `docs/OPEN_QUESTIONS.md` ("Genuinely open" token-cost item) + handoff
`/var/folders/f0/j5cm67kx7jx6tt2yd7zs8_fw0000gn/T/kilo/graphwar_m53_handoff.md`.
Branch `main`, HEAD `6cfb4ad` (local-only; push at close — decided).

## Decisions (settled with Matt — do not reopen)

1. **Sketch files** (`1._Ladder_dispatch*`, `2._Property-test_scaffolding*`):
   adopt only the `cost` accounting hook (satisfied by the documented accounting
   table, no new `CertifiedFit.cost` field); record the rest as review notes.
   NO ladder-dispatch refactor — it conflicts with the shipped, measured M5.2
   `_ordered_candidates` design. The four files get **committed** with the docs
   commit. P2 (widening monotone) / P4 (degenerate reject) property tests are
   recorded as M5.4/M5.5 candidates, not implemented now.
2. **AST depth cap**: parser-level guard raising `MalformedFunction`;
   `config.MAX_AST_DEPTH = 64` `# TUNABLE — not from source`, next to
   `MAX_EXPR_CHARS`. NOT harness-boundary guards.
3. **Simulate wrapper**: `agents/simulate_budget.py`, `BudgetedSimulator`
   delegating to the **unmodified** `agents/simulate_tool.simulate`; per-turn
   counter + ledger; over-budget call **raises** `SimulateBudgetExhausted`;
   default budget **3** `# TUNABLE`. Count-based only — no wall-clock budget
   anywhere in M5.3 (determinism rule from the handoff).
4. **Accounting**: method defined in docs; seams wired
   (`AgentStats`/`AgentMatchStats` counters, defaults 0); ablation grid
   (budget N ∈ {0,3,10}, ASCII on/off) recorded as the planned M5.4/M5.5
   protocol, NOT run (no simulate-consuming agent exists — M3 skipped
   LLMAgent/HybridAgent).
5. **Leaderboard markdown**: unchanged in M5.3 (byte-parity trap —
   `tests/test_eval.py` asserts byte-for-byte reproduction; columns deferred to
   M5.4 when nonzero data exists).
6. **Push** `6cfb4ad` + M5.3 commits at close.
7. **Out of scope**: M5.4 personas migration, M5.5 HybridAgent (5.2.md appendix).

## Why the depth cap is real (verified against the tree)

A left-linear chain `1+1+1+…` (~1000 terms) fits inside `MAX_EXPR_CHARS=2000`
but RecursionErrors in `_reorder_rec` (parser.py:291) during parse;
`Game.fire` (state.py:242) only guards `MalformedFunction` → match crash
(runner.py:220-228 would not catch it). The reference has no depth limit either
(its `evaluateRec` would `StackOverflowError`) — same family as the char limit,
OPEN_QUESTIONS (e). Cap 64: real CCF emissions peak at depth ~9
(`ceil(log2 J)+3`, J_max ≈ 50 under the char limit); 64 leaves ~7× solver
headroom and kills hostile chains ~15× before Python's 1000-frame recursion
limit. `ccf._emit` (ccf.py:624) already emits balanced trees via
`emission.balanced_sum` — **verify only, do not change**.

## Slices (in order; commit per slice, `M5.3:` prefix, never mix files)

### Slice 1 — AST depth cap

1. `graphwar_sim/config.py`: add `MAX_AST_DEPTH: int = 64` with the doc-comment
   above (harness cap, not from source; cites the RecursionError-within-char-
   limit collision and Python-recursion-reality justification).
2. `graphwar_sim/parser.py`:
   - `_reorder_rec` gains a depth parameter; raise `MalformedFunction()` when
     the recursion depth exceeds the cap (protects the parse step itself —
     without it a 1000-term chain dies at Python's stack before any later
     check). All behavior for depth ≤ cap identical; existing parser tests
     must stay green unchanged.
   - `PolishNotationFunction.__init__` computes the prefix token list's
     **evaluation-tree depth iteratively** (explicit stack; leaves depth 1;
     binary op = max(top2)+1; unary = top+1; reuse `config.get_num_param`) and
     raises `MalformedFunction()` if > cap. This is the guard every consumer
     inherits: `Game.fire`, `agents.simulate_tool`, `solver._verify`,
     `ccf._fired_curve`, and the UI server.
   - Docstrings: both checks are documented harness guards on the faithful
     port (like `MAX_EXPR_CHARS`), not reference behavior.
3. `graphwar_sim/ccf.py`: `emission_budget` gains a depth dimension —
   derive `j_max_depth` from `MAX_AST_DEPTH` (balanced tree: depth =
   `ceil(log2 J) + 3`) and take the min with the char-derived `j_max`. The
   binding constraint at current values is the char limit; the test must read
   BOTH config constants so it breaks if either changes (5.2.md §2 step 2).
   No other CCF changes.
4. `tests/test_parser.py` (+ extend `tests/test_emission.py`):
   - Boundary: depth exactly `MAX_AST_DEPTH` parses; one deeper rejects.
   - A left-linear chain within the 2000-char budget is rejected as
     `MalformedFunction` (today it would RecursionError).
   - `Game.fire` on a cap-exceeding expression raises `MalformedFunction`; the
     runner path classifies `PARSE_ERROR` and fires the safe dud without
     crashing (extend `tests/test_eval.py`).
   - `emission_budget` respects both limits; char binds; test reads both from
     config.
   - Solver/CCF regression: every expression `solve()` emits on a battery of
     seeds parses AND has depth ≤ cap with margin (assert ≤ 16 for CCF output,
     ≤ `MAX_AST_DEPTH` overall).
5. `docs/OPEN_QUESTIONS.md`: new entry (k) — the depth cap: reference has
   none (`evaluateRec` recurses unbounded → `StackOverflowError`); harness cap
   like (e); the collision worked example; consequence for LLM/hostile input
   (deep input → `MalformedFunction` → `PARSE_ERROR`, never a crash).

### Slice 2 — simulate budget wrapper

1. New `agents/simulate_budget.py` (imports `simulate` from the untouched
   `agents/simulate_tool.py` — that file must stay byte-identical; M5.4.2
   depends on it being the pure oracle):
   - `DEFAULT_SIMULATE_BUDGET = 3` `# TUNABLE` (middle of the planned
     ablation grid {0, 3, 10}).
   - `SimulateBudgetExhausted(Exception)`.
   - `BudgetedSimulator(game, budget=DEFAULT_SIMULATE_BUDGET)`:
     `simulate(expr) -> SimResult` (budget check FIRST — a denied call is
     recorded in the ledger as `denied` and raises WITHOUT delegating;
     delegated calls count regardless of parseability), `new_turn()` (resets
     the per-turn counter, appends the finished turn to `turn_log`),
     `remaining`, `calls_used`, `turn_log: list[dict]` (turn, calls, denied)
     — the per-turn metadata pattern of `RecordingSolverAgent`
     (`eval/run_ccf_battery.py:52`).
2. `agents/__init__.py`: export the new names; docstring updated.
3. `tests/test_simulate_budget.py`:
   - Purity: `SimResult` fields identical to a direct `simulate` call for the
     same `(game, expr)`.
   - Counting: parseable and unparseable expressions both burn budget.
   - `new_turn()` reset + ledger accumulation across turns.
   - Exhaustion raises and records `denied`; `remaining` never negative.
   - `budget=0` → first call denied.
   - Determinism: same game + same expression sequence → identical ledger.
   - Grep: no `eval`/`exec`/dynamic import in the new module (repo
     convention, cf. 5.2.md §11 grep tests).

### Slice 3 — accounting seams + docs

1. `agents/base.py` `AgentStats`: add `simulate_calls: int = 0`,
   `simulate_denied: int = 0` (docstring: filled by wrapper-using agents;
   zero for the M3 agents).
2. `eval/metrics.py` `AgentMatchStats`: same two fields; `eval/runner.py`
   merge loop (`play_match`, the `internal:` merge block) adds the two
   counters — two lines, nothing else changes there.
3. **Leaderboard markdown unchanged** (decision 5; byte-parity stays green
   with zero artifact regeneration).
4. `docs/OPEN_QUESTIONS.md`: resolve the "Genuinely open — token-cost /
   ablation metrics (M4)" item into the accounting definition:
   - Units: LLM tokens = prompt+completion counted at the model client
     boundary *when it exists* (M5.4/M5.5); simulate calls = 1 per delegated
     call (denied calls counted separately); CCF cost = existing §12
     instrumentation (`solve_seconds`, `milp_fired`).
   - Granularity: per turn (the wrapper ledger); aggregation: distributions,
     not means (5.2.md §12 rule).
   - Ablation grid recorded as the planned protocol: simulate budget
     N ∈ {0, 3, 10}, ASCII board on/off — runs in M5.4/M5.5, not M5.3.
5. `eval/REPORT.md`: M5.3 section — what shipped (cap, wrapper, seams), the
   determinism note (count-based budget → outcome-neutral; full suite green;
   M5.2 battery numbers unchanged), ablations deliberately not run.
6. Review-notes entry (OPEN_QUESTIONS or REPORT): the four sketch files
   recorded — ladder-dispatch refactor rejected (conflicts with shipped M5.2
   design + §13 SHIP decision); P1/P3 already covered
   (`test_ccf_property.py`, the seeds-1–24 sequential-vs-parallel check); P2
   widening-monotonicity and P4 degenerate-reject noted as M5.4/M5.5 property
   candidates; `cost` hook satisfied by the accounting table.

### Slice 4 — close

1. Commit the four sketch files (untracked at repo root) with the docs
   commit.
2. Full gate, in this order:
   - `python3 -m pip show graphwar-sim` → Editable project location MUST be
     `/Users/matthurindo/graph_war` (AGENTS.md rule; fix with
     `python3 -m pip install -e . --no-deps` from the repo root if not).
   - `mypy --strict graphwar_sim/ agents/ eval/`
   - `ruff check graphwar_sim/ agents/ eval/ && ruff format --check
     graphwar_sim/ agents/ eval/`
   - `python3 -m pytest -q --tb=line -p no:warnings` — **exit code is the
     authority**; ~8 min.
3. Confirm M5.2 numbers unchanged: re-run the per-seed first-shot battery on
   a few seeds into a scratch dir
   (`python3 -m eval --from-seeds eval/results/seeds.json --out
   /var/folders/f0/j5cm67kx7jx6tt2yd7zs8_fw0000gn/T/kilo/m53-check`) —
   `--out` scratch keeps committed artifacts intact; assert the rung
   histogram matches the M5.2 report (the cap must be outcome-neutral).
4. Re-check `git status` / `git log` before each commit (parallel sessions
   have committed under this repo before). Then `git push` (includes
   `6cfb4ad`).

## Execution notes / gotchas (each cost real time in prior sessions)

- Shell has no `rg`; homebrew Python 3.14; 10 cores; zsh.
- macOS spawn multiprocessing dies on stdin heredocs (`python3 - <<'PY'`):
  any parallel experiment goes into a real script file first. M5.3 has no
  parallel work, but the property test re-run does (bit-parallel, ~54 s).
- Import context decides which code runs: `python3 -m` / stdin resolve CWD;
  plain script files resolve the script dir; pytest puts rootdir first. If two
  entry points disagree, print `module.__file__` in both before theorizing.
- Never weaken tests or tune constants to hide symptoms; `# TUNABLE` marks
  every new constant.
- Deep-input behavior change is intended and documented: today deep-but-
  parseable input makes `simulate` return `parseable=False,
  error="RecursionError"`; after Slice 1 the parser rejects it as
  `MalformedFunction` before evaluation — update any test asserting the old
  error string (check `tests/test_agents.py`, `tests/test_eval.py`).
- The prefix-tree depth computation must handle the ADD-zero-fill token
  (`PolishNotationFunction` inserts a 0 VALUE for `+x`) and unary SUBTRACT —
  both come free if the stack walk uses `config.get_num_param`.

## Validation summary

- Per slice: mypy --strict + ruff (check + format) on touched packages +
  targeted pytest.
- Close: full gate (above), byte-parity leaderboard test green WITHOUT
  regeneration, M5.2 battery histogram reproduced unchanged, push.
