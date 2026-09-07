# M5.4 — Persona harness + The Master Magician
# M5.5 — HybridAgent: waypoint schema and prompts

## M5.4.0 Persona inventory — enumerate, do not hardcode

SIX personas, not seven or eight:
  sniper, howitzer, serpent, bodyguard, professor, master_magician

Before migrating: diff all six Shared_core_prepend_to_every_bot_*.txt files
against each other. They should be identical. If they are not, the
divergence is a finding — record it in docs/OPEN_QUESTIONS.md and decide
which text is canonical BEFORE collapsing to agents/personas/_core.md.
Do not silently pick the longest one.

Create `agents/personas/manifest.yaml`:

    - id: master_magician
      file: master_magician.md
      constraint_type: multi_hit
      verifier: verify_multi_hit
      degrades_to: MAGICIAN_PARTIAL
      certifiable: true          # methods 1,2,4
      uncertifiable_methods: [lagrange]

The harness enumerates from the manifest. A test asserts
len(manifest) == len(glob("agents/personas/*.md")) - 1  (excluding _core.md)
so adding a persona file without a manifest entry fails CI. Never write the
persona count as a literal anywhere in the harness.

## M5.4.1 Constraints are VERIFIED, not prompted

Every persona constraint gets a machine-checkable verifier that runs on the
EMITTED EXPRESSION after simulation. The LLM's self-report is not evidence.

| persona        | constraint                        | verifier check                          |
|----------------|-----------------------------------|-----------------------------------------|
| sniper         | simplest rung that works          | ladder rung index is minimal-feasible   |
| howitzer       | peak clears corridor terrain +8   | max f over corridor >= terrain_max + 8  |
| serpent        | zero crossings land in gaps       | sign changes of f fall inside free int. |
| bodyguard      | \|f(d_k) - h_k\| > BLAST + 3      | evaluate per teammate, hard fail        |
| professor      | constraint system stated first    | structured plan present before emit     |
| master_magician| curve hits EVERY living enemy     | see M5.4.2                              |

A violation is logged as CONSTRAINT_VIOLATION with the persona id. It does
NOT crash the round-robin. The leaderboard must separate "lost the round"
from "did not play its own game" — those are different failures and
collapsing them makes the persona comparison meaningless.

## M5.4.2 Master Magician — the enforced constraint

CONSTRAINT: a single emitted expression whose simulated trajectory registers
a hit on every living enemy.

Verification uses `agents/simulate_tool.py` UNMODIFIED. That tool already
declines to apply kills specifically so multi-hit can be probed — this
persona is why it works that way. Do not add kill semantics to it.

    verify_multi_hit(expr, state) ->
      hits = simulate_tool(expr).hits_registered
      living = state.living_enemy_ids
      if hits >= living:      MAGICIAN_FULL
      elif hits:              MAGICIAN_PARTIAL(len(hits), len(living))
      else:                   CONSTRAINT_VIOLATION

MAGICIAN_PARTIAL is a first-class outcome, not a failure. N-hit is often
genuinely infeasible. What IS a violation is silently retreating to a
one-enemy shot and reporting success — the persona must declare the largest
subset it achieved and which enemies it dropped, with the reason.

## M5.4.3 The Comb IS CCF — wire it, do not reimplement

Persona method (2) places one Gaussian per enemy with b = 9/s_min^2.
CCF's basis is exp(-(u-c)^2/(2*sigma^2)). Therefore:

    b = 1/(2*sigma^2)   =>   sigma = s_min / sqrt(18) ~ 0.236 * s_min

The persona's tuned constant is a sigma-ladder entry. Implement the Comb as
CCF with:
- centers c_j PINNED to enemy x-positions (shooter-relative), not a grid
- sigma seeded at s_min/sqrt(18), then walked down the existing ladder
- one near-equality corridor tightening per enemy: |f(x_i) - h_i| <= r_kill
  (a TIGHTENING, never an equality — see M5.5.3 for why)

Put the derivation in the docstring, including the leakage argument:
exp(-b*s_min^2) = e^-9 ~ 1.2e-4, i.e. each tooth contributes ~0.01% of its
height at the adjacent enemy, which is why heights can be set directly
instead of solving a dense system.

### Tooth merging — REQUIRED, not an optimization

The certified margin for the Comb is:

    M*du^2/8 = (9/4) * du^2 * sum_i|h_i| / s_min^2

Quadratic in 1/s_min. Clustered enemies destroy the certificate exactly
where the persona looks most impressive, and simultaneously blow the
emission budget with one tooth per enemy.

Before solving: cluster enemies whose separation is small enough that ONE
Gaussian covers both within kill radius. One tooth per cluster. This raises
s_min AND cuts term count — tighter bound and shorter string from the same
change. Log teeth_before and teeth_after every invocation.

Cap teeth at J_max from M5.2 section 2. If clusters still exceed J_max,
return EMIT_OVERFLOW and degrade to MAGICIAN_PARTIAL over the largest
emittable subset. Choose the subset by hit value, not by x-order.

### Method (3), the Lagrange interpolant, is UNCERTIFIABLE

A polynomial interpolant's second derivative is NOT bounded by
sigma^-2 * ||w||_1 — the CCF certificate does not apply. The persona's own
Runge warning is describing precisely the failure the cell-wise corridor
exists to catch: oscillation BETWEEN nodes.

Rules:
- Lagrange output is always UNCERTIFIED. Never CERTIFIED, at any degree.
- It must still pass the cell-wise corridor check numerically before emit.
  Bounded empirically, not proven — label it that way in the log.
- It is a last resort after methods (1), (2), (4). Log when it fires. If it
  fires often, the Comb's clustering step is misconfigured.

Methods (1) lineup and (4) staircase: (1) is CCF with a near-degenerate
corridor and is certifiable. (4) uses sigmoids, which are NOT in the
Gaussian basis — either express the ladder as a Gaussian sum (certifiable)
or mark it uncertifiable like Lagrange. Check the parser whitelist for
sigmoid support before assuming it can be emitted at all.

## M5.5 HybridAgent

### M5.5.1 Division of labour — non-negotiable

The LLM NEVER emits an expression. It emits INTENT. CCF compiles intent into
a certified string, and that string — not the LLM's text — is what reaches
the parser.

This makes the Shared_core rule "emit ONLY the bare expression" structurally
true instead of prompt-enforced. The LLM cannot violate it, because its
output is JSON that never touches the parser. That is the whole reason
HybridAgent exists.

### M5.5.2 System prompt

```text
You plan artillery shots. You do NOT write mathematical expressions.

A certified solver turns your plan into a curve. It proves the curve clears
terrain and teammates, or it tells you exactly why it cannot. Your job is to
choose targets and suggest a route shape. The solver's job is the math.

Coordinates are SHOOTER-RELATIVE: your soldier is at (0, 0). Positive u is
toward the target. All values you emit use this frame.

You will receive:
- enemy and teammate positions, terrain polyline, map bounds
- a corridor summary: for each sample, the free vertical interval(s)
- the outcome and binding constraint of your previous attempt, if any

Return ONLY a JSON object matching the schema. No prose, no markdown fences,
no explanation outside the "rationale" field.

WAYPOINTS ARE SUGGESTIONS. The solver treats them as soft corridor
tightenings and will DROP the ones that make the problem infeasible, then
tell you which it dropped. Do not try to force a curve by over-specifying:
three loose waypoints beat eight tight ones. If you do not know where the
curve should go, return an empty waypoint list and let the solver decide.

You have a limited simulate_tool budget per turn. Spend it on genuine
uncertainty, not on confirming a plan the solver already certified.
```

### M5.5.3 Waypoint schema

```json
{
  "target_id": "enemy_3",
  "secondary_targets": ["enemy_1"],
  "branch_hint": "over",
  "waypoints": [
    {"u": 120.0, "y": 45.0, "tol": 18.0, "priority": 2},
    {"u": 260.0, "y": 12.0, "tol": 25.0, "priority": 1}
  ],
  "style": "howitzer",
  "rationale": "clear the ridge at u=120, drop behind cover before target"
}
```

| field             | type      | rule                                                    |
|-------------------|-----------|---------------------------------------------------------|
| target_id         | string    | required; must be a LIVING enemy id                      |
| secondary_targets | string[]  | optional; multi-hit intent, drives Comb centers          |
| branch_hint       | enum      | "over" \| "under" \| "any"; seeds M5.2 §8 branch search   |
| waypoints         | object[]  | 0 to 5. Strictly increasing u. Empty list is VALID.      |
| u                 | float     | in (0, u_T), exclusive. Endpoints are solver-owned.      |
| y                 | float     | within map bounds, shooter-relative                      |
| tol               | float     | >= SOLDIER_RADIUS from config.py. Clamped up if smaller. |
| priority          | int       | 1 = drop first when relaxing. Higher survives longer.    |
| style             | enum      | persona id or null; selects shape prior, not constraints |
| rationale         | string    | <= 200 chars, logged only, never parsed                  |

Validation rejects and returns a schema error to the LLM WITHOUT burning a
solver attempt: non-monotone u, u outside (0, u_T), dead target_id, more
than 5 waypoints, unknown style. Schema errors are a separate counter from
solver failures — do not mix them in the eval.

### M5.5.4 Waypoints become TIGHTENINGS, never equalities

For waypoint (u_w, y_w, tol), find the cell k containing u_w and intersect:

    L_k <- max(L_k, y_w - tol)
    H_k <- min(H_k, y_w + tol)

Then propagate to the adjacent sample per the M5.2 §5 cell-wise rule so the
tightening is certified across the cell, not just at a point.

Why never an equality:
- An equality can make a feasible problem infeasible. A tightening degrades
  to a wider interval; an equality has no degradation path.
- Tightenings keep the LP in the same form. Equalities at arbitrary u would
  need extra rows and could conflict with the endpoint constraints.
- The margin term (du^2/(8*sigma^2))*s still applies inside a tightened
  interval, so the certificate composes. An equality would have to be
  exempted from the margin, which silently voids the proof at that point.

If a tightening empties an interval (y_w - tol > H_k), that waypoint is
IMMEDIATELY infeasible. Drop it before solving and report it — do not spend
an LP on a contradiction.

### M5.5.5 Relaxation ladder

If the LP is infeasible with all waypoints applied:
1. Drop the lowest-priority waypoint. Ties broken by most-binding first —
   use the LP dual, not the tolerance value.
2. Re-solve. Repeat until feasible or no waypoints remain.
3. If infeasible with ZERO waypoints, the LLM's plan was never the problem.
   Report BASIS_INFEASIBLE or UNREACHABLE and say so explicitly, so the
   model does not waste a turn rewriting waypoints that were not at fault.

Always report which waypoints were dropped and why. That is the model's
primary learning signal across turns.

### M5.5.6 Feedback message

After every attempt, return this to the LLM verbatim:

```text
ATTEMPT 2 — UNCERTIFIED (slack 3.20)

  binding: lower bound at u=184.0, sigma=12.0, branch "over"
  dropped waypoints: [1] (u=120.0, priority 1) — emptied interval at cell 7
  applied waypoints: [2]
  emitted length: 412 / 500   teeth: 4 (merged from 6)
  simulate_tool calls remaining: 2

The curve exists but clearance is NOT proven. Raise the waypoint at u=120
or widen its tol. Terrain peaks at y=52 near u=184.
```

Rules for this message:
- Always name a SPECIFIC u, sigma, and bound. "Infeasible" alone is useless
  to the model and produces flailing.
- Never return the emitted expression to the LLM. It cannot improve it and
  will try to edit it by hand, which defeats the entire architecture.
- State the remaining tool budget every turn or the model will overspend.

### M5.5.7 Persona overlay

`style` prepends a persona's shape prior WITHOUT changing the schema. The
persona sets defaults; it never gets its own output path.

| style           | overlay effect                                            |
|-----------------|-----------------------------------------------------------|
| howitzer        | seeds a high waypoint over corridor max; branch_hint=over |
| serpent         | seeds waypoints at gap centers; prefers small sigma       |
| bodyguard       | teammate bands pre-applied as tightenings before LLM sees |
| sniper          | waypoints discouraged; empty list preferred               |
| professor       | must emit rationale naming the binding constraint         |
| master_magician | secondary_targets populated with ALL living enemies;      |
|                 | Comb centers pinned per M5.4.3 with tooth merging         |

Bodyguard's bands are applied BEFORE the corridor is shown to the model, so
friendly-fire avoidance is enforced by construction rather than by the LLM
remembering. Every persona inherits that — it is not bodyguard-specific,
bodyguard just makes it the headline.

## M5.5.8 Tests

- Schema: reject non-monotone u, dead target, >5 waypoints, tol below
  SOLDIER_RADIUS (assert clamped, not rejected).
- Tightening: a waypoint narrows L_k/H_k and the certificate still holds at
  10x sample density.
- Equality-free: grep that no waypoint ever produces an equality row in the
  LP. This is the architectural invariant — test it directly.
- Relaxation: construct a case where 3 waypoints are infeasible and 2 are
  feasible; assert the correct one is dropped and reported.
- Zero-waypoint: empty list must solve exactly as bare CCF. Byte-identical
  output.
- Magician: synthetic map with 6 enemies, 2 pairs within merge distance;
  assert teeth == 4 and s_min increased.
- Magician: clustered-enemy map where the unmerged Comb fails the margin;
  assert merging recovers CERTIFIED.
- Lagrange: assert it can never return CERTIFIED, at any degree.
- Isolation: assert the LLM's raw output never reaches parser.py. Mock a
  model that returns a valid expression string in "rationale" and assert it
  is never emitted.

## M5.5.9 Kill criteria

Cut HybridAgent if, on the M4 seeds, it does not beat bare CCF with
waypoints disabled. The LLM is expensive; if its waypoints are dropped by
the relaxation ladder more often than they are applied, it is not planning,
it is guessing, and the solver alone is the better agent. Report the
applied-vs-dropped ratio per persona in eval/REPORT.md — that single number
decides whether this component lives.