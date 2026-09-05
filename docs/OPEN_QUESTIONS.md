# Open Questions

This file tracks questions that could not be resolved from the reference source
alone. Per the plan's ground rules, divergence is **reported, not tuned away**,
and any assumption we carry forward is recorded here with its status.

**Status as of M0: all three original open questions are RESOLVED** with
`file:line` citations. Details and citations live in `GROUND_TRUTH.md`.

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
