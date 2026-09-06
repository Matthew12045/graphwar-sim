Build an interactive web UI for graphwar-sim, visually and behaviorally faithful
to the original Java Graphwar client. Do this in phases, committing at each
boundary per this project's existing convention (see PROGRESS_REPORT.txt).

═══════════════════════════════════════════════════════════════════
PHASE 0 — Get the real visual ground truth (do not skip, do not guess)
═══════════════════════════════════════════════════════════════════

The existing docs/GROUND_TRUTH.md explicitly skipped all UI/client rendering
files ("skip all UI/server files beyond Constants.java" — IMPLEMENTATION_PLAN.md
§Phase 0). That gap is what this phase fills.

1. If ref/graphwar/ isn't already present locally (it's gitignored, see
   .gitignore's "/ref/" line), clone it:
     git clone https://github.com/catabriga/graphwar.git ref/graphwar

2. Fetch the reference screenshots (separate orphan branch, not on master):
     git clone --branch screenshots https://github.com/catabriga/graphwar.git /tmp/graphwar-screenshots
   Open /tmp/graphwar-screenshots/ss1graphwar.png and ss2Graphwar.png and look
   at them directly. I've already done this — findings below — but verify
   against the images yourself before building.

3. Read these reference source files (I've already read all of them once;
   line numbers below are from that reading — re-verify, don't trust blindly):
     ref/graphwar/src/Graphwar/GameScreen.java       (595 lines — component layout, constructor ~L71-180)
     ref/graphwar/src/Graphwar/GraphPlane.java       (824 lines — terrain/soldier/trajectory drawing; addHelmet ~L760-780; transparentWhite L84)
     ref/graphwar/src/Graphwar/GraphAngleDisplay.java (110 lines — the compass widget, read in full)
     ref/graphwar/src/Graphwar/GraphButton.java       (169 lines)
     ref/graphwar/src/Graphwar/GraphTextBox.java      (129 lines)
     ref/graphwar/src/Graphwar/Soldier.java           (176 lines)
     ref/graphwar/src/GraphServer/Constants.java:25   (BACKGROUND color)
     ref/graphwar/rsc/GameScreen.txt                  (exact widget pixel bounds, read in full)
     ref/graphwar/rsc/explosion.txt                   (explosion animation timing, read in full)
     ref/graphwar/rsc/soldiers/                       (helmet.png + helmetMask.png + soldierNormal.png — composited per-player via color mask, GraphPlane.java addHelmet)

4. Write docs/UI_GROUND_TRUTH.md in the same file:line-citation style as
   docs/GROUND_TRUTH.md. Seed it with what I already extracted so you're not
   starting cold — verify each against the source, then expand:

   - Background: Constants.java:25 `new Color(158,215,155)` = #9ED79B (the
     green frame around the white play field in both screenshots).
   - Play field ("plane"): GameScreen.java:106 `plane.setBounds(15,15,
     Constants.PLANE_LENGTH, Constants.PLANE_HEIGHT)` → (15,15,770,450), white
     background, black filled terrain circles (matches docs/GROUND_TRUTH.md §1.1).
   - Axis lines + edge labels: thin black cross at game-origin, tick labels
     "25"/"-25" at the horizontal edges and "15"/"-15" at the vertical edges —
     these are the game-coordinate bounds already derived in
     docs/GROUND_TRUTH.md §1.2/§1.3 (PLANE_GAME_LENGTH=50 → x∈[-25,25];
     reachable y-band ≈ [-14.63,+14.63], shown rounded to 15).
   - Compass/angle dial: GraphAngleDisplay.java — 60px-radius circle inset at
     (27,27) inside a 200×113 panel (GameScreen.java:112
     `angleDisplay.setBounds(10,475,200,113)`), black crosshair lines at
     (4,57)-(111,57) and (57,5)-(57,111), a filled pie wedge from 0° to the
     current angle colored per-player, angle text in "Sans" PLAIN 14 near the
     bottom-right of the panel. IMPORTANT: per docs/GROUND_TRUTH.md §2.3/§4 of
     the Corrections section, the angle has NO independent effect in
     NORMAL_FUNC mode — it's derived from f's own tangent. This widget is
     cosmetic/informational only; do not let it drive the shot.
   - Function input: GameScreen.txt "FUNCFIELD" at (195,480), width 275; a
     "y =" label image at (125,468) (swapped for "y'=" / "y''=" in the other
     two game modes we don't implement — NORMAL_FUNC only, per this project's
     existing scope).
   - Fire button: GameScreen.txt bounds (115,500), orange-on-tan styling,
     labeled "Fire" (see screenshot).
   - Quit button: GameScreen.txt bounds (290,545).
   - Right-hand panel: GameScreen.txt "TEXTBOXGAME" at (485,480,300,75) plus a
     chat input below it — in the original this is multiplayer chat. This
     project has no networking; repurpose this exact rectangle as a turn log
     instead (see Phase 3 below), not chat.
   - Soldier icon: a round two-tone ("yin-yang"-style) helmet sprite, colored
     per player by compositing a color mask over a shared base sprite
     (GraphPlane.java addHelmet, ~L760-780). Each alive soldier has a
     rounded-rect "Player N" label floating just above it (see both
     screenshots).
   - Trajectory: thin black polyline, no fill.
   - Explosion: rsc/explosion.txt — 6 frames at 15ms each (~90ms total),
     rendered as a yellow/orange/red radial burst, then a 1000ms blank/gap
     before it's gone. Effectively a fast flash, not a lingering effect.
   - Screenshot 2 additionally shows an optional pedagogical overlay: the
     literal typed function ("Actual Function", blue) vs. the auto-offset
     fired curve ("Translated Function", green) — see Phase 5 stretch goal.

   Flag anything you can't pin down exactly as an open question, same as
   docs/OPEN_QUESTIONS.md does — don't invent a pixel value and present it as
   sourced.

═══════════════════════════════════════════════════════════════════
DECISION POINTS — make these explicitly, note your choice in the PR
═══════════════════════════════════════════════════════════════════

A. Assets: ref/graphwar/rsc/*.png are GPLv3. This repo's README already stakes
   out a clean-room position for the ported code. Default: redraw all visuals
   with canvas primitives (arcs, gradients, paths) that match the ORIGINAL'S
   silhouette/color/behavior — no copied image bytes. If you'd rather vendor
   the actual sprites for pixel-perfect fidelity, that's a real option too,
   but it's a licensing call for me, not something to decide silently either
   way — state which you did and why.

B. Stack: FastAPI + a single-page canvas/vanilla-JS frontend, wrapping
   graphwar_sim.state.Game UNCHANGED. Don't touch graphwar_sim/, agents/, or
   eval/ — this is a pure additive UI layer on top of the tested core.

═══════════════════════════════════════════════════════════════════
PHASE 1 — Backend
═══════════════════════════════════════════════════════════════════

Add fastapi + uvicorn under a new `[project.optional-dependencies].ui` group
in pyproject.toml (don't add them to the base `dependencies`).

Create ui/server.py wrapping this existing, already-tested API directly
(graphwar_sim/state.py):
  - Game.create(seed, num_teams=2, num_soldiers=config.INITIAL_NUM_SOLDIERS) -> Game
  - Game.fire(func_str) -> ShotResult          (does NOT advance turn)
  - Game.play_turn(func_str) -> ShotResult     (fires AND advances turn — use this)
  - Game.all_soldiers() -> list[Soldier]
  - Game.finished() -> bool ; Game.winner() -> int | None
  - Game.state: GameState (.teams: list[Team], .current_turn: int)
  - Team.current_soldier(), Team.num_alive()
  - Game.circles: list[(cx, cy, r)]             (terrain, for drawing — don't
    rasterize the whole obstacle grid like render.py does; send circles
    directly, the frontend draws them as filled circles)
  - ShotResult: .points (list[(x,y)] plane px), .hits (list[(player_idx,
    soldier_idx, position_i)]), .last_x, .last_y, .num_steps

Endpoints:
  POST /api/new_game   {seed?: int, num_soldiers?: int} -> full board state JSON
  GET  /api/state       -> current board state JSON
  POST /api/fire         {func_str: str} -> {shot: ShotResult-as-JSON, board,
                          game_over: bool, winner: int|None} or a distinct
                          error shape on MalformedFunction

Error handling: graphwar_sim.parser.PolishNotationFunction (constructed inside
Game.fire) raises MalformedFunction with NO message — per
docs/GROUND_TRUTH.md §6 correction 2, that's faithful to the source, which has
no diagnostic strings at all. Catch it in ui/server.py and return a distinct,
clearly-labeled-as-Python-side error response; don't invent a fake "source"
error string.

Angle display: physics._get_start_angle (physics.py:100) is private and is
exactly what the compass widget needs to show. Either import it directly for
display purposes, or promote it to a public helper (physics.get_start_angle) —
your call, note which in the PR. It must NOT be used to alter the shot itself.

═══════════════════════════════════════════════════════════════════
PHASE 2 — Frontend (ui/static/index.html, app.js, style.css — no build step)
═══════════════════════════════════════════════════════════════════

Single canvas-based page matching docs/UI_GROUND_TRUTH.md:
  - #9ED79B green background/frame; white 770×450 plane at (15,15)
  - Terrain: filled black circles from Game.circles
  - Axis cross + edge tick labels (±25 x, ±15 y)
  - Soldiers: redrawn per-player-colored helmet icon (see Decision A) with a
    floating "Player N" name bubble above each ALIVE one; dead soldiers drop
    off or show a faint marker (your call — original has a ghost sprite,
    rsc/soldierGhost.png, if you want that behavior specifically)
  - Team colors: reuse render.py's existing _team_color() hex values
    (#e11d48 TEAM1 / #2563eb TEAM2) so the web UI and the offline eval plots
    agree visually
  - "y =" label + text input wired to /api/fire, styled like the original's
    orange bold label
  - Orange "Fire" button
  - Compass/angle dial per Phase 0 §UI_GROUND_TRUTH geometry, display-only
  - "New Match" (calls /api/new_game with a fresh seed — expose an optional
    seed field for reproducibility, since Game.create already takes one) and
    "Quit"/reset buttons
  - Right panel (the old chat-box rectangle, repurposed — see Phase 3)

Turn flow — animate, don't snap:
  1. On fire response, interpolate a marker along `shot.points` over ~500ms
  2. Leave the persistent thin black trajectory line drawn
  3. Flash the explosion (redrawn burst, ~90ms per rsc/explosion.txt timing)
     at last_x/last_y and at each hit position in shot.hits
  4. Update alive/dead soldier rendering from the returned board state
  5. If game_over, show a simple win/loss end state (winner from the API);
     otherwise re-enable the input for the new current-turn soldier

Note: the backend's `inverted` handling (physics.py, TEAM2 mirroring) is
already baked into the points ShotResult returns — the frontend never needs
to know about mirroring, just draw the points it's given.

═══════════════════════════════════════════════════════════════════
PHASE 3 — Match log (repurposed chat panel)
═══════════════════════════════════════════════════════════════════

The original's TEXTBOXGAME rectangle (rsc/GameScreen.txt, (485,480,300,75)) is
multiplayer chat — this project has no networking, so repurpose that same
rectangle as a scrolling turn log instead: one line per turn, e.g.
"Team 1 (Soldier 2): y = x/2 + sin(x) → HIT", color-coded by team. Say
explicitly in the PR that this is a repurposing, not a fidelity claim.

═══════════════════════════════════════════════════════════════════
PHASE 4 — Tests
═══════════════════════════════════════════════════════════════════

No pixel-diff test is possible here (no live reference UI to screenshot
against) — acceptance for visual fidelity is manual: run the server, play a
few turns, compare side by side against /tmp/graphwar-screenshots/ss1graphwar.png
and ss2Graphwar.png.

Do add tests/test_ui_server.py using FastAPI's TestClient:
  - new game -> fire a known-safe function -> assert 200 + expected JSON shape
  - fire a malformed function -> assert the distinct error path, not a 500
  - a full game to completion doesn't crash

Then confirm you haven't touched the core:
  python3 -m pytest tests/ -q --ignore=tests/golden      # must stay green

═══════════════════════════════════════════════════════════════════
PHASE 5 — Stretch (only after Phases 1-4 are solid and committed)
═══════════════════════════════════════════════════════════════════

- Toggle overlaying the literal typed function vs. the auto-offset fired
  curve (screenshot 2's "Actual"/"Translated" split) — this project already
  lives in an educational context, this is a genuinely useful teaching toggle,
  not scope creep for its own sake.
- A per-turn "let the solver fire" button wired to agents.solver_agent
  .SolverAgent + agents.observation.observe(game), so a human can play against
  (or watch) the certified/degradation-ladder solver instead of always typing.

═══════════════════════════════════════════════════════════════════
COMMIT
═══════════════════════════════════════════════════════════════════

Update README.md with a "Play it" section (pip install -e ".[ui]"; python3 -m
uvicorn ui.server:app --reload; open localhost). Add a UI milestone row to
PROGRESS_REPORT.txt's milestone table, matching the existing M0-M5.1 format.
Single commit once the smoke tests are green.