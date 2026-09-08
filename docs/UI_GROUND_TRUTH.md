# UI Ground Truth — the Java game client's look & behavior

Companion to `docs/GROUND_TRUTH.md` (which covers the simulation only — its
Phase 0 explicitly skipped UI/client rendering files). This document is the
authoritative visual/behavioral spec for `ui/`. Every claim carries a
`file:line` citation into the reference under `ref/graphwar/src/` (or an
`rsc/` resource). Where the reference relies on baked-in artwork we say so
and flag the approximation in **Open questions**.

Scope: **NORMAL_FUNC mode, local game** — multiplayer-only widgets (global
room button, chat) are noted but repurposed or omitted.

All pixel coordinates are the reference's 800×600 window space.

---

## 1. Window & layout

- Window is 800×600 (`Constants.java:27-28`), filled with
  `BACKGROUND = new Color(158,215,155)` = `#9ED79B` before anything else
  (`Constants.java:25`, `GameScreen.java:292-296`).
- The frame art `rsc/backGame.png` (800×**481**) is placed at (0,0)
  (`GameScreen.txt:3-5`). It supplies: the green margin artwork, the plane's
  thin black border, and the axis tick labels (see §2.3). The area below
  y=481 is the flat green fill.
- Component bounds, all from `GameScreen.java:106-125` and
  `rsc/GameScreen.txt` (the file is parsed by `GameScreen`'s constructor,
  `GameScreen.java:71-135`; `GraphUtil.makeTextField` adds height
  `FIELDS_HEIGHT = 25`, `Constants.java:30`, `GraphUtil.java:160-170`):

  | element | bounds (x, y, w, h) | source |
  |---|---|---|
  | plane (play field) | (15, 15, 770, 450) | `GameScreen.java:106` |
  | angle display panel | (10, 475, 200, 113) | `GameScreen.java:112` |
  | turn timer component | (221, 564, 57, 22) | `GameScreen.java:109` |
  | `y =` label image | (125, 468), 70×50 | `GameScreen.txt:31-33`, `file` |
  | FUNCFIELD (function input) | (195, 480, 275, 25) | `GameScreen.txt:61-64` |
  | Fire button | (115, 500), image 200×62 | `GameScreen.txt:43-47`, `file` |
  | Quit button | (290, 545), image 200×62 | `GameScreen.txt:49-53`, `file` |
  | global-room button | (290, 500) — hidden in local games | `GameScreen.txt:55-59`, `GameScreen.java:135` |
  | TEXTBOXGAME (chat box) | (485, 480, 300, 75) | `GameScreen.txt:71-75` |
  | TEXTFIELD (chat input) | (485, 565, 300, 25) | `GameScreen.txt:66-69` |
  | two `timer.png` boxes | (135, 548) and (205, 548), image 89×54 | `GameScreen.txt:23-29`, `file` |
  | quit-confirm dialog | panel (184,184); yes (200,240); no (400,240) | `GameScreen.txt:79-93` |

- Fire/Quit buttons are `GraphButton`s: image-swapping JButtons with a
  per-pixel alpha mask for the hit area (`GraphButton.java:75-113`); hover
  swaps in the `*Over.png` variant (`GraphButton.java:77-84`).
- **Note (source quirk):** `GraphUtil.makeButton` sets the button bounds to
  `(x, y, width, width)` — it passes the image *width* for the height
  (`GraphUtil.java:230`), so a 200×62 button gets a 200×200 swing bounds.
  Invisible in practice; not reproduced.

## 2. The plane (`GraphPlane`)

### 2.1 Background / terrain

- The terrain image is a 770×450 `TYPE_3BYTE_BGR` buffer: filled white, then
  black anti-aliased ovals `fillOval(x-r, y-r, 2r, 2r)` per circle
  (`Obstacle.java:38-56`).
- On repaint the plane draws that image, then a thin black axis cross over
  it: horizontal line across the full width at `PLANE_HEIGHT/2 = 225`, and
  vertical down the full height at `PLANE_LENGTH/2 = 385` (`GraphPlane.java:279-282`).
- **Tick labels are not drawn in code** — the "15" / "-15" / "-25" / "25"
  text seen in the screenshots is baked into `backGame.png`, sitting in the
  green margins just outside the plane (verified by pixel scan: the "15"
  glyph spans x≈397-405, y≈5-12 — centered on the plane's horizontal middle,
  x=400; same pattern at the other three edges). Values match the game
  bounds: x ∈ [-25, 25] (`PLANE_GAME_LENGTH = 50`, `Constants.java:64`) and
  the vertical band shown rounded to ±15 (true band ±14.63, see
  `docs/GROUND_TRUTH.md` §1.2-1.3).
- Terrain circles come from the server; the client just draws the raster it
  is given (`GameData` obstacle image). This project's `Game.circles`
  (`graphwar_sim/state.py:184`) is the equivalent input.
- Destructible terrain: every shot ends in a blast that carves an
  `EXPLOSION_RADIUS = 12` px crater (`Obstacle.explodePoint`,
  `Obstacle.java:118-121`; `Game.fire` in `graphwar_sim/state.py`). The board
  JSON carries `carves` alongside `circles` (`ui/server.py::_board_json`); the
  frontend punches them as white-filled circles after the black terrain
  (`ui/static/app.js::draw`). Craters persist for the match; a fresh `Game`
  resets them.
- Think-bubble growth: `#think-bubble` grows with content up to
  `max-height: 300px` (# TUNABLE) instead of scrolling inside 120px, and
  `THINK_LINE_CAP = 14` (# TUNABLE) accumulates lines so the box visibly
  increases as messages arrive (`ui/static/style.css`, `ui/static/app.js`).
  Autoscroll + `positionThinkBubble` clamping are unchanged.

### 2.2 Soldiers

- The alive-soldier sprite is 20×20 (`rsc/soldiers/soldierNormal.png`,
  `file`): a round ball, roughly half yellow `(255,255,0)` / half white with
  a thin dark outline (pixel-verified).
- Per-player tinting: `addHelmet` (`GraphPlane.java:506-526`) builds a 20×20
  image = body + (`helmetMask.png` filled with the player color via
  `AlphaComposite.SRC_IN`, then the semi-transparent shading of
  `helmet.png` on top). I.e. **the helmet region is tinted the player's
  color**; the body stays yellow/white.
- TEAM2 sprites are mirrored horizontally (`GraphPlane.java:641-651`) —
  same rule as the trajectory mirror, XOR-ed with a globally reversed
  terrain.
- The current-turn soldier gets an animated marker image centered on it
  (`drawCurrentPlayerMarker`, `GraphPlane.java:562-597`, images from
  `rsc/currentPlayerMarker.txt`); it is drawn **into the cached background**
  (`GraphPlane.java:238`), so it persists until the next background repaint.
- **Dead soldiers are removed from the plane entirely**: the death
  animation (`rsc/soldierDeath.txt`) plays, then a fade over
  `deathFadeDuration`, then `getSoldierImage` returns `null`
  (`GraphPlane.java:439-504`). There is **no ghost on the game plane** —
  `rsc/soldierGhost.png` (15×15) is used by the pre-game player board only
  (`GraphUtil.java:116-117`).
- Death/explosion timing per soldier: `SOLDIER_MAX_DEATH_TIME = 6000` ms,
  `NAME_FADE_TIME = 1000` ms (`Constants.java:75-76`).

### 2.3 Player names

- Every **alive** soldier has a name label baked into the background layer
  (`drawPlayersNames`, `GraphPlane.java:698-747`; painted at
  `GraphPlane.java:237`).
- `paintPlayerName` (`GraphPlane.java:660-696`): a rounded rect
  (`fillRoundRect(..., 7, 7)`) filled `transparentWhite =
  new Color(255,255,255,170)` (`GraphPlane.java:84`), outlined in the
  player color, with the player's name in black
  `NAME_FONT = new Font("Sans", Font.PLAIN, 14)` (`Constants.java:77`).
- Geometry: 3px border; label width = measured text width + 2×3; height 15;
  positioned above the soldier at `borderY = y - 15 - 2·SOLDIER_RADIUS`
  (i.e. 29px above the center), text baseline at `y - 2 - 2·SOLDIER_RADIUS`;
  **flips below the soldier** if it would clip the top edge
  (`GraphPlane.java:670-674`), and clamps horizontally to the plane
  (`GraphPlane.java:676-686`).
- A dying soldier's label fades out over `NAME_FADE_TIME`
  (`GraphPlane.java:730-744`).
- This project's players are named "Player 1"/"Player 2" (player index + 1)
  to match the screenshots.

### 2.4 Trajectory (function curve)

- The trajectory is drawn **in the current player's color**, as a thin
  (default-width) anti-aliased polyline through the integrated points
  (`drawFunctionImage`, `GraphPlane.java:303-350`, color at :312-313).
  **Correction to the task seed:** it is not black.
- It is drawn **progressively** — the game animates through the integrated
  points at `FUNCTION_VELOCITY = 1500` steps/second (`Constants.java:81`,
  `GameData.getCurrentFunctionPosition`), so the curve traces itself over
  roughly `numSteps/1500` seconds.
- Only the **current** shot exists: each shot starts a fresh transparent
  overlay (`startDrawingFunction`, `GraphPlane.java:285-291`); previous
  shots leave nothing.
- After the explosion starts, the curve **fades out** over
  `FUNC_FADE_TIME = 1000` ms (`GraphPlane.java:361-375`,
  `Constants.java:82`). **Correction to the task seed:** the trajectory is
  not persistent — it disappears after the fade.

### 2.5 Explosion

- `rsc/explosion.txt` lists 7 frames: `explosion0..5.png` at **15 ms** each,
  then `blank.png` at **1000 ms** (read in full; frames are 40×40). The
  visible burst is therefore ~90 ms, then blank for 1 s.
- It is drawn centered on the shot's last point
  (`drawX = lastX - width/2`, `drawY = lastY - height/2`, mirrored for
  reversed teams, `GraphPlane.java:392-404`), stepping through frames by
  elapsed time (`GraphPlane.java:423-435`).
- There is **one explosion, at the last point** — struck soldiers are shown
  by their own death animation, not by extra explosions.
  (**Correction to the task seed**, which flashed at every hit position.)
- Frame art is white-hot center → yellow/orange/red falloff (pixel-sampled);
  this project redraws it as a radial gradient burst (Decision A).

## 3. Angle display (`GraphAngleDisplay`) — display-only

Read in full (110 lines). In NORMAL_FUNC the angle is **derived** from the
fired function (see `docs/GROUND_TRUTH.md` §2.3); this widget never drives
the shot.

- Value source: the current-turn soldier's `angle`. It is set when a
  function is processed: `player.getCurrentTurnSoldier().setAngle(function.getFireAngle())`
  (`GameData.java:1113`). So the dial updates **on fire**, not while typing.
- Panel (10,475,200,113) (`GameScreen.java:112`); the circle-outline artwork
  is `backAngle.png` (139×139) at (-2,463) (`GameScreen.txt:19-21`).
- `paintComponent` (`GraphAngleDisplay.java:42-64`):
  - wedge: `fillArc(27, 27, 60, 60, 0, (int)degrees(angle))` in the player
    color (`GraphAngleDisplay.java:76`) — Java arc angles start at 3 o'clock
    and sweep counter-clockwise, so positive (upward) angles sweep up-left
    from horizontal.
  - **reversed** (functionReversed XOR terrainReversed, i.e. TEAM2 shooters
    in this project): `fillArc(27, 27, 60, 60, 180, -(int)degrees)` — the
    wedge starts at 9 o'clock and sweeps clockwise (`GraphAngleDisplay.java:68-77`).
  - black crosshair: (4,57)-(111,57) and (57,5)-(57,111)
    (`GraphAngleDisplay.java:79-81`) — the lines extend well past the
    30px-radius circle.
  - black "barrel" line from center (57,57) to
    `(57 + 53·cos(angle), 57 - 53·sin(angle))` (y flipped for screen),
    mirrored in the reversed case (`GraphAngleDisplay.java:83-90`).
- Angle text: `Sans PLAIN 14`, black, at (145,108) inside the panel —
  absolute (155,583) — formatted as degrees rounded to 2 decimals + "°"
  (`GraphAngleDisplay.java:93-108`). It lands on top of the left white box
  (§5).

## 4. Turn timer (`GraphTimer`) — omitted from this project

- White 57×22 box at (221,564) (`GameScreen.java:109`), countdown rendered
  as seconds with one decimal (`remaining/100/10`, `GraphTimer.java:62-65`),
  `Sans BOLD 18` black, red when under 5 s (`GraphTimer.java:39,70-75`).
  The "49.3" box in both reference screenshots is this countdown.
- This project has no turn clock (`TURN_TIME` is informational only,
  `graphwar_sim/config.py:81-82`), so the countdown is **omitted**; the
  angle text (§3) keeps the left white box. See Open questions.

## 5. White boxes / leftover art

- `timer.png` (89×54, mostly transparent with a white box) is placed twice,
  at (135,548) and (205,548) (`GameScreen.txt:23-29`). The left box hosts
  the angle text (§3); the right box hosts the countdown (§4). They are
  plain decorations; this project draws the left box as a plain white rect
  behind the angle text and drops the right one with the timer.

## 6. Colors

- Frame green `#9ED79B` (`Constants.java:25`); plane white + black terrain
  (`Obstacle.java:44-55`).
- **Player colors are random** in the reference: `GraphUtil.getRandomColor()`
  draws uniform RGB until `r²+g²+b² ≤ 3·160²` (`GraphUtil.java:41-52`,
  limit `MAXIMUM_COLOR_MODULE_SQUARED`, `Constants.java:53`). One color per
  player, used for the helmet tint, name border, wedge, and trajectory
  (`Player.java:56`).
- **Deliberate divergence:** this project uses the fixed palette from
  `graphwar_sim/render.py::_team_color` — `#e11d48` (TEAM1) /
  `#2563eb` (TEAM2) — so the web UI and the offline eval plots agree, and
  shots are attributable at a glance. Recorded here so nobody mistakes it
  for a fidelity claim.
- Orange accent (the "y =" label and Fire/Quit text) is `#FFAE00` with a
  dark `(63,63,63)` shadow; button faces are pale sage `#CBDACB→#E3EAE2`
  with a dark border (pixel-sampled from `rsc/y.png`, `rsc/fire.png`).
  Approximation of baked-in art — see Open questions.

## 7. Interaction (NORMAL_FUNC, local game)

- Fire: the funcField's action or the Fire button sends the typed string,
  ignored if empty (`GameScreen.java:427-437`).
- Keystrokes in the function field also send a "function preview" to the
  server (`GameScreen.java:550-551`) — a multiplayer courtesy showing others
  the curve being typed. Out of scope here (no networking).
- Quit: opens a confirm dialog (yes/no at (200,240)/(400,240) over
  `backQuitConfirm.png` at (184,184), `GameScreen.txt:79-93`,
  `GameScreen.java:241-261`). This project repurposes it as
  "reset the match".
- Turn advancement, win detection, and message dialogs (`showMessage`) are
  server-driven in the reference; this project's equivalents live in
  `graphwar_sim/state.py` and the UI derives them from the API responses.

## 8. Text box (chat) → turn log repurposing

- The reference's TEXTBOXGAME (485,480,300,75) + TEXTFIELD (485,565,300,25)
  are the multiplayer chat (`GameScreen.java:298-310`,
  `GraphTextBox.java:60-108`: bold colored name + black message, grey
  italic `(160,160,160)` system lines, auto-scroll, always-on vertical
  scrollbar).
- **This project has no networking.** The same rectangles are repurposed as
  a scrolling per-turn match log (one line per turn, team-colored name +
  outcome) and a disabled/hidden chat input. This is a repurposing, not a
  fidelity claim.

---

## Open questions (not inventable from source)

(a) **Exact widget-art colors/gradients.** `fire.png`/`quit.png` faces,
`backAngle.png`'s circle shading, `timer.png`'s box shape, and `backGame.png`'s
label font are baked-in artwork; this project approximates them with
primitives sampled from the PNGs (§6). No source line pins them down.

(b) **Turn countdown omitted.** The right white box + "NN.N" seconds display
(§4) needs a turn clock the simulator does not model. Omitted; the box is
dropped rather than shown empty.

(c) **Death animation.** `rsc/soldierDeath.txt` frames and
`deathFadeDuration` are not reproduced (Decision A redraws everything with
primitives); killed soldiers simply vanish after a short flash. If fidelity
here matters later, the frames' durations are in that resource file.

(d) **Current-turn marker animation** (`rsc/currentPlayerMarker.txt`) is
reproduced as a simple pulsing ring in the shooter's color, not the original
sprite sequence.
