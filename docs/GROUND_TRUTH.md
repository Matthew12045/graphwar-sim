# Ground Truth — Graphwar (NORMAL_FUNC mode)

This document is the **authoritative spec** for the Python reimplementation. Every
claim carries a `file:line` citation into the reference Java source under
`ref/graphwar/src/`. Where the plan's earlier assumptions were wrong, the
**Corrections** section at the bottom records the fix.

Scope: **NORMAL_FUNC only** (the `y = f(x)` artillery mode). The two ODE modes
(`FST_ODE`, `SND_ODE`) are out of scope — see `IMPLEMENTATION_PLAN.md`.

All paths are relative to the repo root. The Java reference lives in
`ref/graphwar/src/`.

---

## 1. Coordinate frames

There are **two** coordinate spaces and the simulation converts between them.

### 1.1 Plane / pixel space (the "board")

- Integer pixel grid, origin top-left, `x` right, `y` **down**.
- Width `PLANE_LENGTH = 770`, height `PLANE_HEIGHT = 450`.
  (`Constants.java:62-63`)
- The terrain bitmap is exactly this size:
  `new BufferedImage(PLANE_LENGTH, PLANE_HEIGHT, TYPE_3BYTE_BGR)`
  (`Obstacle.java:38`).
- Terrain is drawn white, then black circles filled on top
  (`Obstacle.java:44-56`). White = empty, non-white = obstacle.

### 1.2 Game / math space (where `f(x)` lives)

- A **centered** world frame. `x = 0` is the horizontal map center, `y = 0` is the
  vertical map center, and **`y` is positive up**.
- Horizontal span is `PLANE_GAME_LENGTH = 50` game units
  (`Constants.java:64`), so game `x ∈ [-25, 25]`.
- The vertical span reuses the horizontal scale: game `y ∈ [-25, 25]` in the
  transform, but the plane is only 450px tall so the reachable band is
  `y ∈ [-14.63, +14.63]` (see §1.3).

**This is a centered world frame, NOT a shooter-relative frame.** The trajectory
`f(x)` is evaluated in this world frame for the whole map; the shooter's position
enters only through the auto-offset (§2.4) and the mirror (§2.3).

### 1.3 Plane → game transform (forward)

From a plane point `(px, py)` to game `(gx, gy)`:

```
gx = PLANE_GAME_LENGTH * (px - PLANE_LENGTH/2) / PLANE_LENGTH
gy = PLANE_GAME_LENGTH * (-py + PLANE_HEIGHT/2) / PLANE_LENGTH
```

`Function.java:193-194`. Note the **`y` flip** (`-py + H/2`): plane-y-down becomes
game-y-up.

### 1.4 Game → plane transform (inverse)

From a game point `(gx, gy)` to plane `(px, py)`:

```
px = PLANE_LENGTH * gx / PLANE_GAME_LENGTH + PLANE_LENGTH/2
py = -PLANE_LENGTH * gy / PLANE_GAME_LENGTH + PLANE_HEIGHT/2
```

`Function.java:243-244`. (Note the inverse uses `PLANE_LENGTH`, not `PLANE_HEIGHT`,
as the multiplier — the source does this; see §6 note.)

### 1.5 Left-facing mirror (TEAM2)

A TEAM2 shooter is mirrored about the vertical center line. In game space the
start `x` is reflected before the transform:

```
valuesX[0] = PLANE_LENGTH - valuesX[0]   // plane space, pre-transform
```

`Function.java:188-191`. And each integrated point is mirrored back in plane space
after the inverse transform:

```
x = PLANE_LENGTH - x
```

`Function.java:247-250`.

`inverted` is `true` for the current-turn player when that player is TEAM2
(`GameData.java:1023` calls `isFunctionReversed()`; `GameData.java:191`).

---

## 2. Trajectory integration (NORMAL_FUNC)

Entry point: `Function.processFunctionRange(Obstacle, Player[], int numPlayers,
int currentTurn, boolean inverted)` — `Function.java:173`.

### 2.1 Start point

```
valuesX[0] = currentTurnSoldier.getX()      // plane px
valuesY[0] = currentTurnSoldier.getY()      // plane py
if (inverted) valuesX[0] = PLANE_LENGTH - valuesX[0]
valuesX[0] = PLANE_GAME_LENGTH*(valuesX[0]-PLANE_LENGTH/2)/PLANE_LENGTH
valuesY[0] = PLANE_GAME_LENGTH*(-valuesY[0]+PLANE_HEIGHT/2)/PLANE_LENGTH
```

`Function.java:183-194`.

### 2.2 Game-coordinate radius

```
gameCoordinateRadius = (PLANE_GAME_LENGTH * SOLDIER_RADIUS) / PLANE_LENGTH
```

`Function.java:196`. With `SOLDIER_RADIUS = 7` (`Constants.java:71`) this is
`50*7/770 ≈ 0.4545` game units.

### 2.3 Fire angle (derived, not a free input)

`fireAngle = getStartAngle(valuesX[0], gameCoordinateRadius)`
(`Function.java:198`). `getStartAngle` (`Function.java:133-160`) is a fixed-point
iteration that converges the **tangent of the function's own curve** at the start
point:

```
startAngleTangent = (f(x+STEP_SIZE) - f(x)) / STEP_SIZE
angle = atan(startAngleTangent)
# iterate: finalX = x + radius*cos(angle); recompute tangent at finalX; ...
```

It converges when `|newAngle - angle| <= ANGLE_ERROR` (`PI/360`) or after
`MAX_ANGLE_LOOPS` (100) iterations (`Constants.java:92-93`).

**The fire angle has no independent effect in function mode.** It is fully
determined by `f`. The only thing it controls is how far along the initial tangent
the start point is nudged (§2.4). The player-supplied angle UI
(`GameData.getAngle`, `GameData.java:472`, using `ANGLE_ACCELERATION`) is for the
ODE/angle modes, not NORMAL_FUNC.

### 2.4 Auto vertical offset (the load-bearing trick)

The start point is nudged along the initial tangent by one game-coordinate radius,
then the whole curve is shifted vertically so it passes exactly through that
nudged point:

```
if (fireAngle is finite) {
    valuesX[0] += gameCoordinateRadius * cos(fireAngle)
    valuesY[0] += gameCoordinateRadius * sin(fireAngle)
}
offSet = -f(valuesX[0]) + valuesY[0]
```

`Function.java:200-206`.

So the fired trajectory is **`y = f(x) + offSet`**, where
`offSet = y0 - f(x0)`. This forces the curve through the soldier's muzzle
regardless of how large `f(0)` is. **This is why the solver must bake the
auto-offset in** — the player's `f` is not anchored at the origin.

### 2.5 Adaptive step-halving integration

```
stepSize = STEP_SIZE          // 0.01
for i in 1 .. FUNC_MAX_STEPS-1:
    tempStepSize = stepSize
    valuesX[i] = valuesX[i-1] + tempStepSize
    valuesY[i] = f(valuesX[i]) + offSet
    endFunc = false
    while (dx^2 + dy^2 > FUNC_MAX_STEP_DISTANCE_SQUARED):   # dx = X[i]-X[i-1], dy = Y[i]-Y[i-1]
        if (dx > FUNC_MIN_X_STEP_DISTANCE):
            tempStepSize /= 2
            valuesX[i] = valuesX[i-1] + tempStepSize
            valuesY[i] = f(valuesX[i]) + offSet
        else:
            endFunc = true; break
    if endFunc: numSteps = i; break
```

`Function.java:208-241`.

Constants (`Constants.java:85-88`):
- `STEP_SIZE = 0.01`
- `FUNC_MAX_STEP_DISTANCE_SQUARED = 0.001` (≈ step length 0.0316)
- `FUNC_MIN_X_STEP_DISTANCE = 0.00001` (halving floor)
- `FUNC_MAX_STEPS = 20000`

The `while` loop halves the step until the Euclidean step distance² drops to
`0.001` or the x-step reaches the `1e-5` floor. This is what makes steep curves
(`tan`, `x^3`, etc.) integrate accurately. **The Python port must reproduce this
exact loop**, including the fact that `tempStepSize` is reset to `stepSize` at the
start of every outer iteration (the halving does not accumulate across outer
iterations).

### 2.6 Hit test (per integrated point)

After each point, convert to plane coords (§1.4), apply the mirror (§1.5), then
test against every alive soldier of every player **except the shooter's own
current-turn soldier** (`Function.java:252-262`):

```
distX = soldier.x - x
distY = soldier.y - y
distSquared = distX^2 + distY^2
if (distSquared < SOLDIER_RADIUS * SOLDIER_RADIUS):   # strict <
    record hit (dedup via playerAlreadyHit)
```

`Function.java:264-284`.

- **Strict `<`**, not `<=`. `SOLDIER_RADIUS = 7` (`Constants.java:71`).
- **Hit test is in plane/pixel coordinates** — both the trajectory point and the
  soldier positions are in plane space. Units are consistent. (Resolves open
  question (a).)
- **Multi-kill**: the hit loop has **no `break`** — the trajectory continues past
  a hit and can kill additional soldiers on later points. (Resolves open question
  (c).)
- Dedup: `playerAlreadyHit(j, k)` (`Function.java:162-171`) prevents recording the
  same (player, soldier) twice.

### 2.7 Termination

The loop stops (`numSteps = i; break`) on the first of:
1. Step-halving floor reached (`endFunc`) — `Function.java:237-241`.
2. **Terrain collision**: `obstacle.collidePoint((int)x, (int)y)` returns true —
   `Function.java:287-291`.
3. **NaN/Inf y**: `Double.isNaN(y) || Double.isInfinite(y)` — `Function.java:293-297`.
4. `FUNC_MAX_STEPS` exhausted.

`collidePoint` (`Obstacle.java:97-109`) returns **true** (i.e. "hit an obstacle")
when out of bounds OR the pixel is non-white (`getRGB != -1`).

`lastX`/`lastY` are set from the final point (`Function.java:301-302`) and drive
the explosion marker (`GameData.getCurrentFunctionPosition`, `GameData.java:1007`,
radius `EXPLOSION_RADIUS = 12`, `Constants.java:79`).

---

## 3. The parser (`PolishNotationFunction`)

Entry: `new PolishNotationFunction(String)` — `PolishNotationFunction.java:47-65`.
Three stages: `createRegularNotationTokens` → `reorderTokensToPolishNotation` →
`getValuesNeeded()` gate.

### 3.1 Preprocessing (string rewrites)

`createRegularNotationTokens` (`PolishNotationFunction.java:209-217`):

```
funcStr = argStr.toLowerCase()
funcStr = funcStr.replaceAll("-", "+-")     # every '-' becomes '+-'
funcStr = funcStr.replaceAll("exp", "e^")   # 'exp' -> 'e^'
funcStr = funcStr.replaceAll(",", ".")      # ',' -> '.'
```

**`-` → `+-` makes every minus a unary negation.** There is no binary subtract
token; `SUBTRACT` is always unary (see §3.5).

### 3.2 Tokenizer

Regex (`PolishNotationFunction.java:217`):

```
[0-9]*\.?[0-9]+|\(|\)|x|y'|y|\+|\*|/|\^|sqrt|log|abs|sin|sen|cos|tan|tg|-|ln|e|pi
```

Driven by `Matcher.find()` in a loop (`PolishNotationFunction.java:225-313`) —
**`find()` skips non-matching characters** (whitespace, stray symbols). So
`"x + 2"` and `"x+2"` tokenize identically; unknown characters are silently
dropped, not errors.

Token mapping (in order of the `if/else` chain, `PolishNotationFunction.java:228-311`):
- A token that parses as a `double` → `ValueToken(value)`.
- `x` → `VARIABLE1`; `y` → `VARIABLE2`; `y'` → `VARIABLE3`.
- `+` → `ADD`; `-` → `SUBTRACT`; `*` → `MULTIPLY`; `/` → `DIVIDE`; `^` → `POW`.
- `sqrt` → `SQRT`; `log` → `LOG`; `abs` → `ABS`; `sin`|`sen` → `SIN`; `cos` →
  `COS`; `tan`|`tg` → `TAN`; `ln` → `LN`.
- `e` → `ValueToken(Math.E)`; `pi` → `ValueToken(Math.PI)`.
- `(` → `LEFT_BRACKET`; `)` → `RIGHT_BRACKET`.

**Ordering matters**: `Double.parseDouble` is tried first, so a numeric literal is
never mistaken for a keyword. `e` and `pi` are special-cased to constants *after*
the number attempt fails.

### 3.3 Implicit multiplication

`adjustImplicitMultiplications` (`PolishNotationFunction.java:162-192`) inserts a
`MULTIPLY` token between adjacent tokens where `isImplicit(last, next)` is true
(`PolishNotationFunction.java:194-207`):

- `last` is one of: `VALUE`, `VARIABLE1`, `VARIABLE2`, `VARIABLE3`, `RIGHT_BRACKET`.
- `next` is one of: `VALUE`, `VARIABLE1`, `VARIABLE2`, `VARIABLE3`, `LEFT_BRACKET`,
  or a **1-parameter operator** (`getNumParam(type2) == 1`).

So `2x` → `2*x`, `x(x+1)` → `x*(x+1)`, `2sin(x)` → `2*sin(x)`, `x sin(x)` →
`x*sin(x)`. But `x+` is NOT implicit (`+` is 2-param), and `x-` is not.

### 3.4 Reordering to Polish — **recursive lowest-nest-operator, NOT shunting-yard**

`reorderRec(polishTokens, funcTokens, start, end)` — `PolishNotationFunction.java:78-149`.

This is the **correction** to the plan's "shunting-yard" assumption. The algorithm:

1. Scan `[start, end]`, tracking bracket `nest` depth. Find the operator
   `next` with the **lowest nest**; on a tie, the one that **precedes** the
   current best. `precedes(t0, t1) = (t0 < t1)` — i.e. **lower token-type number
   wins** (`PolishNotationFunction.java:151-159`).
2. Dispatch on `getNumParam(next)`:
   - **0 params** (a value/var): append it.
   - **1 param** (unary): append the operator, then recurse on `[next+1, end]`.
   - **2 params** (binary): append the operator, recurse on the **left**
     `[start, next-1]`, then the **right** `[next+1, end]`.
     - Special case: if the operator is `ADD` and the left side is empty
       (`leftExists == false`), a `ValueToken(0)` is inserted as the left operand
       (`PolishNotationFunction.java:131-138`). This is how leading `+` / the
       `+-` rewrite yields a well-formed tree.

Because the operator is emitted **before** its operands and the recursion is
postfix-order, the output is **Reverse Polish / postfix**, not prefix. (The class
name "PolishNotation" is a misnomer in the source; the emitted form is postfix —
`makeString` at `PolishNotationFunction.java:503-547` re-parenthesizes it as
`(left op right)`.)

**Token-type numbers drive precedence** (`FunctionToken.java:22-39`):
`ADD=1, SUBTRACT=2, MULTIPLY=3, DIVIDE=4, POW=5, SQRT=6, LOG=7, ABS=8, SIN=9,
COS=10, TAN=11, LN=12, VARIABLE1=13, VARIABLE2=14, VARIABLE3=15, VALUE=16,
LEFT_BRACKET=17, RIGHT_BRACKET=18`.

So at the same nest level, `ADD` (1) is pulled out before `MULTIPLY` (3) before
`POW` (5) — giving the standard `+ < * < ^` precedence. `SUBTRACT` (2) sits just
above `ADD`.

### 3.5 Evaluation

`evaluateFunction(var1, var2, var3)` → `evaluateRec()` —
`PolishNotationFunction.java:968-1127`. Postfix recursive descent.

- `SUBTRACT` is **unary**: `returnValue = -evaluateRec()`
  (`PolishNotationFunction.java:1081-1082`). `getNumParam(SUBTRACT) == 1`
  (`PolishNotationFunction.java:732-734`).
- `LOG` → `Math.log10` (base 10) — `PolishNotationFunction.java:1097-1098`.
- `LN` → `Math.log` (natural) — `PolishNotationFunction.java:1121-1122`.
- `POW` → `Math.pow` — `PolishNotationFunction.java:1117-1118`.
- Division by zero / sqrt of negative / log of non-positive propagate Java
  `NaN`/`Infinity` (no exception), which the integrator then terminates on (§2.7).

### 3.6 Malformed-function gate

`getValuesNeeded()` (`PolishNotationFunction.java:640-664`) walks the **postfix**
token list maintaining a `valuesNeeded` counter (start 1; +`numParam-1` per
operator, −1 per value/var). If it hits 0 **before** the end, returns −1; the
constructor throws `MalformedFunction` unless the final value is exactly 0
(`PolishNotationFunction.java:61-64`).

**`MalformedFunction` is an EMPTY exception class** — no message, no fields
(`MalformedFunction.java`). There are **no error strings** anywhere in the parser.
The Python port should raise a bare `MalformedFunction` (or a message-free
subclass) to stay faithful; any human-readable message is a Python-side addition
and must be clearly marked as such.

---

## 4. Terrain & soldier generation (server-authoritative)

Authoritative generation is in `GraphServer/GraphServer.java` (the server), NOT the
client `Obstacle.getNumCircles` (which is a divergent copy — see §6).

### 4.1 Circles

`generateCircles()` — `GraphServer.java:632-657`:

```
numCircles = (int)(nextGaussian()*NUM_CIRCLES_STANDARD_DEVIATION + NUM_CIRCLES_MEAN_VALUE)
if (numCircles < 1) numCircles = 1
for each circle:
    x      = nextInt(PLANE_LENGTH)                       # [0, 770)
    y      = nextInt(PLANE_HEIGHT)                       # [0, 450)
    radius = (int)(nextGaussian()*CIRCLE_STANDARD_DEVIATION + CIRCLE_MEAN_RADIUS)
    while (radius < 0) radius = (int)(nextGaussian()*CIRCLE_STANDARD_DEVIATION + CIRCLE_MEAN_RADIUS)
```

Constants: `CIRCLE_MEAN_RADIUS = 40`, `CIRCLE_STANDARD_DEVIATION = 25`,
`NUM_CIRCLES_MEAN_VALUE = 15`, `NUM_CIRCLES_STANDARD_DEVIATION = 7`
(`Constants.java:66-69`).

### 4.2 Soldiers

`generateSoldier(soldiers, circles, team)` — `GraphServer.java:705-724`:

```
do:
    x = nextInt(PLANE_LENGTH/2 - 2*SOLDIER_RADIUS) + SOLDIER_RADIUS   # TEAM1: [7, 371)
    y = nextInt(PLANE_HEIGHT - 2*SOLDIER_RADIUS) + SOLDIER_RADIUS     # [7, 436)
    if (team == TEAM2) x += PLANE_LENGTH/2                            # TEAM2: [392, 756)
    soldier = Soldier(x, y)
while (!testSoldier(soldier, soldiers, circles))
```

**TEAM1 occupies the left half, TEAM2 the right half** of the plane.

`testSoldier` (`GraphServer.java:678-703`) rejects a candidate if:
- **Chebyshev** distance to any existing soldier is `< 20` in both x and y
  (`GraphServer.java:687`), OR
- Euclidean distance to any circle center is `< circleRadius + SOLDIER_SELECTION_RADIUS`
  (`GraphServer.java:696`). `SOLDIER_SELECTION_RADIUS = 15` (`Constants.java:72`).

`distance` is Euclidean (`GraphServer.java:671-676`).

### 4.3 Team assignment & start player

- `reorderPlayers()` (`GraphServer.java:820`) alternates teams starting from a
  random team, so players alternate TEAM1/TEAM2.
- `startGame()` (`GraphServer.java:889`) picks
  `startPlayer = abs(nextInt() % players.size())`, retrying until the start player
  has ≥1 soldier.
- `MAX_PLAYERS = 10`, `MAX_SOLDIERS_PER_PLAYER = 4`,
  `INITIAL_NUM_SOLDIERS = 2` (`Constants.java:48-51`).

---

## 5. Turn order & win rule

- **Round-robin, skipping the dead.** `nextTurnMessage`
  (`GameData.java:873-875`): `currentTurn = (currentTurn+1) % numPlayers`, then
  advance while the player has no alive soldiers
  (`Player.nextTurn`, `Player.java:172`).
- **Win rule**: the game ends when **either team** has zero alive soldiers.
  `checkGameFinished()` (`GameData.java:512-539`):
  `if (team1Alive == false || team2Alive == false) return true;`
  (Resolves open question (c).)
- A player's turn advances through their alive soldiers via
  `Player.nextTurn` (`Player.java:172-186`); a player with no alive soldiers is
  skipped entirely.

---

## 6. Corrections to the plan's earlier assumptions

These are the fixes the plan's Phase 0 gate requires be reported **before** M1.

1. **Parser is recursive lowest-nest-operator, NOT shunting-yard.**
   `reorderRec` (`PolishNotationFunction.java:78-149`) is a recursive
   lowest-nest-operator scan with tie-break `precedes = (t0 < t1)` on token-type
   number. The plan's "shunting-yard" description is wrong. Precedence comes from
   the token-type integer values, not an explicit precedence table.

2. **`MalformedFunction` has no error strings.** It is an empty exception
   (`MalformedFunction.java`). The parser never produces a diagnostic message.
   The Python port must not invent a rich error taxonomy as if it were in the
   source; any messages are Python-side and must be labeled as such.

3. **Centered-world frame + auto-offset + mirror.** The frame is centered on the
   map (§1.2), the trajectory is `f(x) + offSet` with `offSet = y0 − f(x0)`
   (§2.4), and TEAM2 is mirrored about the vertical center (§1.5). The plan's
   "shooter-relative frame" assumption is wrong.

4. **Fire angle has no independent effect in function mode.** It is derived from
   `f`'s own tangent (§2.3) and only nudges the start point along that tangent.
   The angle UI (`GameData.getAngle`) belongs to the ODE modes.

5. **Hit test is strict `<` in plane/pixel coordinates, with multi-kill.**
   §2.6. Both trajectory point and soldier are in plane space; the comparison is
   `distSquared < SOLDIER_RADIUS^2`; the loop does not break on a hit.

6. **`Obstacle.getNumCircles` diverges from the server.** The client-side
   `Obstacle.getNumCircles` (`Obstacle.java:68-80`) uses a `while (numCircles < 0)`
   retry, whereas the authoritative `GraphServer.generateCircles`
   (`GraphServer.java:634-639`) uses `if (numCircles < 1) numCircles = 1`. The
   **server** version is authoritative for our simulator; the client copy is a
   red herring.

7. **Inverse transform uses `PLANE_LENGTH` for both axes.** `Function.java:243-244`
   multiplies both `gx` and `gy` by `PLANE_LENGTH` (not `PLANE_HEIGHT`) when going
   game→plane. This is what the source does; the Python port must match it exactly
   (it is arguably a latent quirk, but fidelity to source wins — see ground rule
   "golden-test divergence is reported, not tuned away").

8. **`soldierCollides` 5-point check is placement validation, not the hit test.**
   `Obstacle.soldierCollides` (`Obstacle.java:123-161`) samples center + 4
   cardinal points at `radius` and is used for soldier **placement** validation,
   distinct from the trajectory hit test (§2.6). (Resolves open question (b).)

---

## 7. Constants table (all from `Constants.java`)

| Name | Value | Line |
|---|---|---|
| `PLANE_LENGTH` | 770 | 62 |
| `PLANE_HEIGHT` | 450 | 63 |
| `PLANE_GAME_LENGTH` | 50 | 64 |
| `CIRCLE_MEAN_RADIUS` | 40 | 66 |
| `CIRCLE_STANDARD_DEVIATION` | 25 | 67 |
| `NUM_CIRCLES_MEAN_VALUE` | 15 | 68 |
| `NUM_CIRCLES_STANDARD_DEVIATION` | 7 | 69 |
| `SOLDIER_RADIUS` | 7 | 71 |
| `SOLDIER_SELECTION_RADIUS` | 15 | 72 |
| `EXPLOSION_RADIUS` | 12 | 79 |
| `STEP_SIZE` | 0.01 | 88 |
| `FUNC_MAX_STEPS` | 20000 | 85 |
| `FUNC_MAX_STEP_DISTANCE_SQUARED` | 0.001 | 86 |
| `FUNC_MIN_X_STEP_DISTANCE` | 0.00001 | 87 |
| `ANGLE_ERROR` | π/360 | 92 |
| `MAX_ANGLE_LOOPS` | 100 | 93 |
| `TEAM1` / `TEAM2` | 1 / 2 | 55-56 |
| `NORMAL_FUNC` / `FST_ODE` / `SND_ODE` | 0 / 1 / 2 | 58-60 |
| `MAX_PLAYERS` | 10 | 48 |
| `MAX_SOLDIERS_PER_PLAYER` | 4 | 49 |
| `INITIAL_NUM_SOLDIERS` | 2 | 51 |
| `TURN_TIME` | 60000 | 46 |

Token types (`FunctionToken.java:22-39`): `ADD=1, SUBTRACT=2, MULTIPLY=3,
DIVIDE=4, POW=5, SQRT=6, LOG=7, ABS=8, SIN=9, COS=10, TAN=11, LN=12, VARIABLE1=13,
VARIABLE2=14, VARIABLE3=15, VALUE=16, LEFT_BRACKET=17, RIGHT_BRACKET=18`.
