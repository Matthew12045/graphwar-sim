/* Graphwar web UI — canvas rendering + turn flow.
 *
 * Visual/behavioral spec: docs/UI_GROUND_TRUTH.md. All visuals are redrawn
 * with canvas primitives (Decision A) — no GPL image bytes are used.
 *
 * Turn flow (per implement.md Phase 2, with the source-verified corrections
 * from UI_GROUND_TRUTH.md §2.4-2.5):
 *   1. animate the curve along shot.points over ~500ms (the original draws
 *      progressively at 1500 steps/s, Constants.java:81)
 *   2. explosion flash at the shot's last point only (~90ms per
 *      rsc/explosion.txt — 6 frames × 15ms), struck soldiers flash
 *   3. the trajectory fades out over FUNC_FADE_TIME = 1000ms
 *      (Constants.java:82, GraphPlane.java:361-375) — it is NOT persistent
 *   4. board updates (kills, next turn), dial updates (on fire only,
 *      GameData.java:1113), win overlay or input re-enabled
 */

"use strict";

var PLANE_W = 770;
var PLANE_H = 450;
var SOLDIER_R = 7; // Constants.java:71
var FUNC_MAX_STEPS = 20000; // Constants.java:85
var FUNC_FADE_TIME = 1000; // Constants.java:82
var EXPLOSION_TIME = 90; // explosion.txt: 6 frames x 15ms
var FLY_TIME = 500; // implement.md: interpolate the shot over ~500ms

var plane = document.getElementById("plane");
var ctx = plane.getContext("2d");
var compass = document.getElementById("compass");
var cctx = compass.getContext("2d");
var funcInput = document.getElementById("func");
var fireBtn = document.getElementById("fire");
var newBtn = document.getElementById("new-match");
var quitBtn = document.getElementById("quit");
var seedInput = document.getElementById("seed");
var logBox = document.getElementById("log");
var angleBox = document.getElementById("angle-box");
var overlay = document.getElementById("overlay");
var overlayMsg = document.getElementById("overlay-msg");
var overlayButtons = document.getElementById("overlay-buttons");
var modeTeam1Sel = document.getElementById("mode-team1");
var modeTeam2Sel = document.getElementById("mode-team2");
var modelTeam1Input = document.getElementById("model-team1");
var modelTeam2Input = document.getElementById("model-team2");
var maxTurnsInput = document.getElementById("max-turns");
var playPauseBtn = document.getElementById("play-pause");
var stepBtn = document.getElementById("step");
var speedSel = document.getElementById("speed");

var board = null;
var seed = null;
var dialAngle = null; // last fired start angle (radians, game space)
var dialColor = "#000000";
var dialInverted = false;
var shotAnim = null; // {points, color, start}
var explosion = null; // {x, y, start}
var hitFlashes = []; // [{x, y, start}]
var fadingTraj = null; // {points, color, fadeStart}
var animating = false;
var gameOver = false;

// --- M5.4 spectator state ----------------------------------------------------
// # TUNABLE — not from source: the inter-turn pacing + speed table scale ONLY
// the delay BETWEEN turns (animation durations above stay constant); the UI
// default max_turns (index.html) is conservative for live-API agent loops
// (eval's MatchConfig 100 is a batch number).
var BASE_INTER_TURN_DELAY = 700; // ms between agent turns at 1x
var DEFAULT_LLM_MODEL = "qwen3.8-27b-fp8"; // the 9arm gateway model
var teamModes = { team1: "human", team2: "human" };
var playing = false;
var playTimer = null; // setTimeout handle for the next agent turn
var speed = 1; // scales the inter-turn delay only (BASE / speed)

// --- helpers ---------------------------------------------------------------

function el(id) {
  return document.getElementById(id);
}

function soldierPos(playerIndex, soldierIndex) {
  var team = board.teams[playerIndex];
  var s = team.soldiers[soldierIndex];
  return { x: s.x, y: s.y };
}

// --- log (repurposed TEXTBOXGAME; textContent only — no HTML injection) ----

function logLine(spans) {
  var div = document.createElement("div");
  for (var i = 0; i < spans.length; i++) {
    var span = document.createElement("span");
    span.className = spans[i].cls;
    span.style.color = spans[i].color || "";
    span.textContent = spans[i].text;
    div.appendChild(span);
  }
  logBox.appendChild(div);
  logBox.scrollTop = logBox.scrollHeight;
}

function logSystem(text) {
  logLine([{ cls: "system", text: text }]);
}

function outcomeOf(shot) {
  if (shot.hits.length > 0) {
    return shot.hits.length > 1 ? "HIT ×" + shot.hits.length : "HIT";
  }
  if (shot.num_steps >= FUNC_MAX_STEPS) return "MAX STEPS";
  var lx = shot.last_x;
  var ly = shot.last_y;
  if (lx < 0 || lx > PLANE_W || ly < 0 || ly > PLANE_H) return "OFF FIELD";
  return "TERRAIN";
}

function logTurn(shooter, funcStr, shot, agentName, solverRung) {
  // M5.4: agent-driven turns name the driver ("Player 1 (Solver)");
  // human fire keeps the soldier label. textContent-only — no HTML injection.
  var nameText = agentName
    ? shooter.label + " (" + agentName + ")"
    : shooter.label + " (Soldier " + (shooter.soldier_index + 1) + ")";
  var spans = [
    { cls: "name", color: shooter.color, text: nameText + ": " },
    { cls: "", text: funcStr + " → " + outcomeOf(shot) },
  ];
  if (solverRung) {
    spans.push({ cls: "system", text: " [rung: " + solverRung + "]" });
  }
  logLine(spans);
}

// --- overlay (quit confirm / game over) ------------------------------------

function showOverlay(msg, buttons) {
  overlayMsg.textContent = msg;
  overlayButtons.innerHTML = "";
  buttons.forEach(function (b) {
    var btn = document.createElement("button");
    btn.className = "gbutton";
    btn.textContent = b.label;
    btn.addEventListener("click", function () {
      overlay.classList.add("hidden");
      b.action();
    });
    overlayButtons.appendChild(btn);
  });
  overlay.classList.remove("hidden");
}

// --- soldier sprite (Decision A: primitive redraw of the 20×20 ball,
//     GraphPlane.java:506-526 tinting the helmet the player color) ----------

function drawSoldier(cx, cy, color, inverted) {
  ctx.save();
  ctx.translate(cx, cy);
  if (inverted) ctx.scale(-1, 1); // TEAM2 sprites mirror (GraphPlane.java:641-651)

  ctx.beginPath();
  ctx.arc(0, 0, 9, 0, Math.PI * 2);
  ctx.fillStyle = "#ffff00";
  ctx.fill();

  // white right-half highlight (yin-yang style split)
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.arc(0, 0, 9, -Math.PI / 2, Math.PI / 2);
  ctx.closePath();
  ctx.fillStyle = "#ffffff";
  ctx.fill();

  // helmet cap tinted with the player color (addHelmet's SRC_IN fill)
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.arc(0, 0, 9, Math.PI, Math.PI * 2);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();

  ctx.beginPath();
  ctx.arc(0, 0, 9, 0, Math.PI * 2);
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();
}

// --- name labels (GraphPlane.java:660-696) ---------------------------------

function roundedRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

function drawNameLabel(x, y, name, color) {
  var border = 3;
  ctx.font = '14px "Sans", "DejaVu Sans", Verdana, sans-serif';
  var textW = ctx.measureText(name).width;
  var boxW = textW + 2 * border;
  var boxY = y - 15 - 2 * SOLDIER_R;
  var boxX = x - textW / 2 - border;
  var textY = y - 2 - 2 * SOLDIER_R;
  if (boxY < 0) {
    boxY = y - 2 + 2 * SOLDIER_R;
    textY = y + 11 + 2 * SOLDIER_R;
  }
  if (boxX < 0) {
    boxX = 0;
  }
  if (boxX + boxW > PLANE_W) {
    boxX = PLANE_W - boxW;
  }

  ctx.fillStyle = "rgba(255,255,255,0.667)"; // transparentWhite, GraphPlane.java:84
  roundedRect(boxX, boxY, boxW, 15, 7);
  ctx.fill();
  ctx.strokeStyle = color;
  roundedRect(boxX, boxY, boxW, 15, 7);
  ctx.stroke();
  ctx.fillStyle = "#000";
  ctx.fillText(name, boxX + border, textY);
}

// --- explosion (explosion.txt: ~90ms burst, redrawn as radial gradient) ----

function drawExplosion(now) {
  if (!explosion) return;
  var t = now - explosion.start;
  if (t < 0 || t > EXPLOSION_TIME) return;
  var k = t / EXPLOSION_TIME;
  var r = 4 + 16 * k;
  var g = ctx.createRadialGradient(
    explosion.x,
    explosion.y,
    0,
    explosion.x,
    explosion.y,
    r,
  );
  g.addColorStop(0, "rgba(255,255,255," + (1 - k) + ")");
  g.addColorStop(0.4, "rgba(255,174,0," + (1 - k) + ")");
  g.addColorStop(0.75, "rgba(226,73,47," + (0.9 - k) + ")");
  g.addColorStop(1, "rgba(226,73,47,0)");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(explosion.x, explosion.y, r, 0, Math.PI * 2);
  ctx.fill();
}

// --- main plane draw --------------------------------------------------------

function draw(now) {
  ctx.clearRect(0, 0, PLANE_W, PLANE_H);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, PLANE_W, PLANE_H);
  if (!board) return;

  // terrain: filled black circles straight from Game.circles
  ctx.fillStyle = "#000";
  board.circles.forEach(function (c) {
    ctx.beginPath();
    ctx.arc(c[0], c[1], c[2], 0, Math.PI * 2);
    ctx.fill();
  });

  // axis cross (GraphPlane.java:279-282)
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, PLANE_H / 2);
  ctx.lineTo(PLANE_W, PLANE_H / 2);
  ctx.moveTo(PLANE_W / 2, 0);
  ctx.lineTo(PLANE_W / 2, PLANE_H);
  ctx.stroke();

  // trajectories: the in-flight one, then the fading previous one
  function strokePoints(points, count, color, alpha) {
    if (count < 2) return;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.beginPath();
    var pen = false;
    for (var i = 0; i < count; i++) {
      var p = points[i];
      if (p[0] === null || p[1] === null) {
        pen = false; // shot ended on NaN/Inf (server sends null)
        continue;
      }
      if (!pen) {
        ctx.moveTo(p[0], p[1]);
        pen = true;
      } else {
        ctx.lineTo(p[0], p[1]);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  var head = null;
  if (shotAnim) {
    var k = Math.min(1, (now - shotAnim.start) / FLY_TIME);
    var count = Math.max(2, Math.round(k * shotAnim.points.length));
    strokePoints(shotAnim.points, count, shotAnim.color, 1);
    var p = shotAnim.points[count - 1];
    if (p[0] !== null && p[1] !== null) head = { x: p[0], y: p[1] };
  }
  if (fadingTraj) {
    var age = now - fadingTraj.fadeStart;
    if (age >= FUNC_FADE_TIME) {
      fadingTraj = null;
    } else {
      strokePoints(
        fadingTraj.points,
        fadingTraj.points.length,
        fadingTraj.color,
        1 - age / FUNC_FADE_TIME,
      );
    }
  }

  // current-turn marker: pulsing ring in the shooter's color
  // (drawCurrentPlayerMarker; simplified per Open question (d))
  if (board.shooter && !gameOver) {
    var cur = soldierPos(board.shooter.player_index, board.shooter.soldier_index);
    ctx.strokeStyle = board.shooter.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cur.x, cur.y, 11 + 2 * Math.sin(now / 200), 0, Math.PI * 2);
    ctx.stroke();
  }

  // soldiers + name labels (alive only; dead vanish, §2.2)
  board.teams.forEach(function (team) {
    team.soldiers.forEach(function (s) {
      if (!s.alive) return;
      drawSoldier(s.x, s.y, team.color, team.team === 2);
    });
  });
  board.teams.forEach(function (team) {
    team.soldiers.forEach(function (s) {
      if (!s.alive) return;
      drawNameLabel(s.x, s.y, team.label, team.color);
    });
  });

  // hit flashes + explosion (explosion window only)
  hitFlashes.forEach(function (f) {
    var t = now - f.start;
    if (t < 0 || t > EXPLOSION_TIME) return;
    ctx.strokeStyle = "rgba(255,255,255," + (1 - t / EXPLOSION_TIME) + ")";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(f.x, f.y, SOLDIER_R + 3, 0, Math.PI * 2);
    ctx.stroke();
  });
  drawExplosion(now);

  // shot head marker
  if (head) {
    ctx.fillStyle = "#000";
    ctx.beginPath();
    ctx.arc(head.x, head.y, 2, 0, Math.PI * 2);
    ctx.fill();
  }
}

// --- compass (GraphAngleDisplay.java:42-108, display-only) ------------------

function drawCompass() {
  cctx.clearRect(0, 0, 200, 113);
  var cx = 57;
  var cy = 57;
  var wedgeR = 30; // fillArc(27,27,60,60) -> radius 30, GraphAngleDisplay.java:76
  var circleR = 53; // backAngle.png outline, chord-measured off screenshot 1

  if (dialAngle !== null) {
    // Java fillArc: 0° at 3 o'clock, positive counter-clockwise.
    // Non-reversed: wedge from 0 sweeping to -angle (canvas y-down).
    // Reversed (TEAM2): wedge from 180° (GraphAngleDisplay.java:68-77).
    var base = dialInverted ? Math.PI : 0;
    var sweep = dialInverted ? dialAngle : -dialAngle;
    var a0 = Math.min(base, base + sweep);
    var a1 = Math.max(base, base + sweep);
    cctx.fillStyle = dialColor;
    cctx.beginPath();
    cctx.moveTo(cx, cy);
    cctx.arc(cx, cy, wedgeR, a0, a1, false);
    cctx.closePath();
    cctx.fill();
  }

  cctx.strokeStyle = "#000";
  cctx.lineWidth = 1;
  cctx.beginPath();
  cctx.moveTo(4, 57);
  cctx.lineTo(111, 57);
  cctx.moveTo(57, 5);
  cctx.lineTo(57, 111);
  cctx.stroke();

  if (dialAngle !== null) {
    // barrel line, mirrored for reversed (GraphAngleDisplay.java:83-90)
    var sx = dialInverted ? -1 : 1;
    cctx.beginPath();
    cctx.moveTo(cx, cy);
    cctx.lineTo(cx + sx * 53 * Math.cos(dialAngle), cy - 53 * Math.sin(dialAngle));
    cctx.stroke();
  }

  // circle outline (backAngle.png artwork, redrawn; tips touch the crosshair)
  cctx.beginPath();
  cctx.arc(cx, cy, circleR, 0, Math.PI * 2);
  cctx.stroke();

  // angle text (GraphAngleDisplay.java:93-108: degrees, 2 decimals, °)
  if (dialAngle === null) {
    angleBox.textContent = "";
  } else {
    var deg = dialAngle * (180 / Math.PI);
    angleBox.textContent = deg.toFixed(2) + "°";
  }
}

// --- server interaction ------------------------------------------------------

function apiErrorText(status, body) {
  if (body && body.detail) return body.detail;
  return "Request failed (HTTP " + status + ")";
}

function postJSON(path, body) {
  return fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  }).then(function (res) {
    return res.json().then(function (data) {
      return { status: res.status, data: data };
    });
  });
}

function applyBoard(newBoard) {
  board = newBoard;
  gameOver = board.finished;
  if (board.shooter) {
    dialColor = board.shooter.color;
    dialInverted = board.shooter.inverted;
  }
  if (dialAngle === null) dialAngle = 0; // soldier angle starts at 0 (display-only)
  drawCompass();
}

function setInputEnabled(enabled) {
  funcInput.disabled = !enabled;
  fireBtn.disabled = !enabled;
}

// --- M5.4 setup + playback controls -----------------------------------------

function updateModelInputs() {
  modelTeam1Input.classList.toggle("hidden", modeTeam1Sel.value !== "llm");
  modelTeam2Input.classList.toggle("hidden", modeTeam2Sel.value !== "llm");
}

function composedMode(modeSelect, modelInput) {
  // The wire string is the roster entry: "llm:<model>" for LLM sides.
  if (modeSelect.value !== "llm") return modeSelect.value;
  var model = modelInput.value.trim();
  return "llm:" + (model === "" ? DEFAULT_LLM_MODEL : model);
}

function wireToSelect(mode) {
  if (mode === "human" || mode === "solver" || mode === "random" || mode === "straight") {
    return mode;
  }
  return mode.indexOf("llm:") === 0 ? "llm" : "human";
}

function wireToModel(mode) {
  return mode.indexOf("llm:") === 0 ? mode.slice(4) : DEFAULT_LLM_MODEL;
}

function syncSetupFromServer(data) {
  if (data.team_modes) {
    teamModes = {
      team1: data.team_modes.team1 || "human",
      team2: data.team_modes.team2 || "human",
    };
    modeTeam1Sel.value = wireToSelect(teamModes.team1);
    modelTeam1Input.value = wireToModel(teamModes.team1);
    modeTeam2Sel.value = wireToSelect(teamModes.team2);
    modelTeam2Input.value = wireToModel(teamModes.team2);
    updateModelInputs();
  }
  if (typeof data.max_turns === "number") {
    maxTurnsInput.value = data.max_turns;
  }
}

function currentSideMode() {
  if (!board || !board.shooter) return "human";
  return board.shooter.player_index === 0 ? teamModes.team1 : teamModes.team2;
}

// The single source of truth for input/button state: the function input is
// enabled only when the game is live and a HUMAN side is up (human-vs-human
// behavior unchanged); agent sides are server-driven.
function updateControls() {
  var humanUp = !gameOver && !animating && currentSideMode() === "human";
  setInputEnabled(humanUp);
  playPauseBtn.disabled = gameOver;
  stepBtn.disabled = gameOver || playing || animating || currentSideMode() === "human";
}

function scheduleNext() {
  // Autoplay driver: exactly one agent turn per tick; a human side just
  // waits (input re-enabled via updateControls).
  if (gameOver || !playing || currentSideMode() === "human") return;
  playTimer = setTimeout(function () {
    playTimer = null;
    agentTurn();
  }, BASE_INTER_TURN_DELAY / speed);
}

function setPlaying(on) {
  playing = on;
  playPauseBtn.textContent = on ? "Pause" : "Play";
  if (!on && playTimer !== null) {
    clearTimeout(playTimer);
    playTimer = null;
  }
  updateControls();
  if (on && !animating && !gameOver) scheduleNext();
}

function agentTurn() {
  if (animating || gameOver) return;
  animating = true;
  updateControls();
  postJSON("/api/agent_turn", {})
    .then(function (r) {
      if (r.status === 409) {
        animating = false;
        if (r.data.error === "human_turn") {
          updateControls();
          return;
        }
        applyBoard(r.data.board);
        gameOver = true;
        updateControls();
        return;
      }
      if (r.status !== 200) {
        animating = false;
        updateControls();
        logSystem(apiErrorText(r.status, r.data));
        return;
      }
      var data = r.data;
      if (data.draw_reason === "TURN_CAP" && data.shot === undefined) {
        // Arrived at an already-spent cap: no shot traveled.
        animating = false;
        applyBoard(data.board);
        gameOver = true;
        updateControls();
        showOverlay("Draw — turn cap reached", [
          { label: "OK", action: function () {} },
        ]);
        return;
      }
      animateShotResponse(data);
    })
    .catch(function (err) {
      animating = false;
      updateControls();
      logSystem("Server unreachable: " + err);
    });
}

function newGame() {
  var body = {};
  var seedText = seedInput.value.trim();
  if (seedText !== "") {
    var parsed = Number(seedText);
    if (!isNaN(parsed)) body.seed = Math.trunc(parsed);
  }
  body.team_modes = {
    team1: composedMode(modeTeam1Sel, modelTeam1Input),
    team2: composedMode(modeTeam2Sel, modelTeam2Input),
  };
  var mt = parseInt(maxTurnsInput.value, 10);
  if (!isNaN(mt) && mt >= 1) body.max_turns = mt;
  setPlaying(false);
  postJSON("/api/new_game", body)
    .then(function (r) {
      if (r.status !== 200) {
        logSystem(apiErrorText(r.status, r.data));
        return;
      }
      seed = r.data.seed;
      shotAnim = null;
      fadingTraj = null;
      explosion = null;
      hitFlashes = [];
      animating = false;
      dialAngle = null;
      applyBoard(r.data);
      syncSetupFromServer(r.data);
      updateControls();
      overlay.classList.add("hidden");
      logSystem("New match started (seed " + seed + ")");
    })
    .catch(function (err) {
      logSystem("Server unreachable: " + err);
    });
}

function finishShot(now, data) {
  explosion = null;
  hitFlashes = [];
  shotAnim = null;
  fadingTraj = {
    points: data.shot.points,
    color: data.shooter.color,
    fadeStart: now,
  };
  applyBoard(data.board);
  // Soft turn-cap draw: the board itself is not finished, but the budget is
  // spent — treat the match as over (draw_reason TURN_CAP).
  if (data.game_over && !gameOver) gameOver = true;
  animating = false;
  updateControls();
  if (gameOver) {
    if (playing) setPlaying(false);
    var winner = data.winner;
    if (winner !== null && winner !== undefined) {
      showOverlay("Team " + winner + " wins the match!", [
        { label: "OK", action: function () {} },
      ]);
    } else if (data.draw_reason === "TURN_CAP") {
      showOverlay("Draw — turn cap reached", [
        { label: "OK", action: function () {} },
      ]);
    } else {
      showOverlay("Nobody wins the match!", [
        { label: "OK", action: function () {} },
      ]);
    }
  } else {
    scheduleNext();
  }
}

// The single post-response animation pipeline, shared by the human fire
// path and the spectator agent path: log → dial → shotAnim → explosion →
// hitFlashes → waitExplosion → finishShot.
function animateShotResponse(data) {
  var now = performance.now();
  logTurn(data.shooter, data.func_str, data.shot, data.agent, data.solver_rung);
  if (data.start_angle !== null && data.start_angle !== undefined) {
    dialAngle = data.start_angle; // display-only; set on fire (GameData.java:1113)
  }
  dialColor = data.shooter.color;
  dialInverted = data.shooter.inverted;
  drawCompass();

  shotAnim = { points: data.shot.points, color: data.shooter.color, start: now };
  if (data.shot.last_x !== null && data.shot.last_y !== null) {
    explosion = { x: data.shot.last_x, y: data.shot.last_y, start: now + FLY_TIME };
  }
  hitFlashes = data.shot.hits
    .filter(function (h) {
      return (
        h[2] < data.shot.points.length &&
        data.shot.points[h[2]][0] !== null &&
        data.shot.points[h[2]][1] !== null
      );
    })
    .map(function (h) {
      var p = data.shot.points[h[2]];
      return { x: p[0], y: p[1], start: now + FLY_TIME };
    });

  // board (kills + turn) lands when the explosion ends
  var waitExplosion = function () {
    var t = performance.now();
    if (!explosion || t - explosion.start >= EXPLOSION_TIME) {
      finishShot(t, data);
    } else {
      requestAnimationFrame(waitExplosion);
    }
  };
  requestAnimationFrame(waitExplosion);
}

function fire() {
  if (animating || gameOver || funcInput.disabled) return;
  var funcStr = funcInput.value;
  if (funcStr.length === 0) return; // GameScreen.java:433

  animating = true;
  updateControls();
  postJSON("/api/fire", { func_str: funcStr })
    .then(function (r) {
      if (r.status === 400) {
        animating = false;
        updateControls();
        logLine([{ cls: "error", text: "malformed function (Python-side parse error)" }]);
        return;
      }
      if (r.status === 409) {
        animating = false;
        applyBoard(r.data.board);
        gameOver = true;
        updateControls();
        return;
      }
      if (r.status !== 200) {
        animating = false;
        updateControls();
        logSystem(apiErrorText(r.status, r.data));
        return;
      }
      var data = r.data;
      if (data.draw_reason === "TURN_CAP" && data.shot === undefined) {
        // Fired at an already-spent cap: no shot traveled.
        animating = false;
        applyBoard(data.board);
        gameOver = true;
        updateControls();
        showOverlay("Draw — turn cap reached", [
          { label: "OK", action: function () {} },
        ]);
        return;
      }
      animateShotResponse(data);
    })
    .catch(function (err) {
      animating = false;
      updateControls();
      logSystem("Server unreachable: " + err);
    });
}

// --- wiring ------------------------------------------------------------------

fireBtn.addEventListener("click", fire);
funcInput.addEventListener("keydown", function (ev) {
  if (ev.key === "Enter") fire();
});
newBtn.addEventListener("click", newGame);
seedInput.addEventListener("keydown", function (ev) {
  if (ev.key === "Enter") newGame();
});
modeTeam1Sel.addEventListener("change", updateModelInputs);
modeTeam2Sel.addEventListener("change", updateModelInputs);
playPauseBtn.addEventListener("click", function () {
  setPlaying(!playing);
});
stepBtn.addEventListener("click", function () {
  if (animating || playing || gameOver) return;
  agentTurn(); // exactly one agent turn while paused
});
speedSel.addEventListener("change", function () {
  speed = parseFloat(speedSel.value); // scales the inter-turn delay only
});
quitBtn.addEventListener("click", function () {
  if (animating) return;
  showOverlay("Quit the current match?", [
    { label: "Yes", action: newGame },
    { label: "No", action: function () {} },
  ]);
});

function frame(now) {
  draw(now);
  requestAnimationFrame(frame);
}

postJSON("/api/new_game", {})
  .then(function (r) {
    if (r.status !== 200) {
      logSystem(apiErrorText(r.status, r.data));
      updateControls();
      return;
    }
    seed = r.data.seed;
    applyBoard(r.data);
    syncSetupFromServer(r.data);
    updateControls();
    logSystem("New match started (seed " + seed + ")");
  })
  .catch(function (err) {
    logSystem("Server unreachable: " + err);
  });
requestAnimationFrame(frame);
