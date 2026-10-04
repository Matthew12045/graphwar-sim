// Reel mode: a recording-friendly presentation of the classic game screen.
//
// Keeps the reference look (#app is untouched) and adds what an "AI vs AI"
// clip needs: a vertical 9:16 black letterbox with the game scaled to fit,
// bot names on the soldier labels instead of "Player N", trails that stay on
// the board for the whole match, crossed-out markers for dead soldiers, and
// one-line taunts from LLM sides (server "say" events) shown in the speech
// bubble and the log. Toggled from the playback bar or with ?reel=1; the
// choice is remembered per browser. app.js calls the window.Reel hooks.
(function () {
  "use strict";

  var STORE_KEY = "graphwar.reel"; // per-viewer convenience only
  var TRAIL_COLOR = "#3e3c3d"; // old trails: thin dark gray
  var TRAIL_ALPHA = 0.55;
  var LETTERBOX = "#090f0d";

  var on = false;
  var modes = { team1: "human", team2: "human" };
  var trails = []; // [{points, color}]
  var toggle = null;

  function prettyModel(name) {
    name = name.replace(/^.*\//, ""); // "Qwen/Qwen3.8-27B" -> "Qwen3.8-27B"
    name = name.replace(/[-_:]?(fp8|fp16|bf16|awq|gptq|int4|int8|q\d\w*)$/i, "");
    return name.replace(/^qwen/i, "Qwen").replace(/(\d+)b\b/gi, "$1B");
  }

  function titleCase(s) {
    return s.replace(/_/g, " ").replace(/\b\w/g, function (c) {
      return c.toUpperCase();
    });
  }

  // The on-board name for a side's driver: "Sniper Bot", "Qwen3.8-27B",
  // "Solver Bot", or the classic "Player N" for a human.
  function driverName(mode, fallback) {
    if (!mode || mode === "human") return fallback;
    var named = { solver: "Solver Bot", random: "Random Bot", straight: "Straight Bot", "67": "Bot 67" };
    if (named[mode]) return named[mode];
    var rest = mode.replace(/^(llm|hybrid):/, "");
    var at = rest.lastIndexOf("@");
    if (at > 0) return titleCase(rest.slice(at + 1)) + " Bot";
    return prettyModel(rest);
  }

  function names(board) {
    var a = driverName(modes.team1, board.teams[0].label);
    var b = driverName(modes.team2, board.teams[1].label);
    if (a === b) {
      a += " (1)";
      b += " (2)";
    }
    return [a, b];
  }

  function fit() {
    var app = document.getElementById("app");
    if (!app) return;
    if (!on) {
      app.style.transform = "";
      document.body.style.removeProperty("--reel-h");
      return;
    }
    // A 9:16 stage that fits the viewport; the 800x600 game spans its width.
    var stageH = Math.min(window.innerHeight, (window.innerWidth * 16) / 9);
    var stageW = (stageH * 9) / 16;
    var scale = stageW / 800;
    app.style.transform = "scale(" + scale + ")";
    document.body.style.setProperty("--reel-w", stageW + "px");
    document.body.style.setProperty("--reel-h", stageH + "px");
    document.body.style.setProperty("--reel-scale", String(scale));
  }

  function setOn(value) {
    on = !!value;
    document.body.classList.toggle("reel", on);
    if (toggle) toggle.checked = on;
    try {
      localStorage.setItem(STORE_KEY, on ? "1" : "0");
    } catch (e) {
      /* storage blocked: the toggle still works for this page */
    }
    fit();
  }

  window.Reel = {
    isOn: function () {
      return on;
    },
    letterbox: LETTERBOX,

    setModes: function (m) {
      if (m) modes = { team1: m.team1, team2: m.team2 };
    },

    newMatch: function (m) {
      trails = [];
      this.setModes(m);
    },

    // The label app.js draws over each soldier of team index i.
    label: function (board, i) {
      return on ? names(board)[i] : board.teams[i].label;
    },

    addTrail: function (points, color) {
      trails.push({ points: points, color: color });
    },

    // Draw every finished trail of this match (oldest first) under the
    // in-flight shot. strokePoints is app.js's own polyline helper.
    drawTrails: function (strokePoints) {
      if (!on) return;
      for (var i = 0; i < trails.length; i++) {
        var last = i === trails.length - 1;
        strokePoints(
          trails[i].points,
          trails[i].points.length,
          last ? trails[i].color : TRAIL_COLOR,
          last ? 0.9 : TRAIL_ALPHA,
        );
      }
    },

    // A dead soldier: red circle with a cross through it.
    drawDead: function (ctx, x, y) {
      if (!on) return;
      ctx.save();
      ctx.strokeStyle = "#e11d1d";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.moveTo(x - 6, y - 6);
      ctx.lineTo(x + 6, y + 6);
      ctx.moveTo(x + 6, y - 6);
      ctx.lineTo(x - 6, y + 6);
      ctx.stroke();
      ctx.restore();
    },
  };

  var bar = document.getElementById("playback-bar");
  if (bar) {
    var label = document.createElement("label");
    label.className = "reel-toggle";
    toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.id = "reel-toggle";
    label.appendChild(toggle);
    label.appendChild(document.createTextNode(" Reel mode"));
    bar.appendChild(label);
    toggle.addEventListener("change", function () {
      setOn(toggle.checked);
    });
  }
  // In reel mode the controls hide so the clip shows only the game; this
  // button (or the "c" key) opens them as a sheet.
  var btn = document.createElement("button");
  btn.id = "reel-controls-btn";
  btn.type = "button";
  btn.textContent = "Controls";
  btn.addEventListener("click", function () {
    document.body.classList.toggle("reel-controls");
  });
  document.body.appendChild(btn);
  document.addEventListener("keydown", function (e) {
    var t = e.target && e.target.tagName;
    if (!on || t === "INPUT" || t === "SELECT" || t === "TEXTAREA") return;
    if (e.key === "c" || e.key === "C") document.body.classList.toggle("reel-controls");
  });
  window.addEventListener("resize", fit);

  var want = /[?&]reel=1\b/.test(location.search);
  if (!want) {
    try {
      want = localStorage.getItem(STORE_KEY) === "1";
    } catch (e) {
      want = false;
    }
  }
  setOn(want);
})();
