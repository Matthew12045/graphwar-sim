// Broadcast view: an "AI vs AI" presentation layer for recording matches.
//
// Purely additive: app.js calls the window.Broadcast hooks below at the
// moments it already handles (new board, turn start, live LLM deltas, shot
// landed); nothing inside #app is repositioned. Toggled from the playback
// bar or with ?broadcast=1; the choice is remembered per browser.
//
// Layout when on: a scoreboard strip above the game (driver/model per side,
// soldiers alive, hits, turn), a large caption under it with the function
// just fired and its outcome, and one live "mind" panel per side streaming
// that side's reasoning while it thinks. textContent only — model output is
// never parsed as HTML.
(function () {
  "use strict";

  var STORE_KEY = "graphwar.broadcast"; // # TUNABLE — per-viewer convenience
  var MIND_CHAR_CAP = 4000; // keep the tail of long reasoning streams

  var on = false;
  var modes = { team1: "human", team2: "human" };
  var board = null;
  var hits = [0, 0];
  var thinkingSide = null; // player_index streaming right now
  var root = {};

  function prettyDriver(mode) {
    if (!mode) return "Human";
    var named = {
      human: "Human",
      solver: "Solver",
      random: "Random",
      straight: "Straight Shot",
      "67": "Bot 67",
    };
    if (named[mode]) return named[mode];
    var plan = mode.indexOf("hybrid:") === 0;
    var rest = mode.replace(/^(llm|hybrid):/, "");
    var persona = "";
    var at = rest.lastIndexOf("@");
    if (at > 0) {
      persona = rest.slice(at + 1).replace(/_/g, " ");
      rest = rest.slice(0, at);
    }
    rest = rest.replace(/^.*\//, ""); // "Qwen/Qwen3.8-27B" -> "Qwen3.8-27B"
    rest = rest.replace(/[-_:]?(fp8|fp16|bf16|awq|gptq|int4|int8|q\d\w*)$/i, "");
    // qwen3.8-27b -> Qwen3.8-27B
    rest = rest.replace(/^qwen/i, "Qwen").replace(/(\d+)b\b/gi, "$1B");
    var parts = [rest];
    if (plan) parts.push("plan");
    if (persona) parts.push(persona);
    return parts.join(" · ");
  }

  function isLLM(mode) {
    return !!mode && (mode.indexOf("llm:") === 0 || mode.indexOf("hybrid:") === 0);
  }

  function mk(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function build() {
    var app = document.getElementById("app");
    var top = mk("div", "bc-top hidden");
    top.id = "bc-top";
    root.cards = [];
    for (var i = 0; i < 2; i++) {
      var card = mk("div", "bc-card");
      var name = mk("div", "bc-label");
      var driver = mk("div", "bc-driver");
      var stats = mk("div", "bc-stats");
      card.appendChild(name);
      card.appendChild(driver);
      card.appendChild(stats);
      root.cards.push({ card: card, name: name, driver: driver, stats: stats });
    }
    var mid = mk("div", "bc-vs");
    root.vs = mk("div", "bc-vs-big", "VS");
    root.turn = mk("div", "bc-turn", "");
    mid.appendChild(root.vs);
    mid.appendChild(root.turn);
    top.appendChild(root.cards[0].card);
    top.appendChild(mid);
    top.appendChild(root.cards[1].card);
    app.parentNode.insertBefore(top, app);

    var caption = mk("div", "bc-caption hidden");
    caption.id = "bc-caption";
    root.capWho = mk("span", "bc-cap-who", "");
    root.capFunc = mk("span", "bc-cap-func", "Waiting for the first shot…");
    root.capOutcome = mk("span", "bc-cap-outcome", "");
    caption.appendChild(root.capWho);
    caption.appendChild(root.capFunc);
    caption.appendChild(root.capOutcome);

    var minds = mk("div", "bc-minds hidden");
    minds.id = "bc-minds";
    root.minds = [];
    for (var j = 0; j < 2; j++) {
      var panel = mk("div", "bc-mind");
      var head = mk("div", "bc-mind-head");
      var status = mk("span", "bc-mind-status", "");
      var title = mk("span", "bc-mind-title", "");
      head.appendChild(title);
      head.appendChild(status);
      var body = mk("div", "bc-mind-body");
      panel.appendChild(head);
      panel.appendChild(body);
      minds.appendChild(panel);
      root.minds.push({ panel: panel, title: title, status: status, body: body });
    }
    app.parentNode.insertBefore(caption, app.nextSibling);
    caption.parentNode.insertBefore(minds, caption.nextSibling);
    root.top = top;
    root.caption = caption;
    root.mindsBox = minds;

    var bar = document.getElementById("playback-bar");
    if (bar) {
      var label = mk("label", "bc-toggle");
      var box = document.createElement("input");
      box.type = "checkbox";
      box.id = "broadcast-toggle";
      label.appendChild(box);
      label.appendChild(document.createTextNode(" Broadcast view"));
      bar.appendChild(label);
      box.addEventListener("change", function () {
        setOn(box.checked);
      });
      root.toggle = box;
    }
  }

  function setOn(value) {
    on = !!value;
    document.body.classList.toggle("broadcast", on);
    root.top.classList.toggle("hidden", !on);
    root.caption.classList.toggle("hidden", !on);
    root.mindsBox.classList.toggle("hidden", !on);
    if (root.toggle) root.toggle.checked = on;
    try {
      localStorage.setItem(STORE_KEY, on ? "1" : "0");
    } catch (e) {
      /* storage blocked: the toggle still works for this page */
    }
    render();
  }

  function teamMode(i) {
    return i === 0 ? modes.team1 : modes.team2;
  }

  function render() {
    if (!board || !root.cards) return;
    var shooterIdx = board.shooter ? board.shooter.player_index : -1;
    for (var i = 0; i < 2; i++) {
      var team = board.teams[i];
      var c = root.cards[i];
      c.card.style.setProperty("--team", team.color);
      c.card.classList.toggle("active", !board.finished && i === shooterIdx);
      c.card.classList.toggle("winner", board.finished && board.winner === team.team);
      c.name.textContent = team.label;
      c.driver.textContent = prettyDriver(teamMode(i));
      c.stats.textContent =
        "alive " + team.num_alive + "/" + team.soldiers.length + "  ·  hits " + hits[i];
      var m = root.minds[i];
      m.panel.style.setProperty("--team", team.color);
      m.panel.classList.toggle("idle", !isLLM(teamMode(i)));
      m.title.textContent = team.label + " — " + prettyDriver(teamMode(i));
    }
    root.turn.textContent = board.finished ? "FINAL" : "TURN " + (board.turns_played + 1);
  }

  function setStatus(i, text, live) {
    var m = root.minds[i];
    m.status.textContent = text;
    m.panel.classList.toggle("live", !!live);
  }

  function appendMind(i, text, cls) {
    var body = root.minds[i].body;
    var last = body.lastChild;
    if (!last || last.className !== cls) {
      last = mk("span", cls, "");
      body.appendChild(last);
    }
    last.textContent += text;
    // Trim the oldest text so long reasoning never grows the page unbounded.
    while (body.textContent.length > MIND_CHAR_CAP && body.firstChild) {
      var first = body.firstChild;
      var excess = body.textContent.length - MIND_CHAR_CAP;
      if (first.textContent.length <= excess) body.removeChild(first);
      else first.textContent = "…" + first.textContent.slice(excess + 1);
    }
    body.scrollTop = body.scrollHeight;
  }

  window.Broadcast = {
    prettyDriver: prettyDriver,

    newMatch: function (b, m) {
      hits = [0, 0];
      thinkingSide = null;
      if (root.minds) {
        for (var i = 0; i < 2; i++) {
          root.minds[i].body.textContent = "";
          setStatus(i, "", false);
        }
        root.capWho.textContent = "";
        root.capFunc.textContent = "Waiting for the first shot…";
        root.capOutcome.textContent = "";
        root.capOutcome.className = "bc-cap-outcome";
      }
      this.board(b, m);
    },

    board: function (b, m) {
      board = b;
      if (m) modes = { team1: m.team1, team2: m.team2 };
      render();
    },

    turnStart: function (shooter) {
      if (!root.minds || !shooter) return;
      thinkingSide = shooter.player_index;
      var m = root.minds[thinkingSide];
      m.body.textContent = "";
      setStatus(thinkingSide, "thinking…", true);
    },

    // Live LLM feed: kind is "thinking" or "text" (answer tokens).
    delta: function (kind, text) {
      if (thinkingSide === null || !text) return;
      appendMind(thinkingSide, text, kind === "thinking" ? "bc-think" : "bc-answer");
    },

    note: function (text) {
      if (thinkingSide === null || !text) return;
      appendMind(thinkingSide, "\n" + text + "\n", "bc-note");
    },

    shot: function (data) {
      if (!root.capFunc) return;
      var idx = data.shooter.player_index;
      var hit = data.shot.hits.length;
      hits[idx] += hit;
      root.caption.style.setProperty("--team", data.shooter.color);
      root.capWho.textContent = data.shooter.label + "  y =";
      root.capFunc.textContent = data.func_str;
      root.capOutcome.textContent = hit ? (hit > 1 ? "HIT ×" + hit : "HIT") : "MISS";
      root.capOutcome.className = "bc-cap-outcome " + (hit ? "hit" : "miss");
      // Restart the pop animation.
      void root.capOutcome.offsetWidth;
      root.capOutcome.classList.add("pop");
      setStatus(idx, hit ? "fired · HIT" : "fired · miss", false);
      thinkingSide = null;
      render();
    },

    turnEnded: function () {
      if (thinkingSide !== null) setStatus(thinkingSide, "", false);
      thinkingSide = null;
    },
  };

  build();
  var want = /[?&]broadcast=1\b/.test(location.search);
  if (!want) {
    try {
      want = localStorage.getItem(STORE_KEY) === "1";
    } catch (e) {
      want = false;
    }
  }
  setOn(want);
})();
