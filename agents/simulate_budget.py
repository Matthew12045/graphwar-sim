"""Per-turn budget wrapper around the simulate tool (M5.3).

A thin, stateful accounting shell over the **unmodified** pure oracle
:mod:`agents.simulate_tool` (5.2.md appendix, M5.3 patch: "Wrap it with a call
counter and per-turn budget. Do not rewrite it."). M5.4.2 depends on
``simulate_tool`` staying byte-identical — the pure oracle every persona
prompt cites — so every budget rule lives here:

- Count-based only. There is deliberately NO wall-clock budget anywhere in
  M5.3: a count is a function of the (game, expression) sequence alone, so
  seeded determinism survives by construction.
- A delegated call burns budget **regardless of parseability** — the agent
  spent the oracle's attention on the string either way.
- An over-budget call is recorded as ``denied`` in the ledger and raises
  :class:`SimulateBudgetExhausted` WITHOUT delegating (a denied call must not
  leak oracle information).
- The budget is PER TURN: :meth:`BudgetedSimulator.new_turn` resets the
  per-turn counters and closes the finished turn into :attr:`turn_log`.

No ``eval``/``exec``/dynamic import (repo security ground rule); the
expression still reaches the physics only through the ported parser inside
``simulate_tool.simulate``.
"""

from __future__ import annotations

from graphwar_sim import Game

from .simulate_tool import SimResult, simulate

# The default per-turn simulate budget. Middle of the planned M5.4/M5.5
# ablation grid N in {0, 3, 10}; a turn fires exactly once, so 3 probes per
# turn is roughly "verify a candidate, then one adjust-and-recheck".
# TUNABLE — not from source (no simulate consumer exists yet; the ablation
# protocol in docs/OPEN_QUESTIONS.md fixes the grid).
DEFAULT_SIMULATE_BUDGET: int = 3


class SimulateBudgetExhausted(Exception):
    """Raised by :meth:`BudgetedSimulator.simulate` when this turn's budget
    is spent. Message-free by convention (mirrors MalformedFunction); the
    ledger records the denial."""


class BudgetedSimulator:
    """Per-turn call budget + ledger over :func:`agents.simulate_tool.simulate`.

    The per-turn metadata pattern imitates ``RecordingSolverAgent``
    (eval/run_ccf_battery.py:52): a thin recording shell whose per-turn
    ``dict`` records are harvested after the match by whoever owns the agent
    loop. This is **infrastructure for the M5.4/M5.5 consumers** (the M3
    minimal slice shipped no simulate-consuming agent); until one exists the
    class is exercised only by its own tests.

    Contract for the consuming agent loop (no consumer exists yet — this is
    the documented assumption ``HybridAgent`` (M5.5) must satisfy):

    - Call :meth:`new_turn` ONCE at the START of every turn. It closes the
      turn that just finished into :attr:`turn_log` (``{"turn", "calls",
      "denied"}``, turn index 0-based from construction) and zeroes the
      per-turn counters.
    - Call :meth:`new_turn` once more after the final turn to flush it — a
      turn still open has no ledger record yet (its counters are live via
      :attr:`calls_used` / :attr:`denied_used`).
    - :attr:`remaining` is the string-ready count for the M5.5.6 feedback
      format ("simulate_tool calls remaining: N"); it never goes negative
      (denied calls don't consume).
    """

    __slots__ = (
        "_budget",
        "_calls_used",
        "_denied_used",
        "_game",
        "_turn_calls",
        "_turn_denied",
        "_turn_index",
        "turn_log",
    )

    def __init__(self, game: Game, budget: int = DEFAULT_SIMULATE_BUDGET) -> None:
        self._game = game
        self._budget = budget
        self._calls_used = 0
        self._denied_used = 0
        self._turn_index = 0
        self._turn_calls = 0
        self._turn_denied = 0
        self.turn_log: list[dict[str, int]] = []

    def simulate(self, expr: str) -> SimResult:
        """Fire ``expr`` through the pure oracle against this turn's budget.

        Budget is checked FIRST: an over-budget call is recorded as denied
        and raises without delegating. A delegated call counts regardless of
        whether the expression parses. Delegation is a pass-through — the
        returned :class:`~agents.simulate_tool.SimResult` is exactly what a
        direct ``simulate(self._game, expr)`` call would return (purity).
        """
        if self._turn_spent():
            self._record_denied()
            raise SimulateBudgetExhausted()
        self._calls_used += 1
        self._turn_calls += 1
        return simulate(self._game, expr)

    def new_turn(self) -> None:
        """Close the finished turn into :attr:`turn_log` and open the next one.

        See the class docstring for the consuming-loop contract. Identical
        call sequences on identical games produce identical ledgers
        (count-based determinism).
        """
        self.turn_log.append(
            {"turn": self._turn_index, "calls": self._turn_calls, "denied": self._turn_denied}
        )
        self._turn_index += 1
        self._turn_calls = 0
        self._turn_denied = 0

    @property
    def remaining(self) -> int:
        """Simulate calls this turn can still delegate (never negative)."""
        return max(0, self._budget - self._turn_calls)

    @property
    def calls_used(self) -> int:
        """Delegated calls since construction (all turns, incl. the open one)."""
        return self._calls_used

    @property
    def denied_used(self) -> int:
        """Denied calls since construction (all turns, incl. the open one)."""
        return self._denied_used

    # -- internals ------------------------------------------------------------

    def _turn_spent(self) -> bool:
        return self._turn_calls >= self._budget

    def _record_denied(self) -> None:
        self._denied_used += 1
        self._turn_denied += 1


__all__ = ["DEFAULT_SIMULATE_BUDGET", "BudgetedSimulator", "SimulateBudgetExhausted"]
