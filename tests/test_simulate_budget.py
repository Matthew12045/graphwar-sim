"""Tests for the M5.3 per-turn simulate budget wrapper.

The wrapper must be a pure accounting shell: identical ``SimResult`` fields to
a direct oracle call, budget burned by every delegated call (parseable or
not), denials recorded without delegating, and a ledger that is a pure
function of the (game, call-sequence) — count-based, so seeded determinism
holds by construction.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import agents.simulate_budget as simulate_budget_module
from agents import (
    DEFAULT_SIMULATE_BUDGET,
    BudgetedSimulator,
    SimulateBudgetExhausted,
    simulate,
)
from graphwar_sim import Game

PARSEABLE = "0.05*x"
UNPARSEABLE = "x +"


def _game(seed: int = 7) -> Game:
    return Game.create(seed, num_soldiers=1)


# --- Purity -------------------------------------------------------------------


def test_delegation_is_pure() -> None:
    """The wrapper returns exactly what a direct ``simulate(game, expr)``
    returns — same frozen ``SimResult`` fields for the same (game, expr)."""
    game = _game()
    wrapper = BudgetedSimulator(game)
    for expr in (PARSEABLE, UNPARSEABLE):
        assert wrapper.simulate(expr) == simulate(game, expr)


# --- Counting -----------------------------------------------------------------


def test_every_delegated_call_burns_budget() -> None:
    """Delegated calls count regardless of parseability; denied calls do not
    consume (they never reach the oracle)."""
    wrapper = BudgetedSimulator(_game(), budget=2)
    wrapper.simulate(PARSEABLE)
    wrapper.simulate(UNPARSEABLE)
    assert wrapper.calls_used == 2
    with pytest.raises(SimulateBudgetExhausted):
        wrapper.simulate(PARSEABLE)
    assert wrapper.calls_used == 2
    assert wrapper.remaining == 0


def test_zero_budget_denies_first_call() -> None:
    """``budget=0``: the very first call is denied without delegating."""
    wrapper = BudgetedSimulator(_game(), budget=0)
    with pytest.raises(SimulateBudgetExhausted):
        wrapper.simulate(PARSEABLE)
    assert wrapper.calls_used == 0
    assert wrapper.remaining == 0
    wrapper.new_turn()
    assert wrapper.turn_log == [{"turn": 0, "calls": 0, "denied": 1}]


def test_exhaustion_denies_and_ledgers() -> None:
    """Exhaustion raises every time, records each denial, and ``remaining``
    never goes negative."""
    wrapper = BudgetedSimulator(_game(), budget=1)
    wrapper.simulate(PARSEABLE)
    for _ in range(3):
        with pytest.raises(SimulateBudgetExhausted):
            wrapper.simulate(PARSEABLE)
        assert wrapper.remaining == 0
    wrapper.new_turn()
    assert wrapper.turn_log == [{"turn": 0, "calls": 1, "denied": 3}]
    assert wrapper.denied_used == 3


# --- Turn ledger ----------------------------------------------------------------


def test_new_turn_resets_and_accumulates_ledger() -> None:
    """``new_turn()`` closes the finished turn into ``turn_log`` and zeroes
    the per-turn counters; cumulative counters keep counting."""
    wrapper = BudgetedSimulator(_game(), budget=2)
    wrapper.simulate(PARSEABLE)
    wrapper.new_turn()
    wrapper.simulate(PARSEABLE)
    wrapper.simulate(UNPARSEABLE)
    wrapper.new_turn()
    assert wrapper.turn_log == [
        {"turn": 0, "calls": 1, "denied": 0},
        {"turn": 1, "calls": 2, "denied": 0},
    ]
    assert wrapper.remaining == 2  # per-turn budget restored
    assert wrapper.calls_used == 3  # cumulative across turns


def _scripted_run() -> tuple[list[dict], int, int]:
    """One fixed call sequence: 2 calls, turn close, 2 calls + 1 denial,
    turn close. Returns (ledger, cumulative calls, cumulative denials)."""
    wrapper = BudgetedSimulator(_game(), budget=2)
    for expr in (PARSEABLE, UNPARSEABLE):
        wrapper.simulate(expr)
    wrapper.new_turn()
    wrapper.simulate(PARSEABLE)
    wrapper.simulate(PARSEABLE)  # second call: budget exactly spent
    with pytest.raises(SimulateBudgetExhausted):
        wrapper.simulate(PARSEABLE)
    wrapper.new_turn()
    return wrapper.turn_log, wrapper.calls_used, wrapper.denied_used


def test_identical_sequences_produce_identical_ledgers() -> None:
    """Count-based determinism: same seed + same call sequence => identical
    ledger and counters (no wall-clock term anywhere)."""
    first = _scripted_run()
    second = _scripted_run()
    assert first == second
    log, calls, denied = first
    assert log == [
        {"turn": 0, "calls": 2, "denied": 0},
        {"turn": 1, "calls": 2, "denied": 1},
    ]
    assert (calls, denied) == (4, 1)


# --- Constants / hygiene ---------------------------------------------------------


def test_default_budget_is_on_the_ablation_grid() -> None:
    """The default sits on the planned M5.4/M5.5 ablation grid N in {0, 3, 10}."""
    assert DEFAULT_SIMULATE_BUDGET in (0, 3, 10)


def test_grep_no_eval_exec_dynamic_import() -> None:
    """Repo security convention (5.2.md §11): the wrapper module never calls
    eval/exec or imports dynamically."""
    src = Path(simulate_budget_module.__file__).read_text()
    for pat in (r"\beval\s*\(", r"\bexec\s*\(", r"importlib", r"__import__"):
        assert re.search(pat, src) is None, f"simulate_budget matches {pat!r}"
