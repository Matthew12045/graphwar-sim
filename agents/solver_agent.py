"""SolverAgent: wraps the deterministic M2 solver, no LLM involved.

``solve(game)`` returns a :class:`~graphwar_sim.solver.SolverResult` whose
``expression`` is legal, self-verified through the real integrator, and never
hits a teammate (see ``graphwar_sim/solver.py``). The agent is therefore
*stateless*: every turn it re-runs ``solve`` on the live game and emits exactly
``result.expression`` — a diff against ``solve``'s output is part of the M3
tests (``tests/test_agents.py``).
"""

from __future__ import annotations

from graphwar_sim import Game
from graphwar_sim.solver import solve

from .base import AgentStats, Observation

__all__ = ["SolverAgent"]


class SolverAgent:
    """Plays with the deterministic M2 solver. ``rung_history`` records the
    degradation rung of every emitted shot (a Phase-4 informational metric)."""

    name: str = "solver"

    def __init__(self) -> None:
        self._stats = AgentStats()
        self.rung_history: list[str] = []

    def act(self, game: Game, obs: Observation) -> str:
        result = solve(game)  # ignores the observation: it has the full game
        self.rung_history.append(result.rung)
        return result.expression

    def stats(self) -> AgentStats:
        return self._stats
