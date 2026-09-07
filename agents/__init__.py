"""Graphwar agents.

M3 minimal slice (IMPLEMENTATION_PLAN.md §"Recommended first milestone"): a
deterministic solver agent plus the baselines, headless, no LLM.

M5.3 added the simulate budget wrapper (:class:`BudgetedSimulator`) around the
untouched pure oracle — per-turn call budget + ledger for the M5.4/M5.5
consume side.

M5.4 added :class:`LLMAgent` — the Anthropic tool-use shot generator over the
budgeted simulate tool (optional ``anthropic`` extra, lazily imported; safe
to import without it) — plus the WIRED persona harness
(:mod:`agents.personas`): the manifest (:data:`agents.personas.PERSONAS`,
enumerated from the adapted style texts), the per-persona machine verifiers,
and roster entries ``llm:<model>[@<persona>]``. Persona verdicts ride
``rung_history`` so the runner and the UI ``[rung: ...]`` line work unchanged.

M5.5 added :class:`HybridAgent` (roster ``hybrid:<model>[@<persona>]``): the
LLM emits ONLY a waypoint plan (:mod:`agents.waypoints` schema); the
certified CCF solver compiles the shot through the M5.5.5 relaxation ladder
(:func:`agents.hybrid_agent.solve_plan_with_ladder`) — the LLM's text never
reaches the parser.

Public surface:
    - :class:`Observation`, :class:`AgentStats`, :class:`Agent` protocol
    - :func:`agents.observation.observe` (centered world frame)
    - :func:`agents.simulate_tool.simulate` (pure physics oracle)
    - :class:`agents.simulate_budget.BudgetedSimulator` (per-turn budget)
    - :class:`SolverAgent`, :class:`RandomAgent`, :class:`StraightShotAgent`,
      :class:`Bot67Agent` (roster key ``"67"``; not in the default round-robin)
    - :class:`LLMAgent` (Anthropic tool-use; optional ``anthropic`` extra)
    - :class:`HybridAgent` + the M5.5.5 ladder entry
      (:func:`agents.hybrid_agent.solve_plan_with_ladder`)
    - :class:`TurnCancelled` (the mid-turn cancel path)
    - :mod:`agents.personas` (manifest + verifiers) and
      :mod:`agents.waypoints` (the M5.5.3 plan schema)
"""

from __future__ import annotations

from .base import (
    FRAME_CENTERED_WORLD,
    Agent,
    AgentStats,
    Observation,
    hit_team_counts,
    plane_from_world,
    world_coords,
)
from .baselines import Bot67Agent, RandomAgent, StraightShotAgent
from .emission import format_literal
from .hybrid_agent import HybridAgent
from .llm_agent import LLMAgent, TurnCancelled
from .observation import observe
from .simulate_budget import (
    DEFAULT_SIMULATE_BUDGET,
    BudgetedSimulator,
    SimulateBudgetExhausted,
)
from .simulate_tool import SimResult, simulate
from .solver_agent import SolverAgent

__all__ = [
    "Agent",
    "AgentStats",
    "Bot67Agent",
    "BudgetedSimulator",
    "DEFAULT_SIMULATE_BUDGET",
    "HybridAgent",
    "Observation",
    "FRAME_CENTERED_WORLD",
    "LLMAgent",
    "RandomAgent",
    "SimResult",
    "SimulateBudgetExhausted",
    "SolverAgent",
    "StraightShotAgent",
    "TurnCancelled",
    "format_literal",
    "hit_team_counts",
    "observe",
    "plane_from_world",
    "simulate",
    "world_coords",
]
