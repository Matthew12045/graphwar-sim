"""M3: Graphwar agents.

Minimal viable slice (IMPLEMENTATION_PLAN.md §"Recommended first milestone"):
a deterministic solver agent plus two baselines, headless, no LLM.

M5.3 adds the simulate budget wrapper (:class:`BudgetedSimulator`) around the
untouched pure oracle — per-turn call budget + ledger for the M5.4/M5.5
consume side.

M5.4 adds :class:`LLMAgent` — the Anthropic tool-use shot generator over the
budgeted simulate tool (optional ``anthropic`` extra, lazily imported; safe
to import without it). ``HybridAgent`` and the persona style prompts remain
deferred (the separate personas workstream).

Public surface:
    - :class:`Observation`, :class:`AgentStats`, :class:`AgentProto` protocol
    - :func:`agents.observation.observe` (centered world frame)
    - :func:`agents.simulate_tool.simulate` (pure physics oracle)
    - :class:`agents.simulate_budget.BudgetedSimulator` (per-turn budget)
    - :class:`SolverAgent`, :class:`RandomAgent`, :class:`StraightShotAgent`
    - :class:`LLMAgent` (Anthropic tool-use; optional ``anthropic`` extra)
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
from .baselines import RandomAgent, StraightShotAgent
from .emission import format_literal
from .llm_agent import LLMAgent
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
    "BudgetedSimulator",
    "DEFAULT_SIMULATE_BUDGET",
    "Observation",
    "FRAME_CENTERED_WORLD",
    "LLMAgent",
    "RandomAgent",
    "SimResult",
    "SimulateBudgetExhausted",
    "SolverAgent",
    "StraightShotAgent",
    "format_literal",
    "hit_team_counts",
    "observe",
    "plane_from_world",
    "simulate",
    "world_coords",
]
