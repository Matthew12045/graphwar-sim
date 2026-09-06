"""M3: Graphwar agents.

Minimal viable slice (IMPLEMENTATION_PLAN.md §"Recommended first milestone"):
a deterministic solver agent plus two baselines, headless, no LLM. The plan's
``LLMAgent`` / ``HybridAgent`` and the style prompts are deferred.

Public surface:
    - :class:`Observation`, :class:`AgentStats`, :class:`AgentProto` protocol
    - :func:`agents.observation.observe` (centered world frame)
    - :func:`agents.simulate_tool.simulate` (pure physics oracle)
    - :class:`SolverAgent`, :class:`RandomAgent`, :class:`StraightShotAgent`
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
from .observation import observe
from .simulate_tool import SimResult, simulate
from .solver_agent import SolverAgent

__all__ = [
    "Agent",
    "AgentStats",
    "Observation",
    "FRAME_CENTERED_WORLD",
    "RandomAgent",
    "SimResult",
    "SolverAgent",
    "StraightShotAgent",
    "format_literal",
    "hit_team_counts",
    "observe",
    "plane_from_world",
    "simulate",
    "world_coords",
]
