"""Numeric literal emission for agents producing expressions.

Critical emission constraint (see IMPLEMENTATION_PLAN.md and
``tests/test_solver.py``): the game parser rewrites every ``-`` to ``+-``
(unary negation), which corrupts scientific notation (``1e-06`` -> ``1e+-06``
misparses as ``1*e - 6``). Every numeric literal an agent emits must be a
**plain decimal**, never an exponent.

The canonical formatter lives in :mod:`graphwar_sim.emission` (moved there in
M5.2 so the solver and the CCF rung share one source of truth; previously
``graphwar_sim.solver._num`` — same implementation, validated by the same
round-trip tests); this module re-exports it so all agents share it.
"""

from __future__ import annotations

# The canonical implementation, validated by tests/test_emission.py and the
# solver round-trips.
from graphwar_sim.emission import format_literal

__all__ = ["format_literal"]
