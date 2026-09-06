"""Numeric literal emission for agents producing expressions.

Critical emission constraint (see IMPLEMENTATION_PLAN.md and
``tests/test_solver.py``): the game parser rewrites every ``-`` to ``+-``
(unary negation), which corrupts scientific notation (``1e-06`` -> ``1e+-06``
misparses as ``1*e - 6``). Every numeric literal an agent emits must be a
**plain decimal**, never an exponent.

The canonical formatter lives in ``graphwar_sim.solver._num`` (validated by the
M2 round-trip tests); this module re-exports it so all agents share one source
of truth.
"""

from __future__ import annotations

# The canonical implementation, validated by tests/test_solver.py round-trips.
from graphwar_sim.solver import _num as format_literal

__all__ = ["format_literal"]
