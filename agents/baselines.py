"""Baseline agents without LLMs (Phase 3 / M3).

- :class:`StraightShotAgent` — aims a straight line at the nearest enemy in
  the observation's centered world frame (the same ``m·x`` line the solver's
  ``line`` rung uses; terrain occlusion means it usually dies at a rock —
  that is the point of a baseline).
- :class:`RandomAgent` — samples random members of a small closed grammar of
  valid Graphwar expressions (line / quadratic / sine / Gaussian), validating
  each emit through the real parser and logging parse failures / retries.

Every expression is emitted with :func:`agents.emission.format_literal` (plain
decimals, never scientific notation) because the parser rewrites ``-`` to
``+-`` and would corrupt an exponent (see ``agents/emission.py``).
"""

from __future__ import annotations

import random

from graphwar_sim import Game, PolishNotationFunction
from graphwar_sim.parser import MalformedFunction

from .base import AgentStats, Observation
from .emission import format_literal

# RandomAgent retry budget before falling back to a guaranteed-valid dud.
# # TUNABLE — defensive; the grammar below is valid by construction.
_MAX_EMIT_RETRIES: int = 10
# RandomAgent ranges (all # TUNABLE; chosen to keep curves inside the reachable
# world band y in [-14.6, +14.6] most of the time rather than instantly OOB).
_SLOPE_MIN, _SLOPE_MAX = -0.6, 0.6
_QUAD_A_MIN, _QUAD_A_MAX = -0.06, 0.06
_SINE_AMP_MIN, _SINE_AMP_MAX = -6.0, 6.0
_SINE_K_MIN, _SINE_K_MAX = 0.05, 0.4
_GAUSS_AMP_MIN, _GAUSS_AMP_MAX = -12.0, 12.0
_GAUSS_B_MIN, _GAUSS_B_MAX = 0.02, 0.2
_GAUSS_C_MIN, _GAUSS_C_MAX = -10.0, 20.0


def _center_term(c: float) -> str:
    """``x - c`` with the sign folded so the parser's ``-``->``+-`` rewrite
    does not double-negate a negative centre (same trick as solver._gauss_term)."""
    return f"x-({format_literal(c)})" if c >= 0 else f"x+({format_literal(-c)})"


class StraightShotAgent:
    """A straight line toward the nearest enemy (world-frame secant)."""

    name: str = "straight"
    _stats: AgentStats

    def __init__(self) -> None:
        self._stats = AgentStats()

    def act(self, game: Game, obs: Observation) -> str:
        if not obs.enemy_soldiers:
            return "0*x"
        mx, my = obs.shooter
        tx, ty = min(obs.enemy_soldiers, key=lambda p: (p[0] - mx) ** 2 + (p[1] - my) ** 2)
        if abs(tx - mx) < 1e-9:
            return "0*x"
        m = (ty - my) / (tx - mx)
        return f"({format_literal(m)})*x"

    def stats(self) -> AgentStats:
        return self._stats


class RandomAgent:
    """A random member of a small closed grammar of valid expressions.

    Deterministic when constructed with a ``seed`` — the M4 runner seeds one
    instance per match so leaderboards are reproducible from the seed file.
    """

    name: str = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._stats = AgentStats()

    def act(self, game: Game, obs: Observation) -> str:
        for _ in range(_MAX_EMIT_RETRIES):
            expr = self._sample()
            try:
                PolishNotationFunction(expr)
            except MalformedFunction:
                self._stats.parse_failures += 1
                self._stats.retries += 1
                continue
            return expr
        return "0*x"

    def stats(self) -> AgentStats:
        return self._stats

    def _sample(self) -> str:
        rng = self._rng
        kind = rng.randrange(4)
        if kind == 0:  # line
            return f"({format_literal(rng.uniform(_SLOPE_MIN, _SLOPE_MAX))})*x"
        if kind == 1:  # quadratic
            a = rng.uniform(_QUAD_A_MIN, _QUAD_A_MAX)
            b = rng.uniform(_SLOPE_MIN, _SLOPE_MAX)
            return f"({format_literal(a)})*x^2+({format_literal(b)})*x"
        if kind == 2:  # sine
            amp = rng.uniform(_SINE_AMP_MIN, _SINE_AMP_MAX)
            k = rng.uniform(_SINE_K_MIN, _SINE_K_MAX)
            return f"({format_literal(amp)})*sin(({format_literal(k)})*x)"
        # Gaussian lump (kind == 3)
        amp = rng.uniform(_GAUSS_AMP_MIN, _GAUSS_AMP_MAX)
        b = rng.uniform(_GAUSS_B_MIN, _GAUSS_B_MAX)
        c = rng.uniform(_GAUSS_C_MIN, _GAUSS_C_MAX)
        return f"({format_literal(amp)})*e^(-({format_literal(b)})*({_center_term(c)})^2)"


__all__ = ["RandomAgent", "StraightShotAgent"]
