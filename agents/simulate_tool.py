"""The simulate tool: fire a candidate expression through the real physics.

This is the Phase-3 ``simulate(state, expr)`` driver (the sketch
``3._Give_it_a_simulate_tool.py`` used ``sympify`` — banned in the hot path —
so the *structure* is kept but the math is the faithful ported parser and
integrator). It is a **pure oracle**: it does not apply kills (unlike
``Game.fire``), so an agent can probe a candidate before committing to it.

No ``eval``/``exec``/``sympify``: the expression goes through
:class:`~graphwar_sim.parser.PolishNotationFunction` only.
"""

from __future__ import annotations

from dataclasses import dataclass

from graphwar_sim import TEAM2, Game, PolishNotationFunction
from graphwar_sim.physics import process_function_range

from .base import Observation, hit_team_counts


@dataclass(frozen=True)
class SimResult:
    """What firing ``expr`` on the current turn *would* do."""

    parseable: bool
    hit_enemy: bool
    hit_teammate: bool
    num_hits: int  # distinct soldiers struck (enemy + teammate)
    num_steps: int
    # Python-side diagnostic (the reference exception is message-free; see
    # docs/GROUND_TRUTH.md §3.6).
    error: str | None = None


def simulate(game: Game, expr: str, obs: Observation | None = None) -> SimResult:
    """Fire ``expr`` through the physics without applying kills.

    ``obs`` is accepted for interface symmetry with :meth:`Agent.act` (LLM
    agents later build their prompt from it) but is not required — the game
    state alone determines the outcome.
    """
    try:
        f = PolishNotationFunction(expr)
    except Exception as exc:  # noqa: BLE001 - the reference raises no message
        return SimResult(
            parseable=False,
            hit_enemy=False,
            hit_teammate=False,
            num_hits=0,
            num_steps=0,
            error=f"{type(exc).__name__}",
        )

    team = game.state.current_team()
    shooter = team.current_soldier()
    inverted = team.team == TEAM2
    result = process_function_range(f, shooter, game.all_soldiers(), game.terrain, inverted)
    enemy_hits, teammate_hits = hit_team_counts(game, result)
    return SimResult(
        parseable=True,
        hit_enemy=enemy_hits > 0,
        hit_teammate=teammate_hits > 0,
        num_hits=enemy_hits + teammate_hits,
        num_steps=result.num_steps,
    )


__all__ = ["SimResult", "simulate"]
