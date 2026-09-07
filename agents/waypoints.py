"""Waypoint plan schema (M5.5.3): the ONLY thing the hybrid LLM emits.

The LLM never emits an expression (M5.5.1 division of labour, non-negotiable):
its output is a JSON plan that :func:`parse_plan` validates into a
:class:`WaypointPlan`. The certified solver compiles the plan into the curve —
the LLM's text never reaches the parser (that is the whole reason
HybridAgent exists).

Validation failures raise :class:`PlanSchemaError` (a ``ValueError``) whose
message is fed back to the LLM WITHOUT burning a solver attempt (M5.5.3) and
counted in the agent's own ``schema_errors`` counter — never mixed into
solver failures or parse failures.

Strictness follows the bridge's reject list: non-monotone ``u``, ``u``
outside ``(0, u_T)``, a dead ``target_id``, more than five waypoints, and an
unknown ``style`` are REJECTED. ``tol`` below the SOLDIER_RADIUS hit radius
is CLAMPED up, not rejected (M5.5.8). ``rationale`` is logged only, never
parsed (truncated to :data:`RATIONALE_MAX_CHARS`, lenient).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from graphwar_sim.corridor import WORLD_RADIUS

# The waypoint count cap (M5.5.3: "0 to 5. Strictly increasing u.").
MAX_WAYPOINTS: int = 5

# ``tol`` is clamped UP to the Phase 0 hit radius (world units), never
# rejected (M5.5.8: "tol below SOLDIER_RADIUS (assert clamped, not rejected)").
MIN_WAYPOINT_TOL: float = WORLD_RADIUS

# Default tol when the plan omits it (loose — the solver drops what binds).
# # TUNABLE — not from source.
DEFAULT_WAYPOINT_TOL: float = max(2.0, MIN_WAYPOINT_TOL)

# rationale is logged only, never parsed (M5.5.3); over-long text is
# truncated rather than rejected (not in the bridge's reject list).
RATIONALE_MAX_CHARS: int = 200

_BRANCH_HINTS: frozenset[str] = frozenset({"over", "under", "any"})


class PlanSchemaError(ValueError):
    """The plan does not match the M5.5.3 schema. The message goes back to
    the LLM verbatim (a schema error is a CHEAP correction: no solver
    attempt, no LP, no oracle call)."""


@dataclass(frozen=True)
class WaypointSpec:
    """One schema waypoint: ``(u, y, tol, priority)``.

    ``u`` is strictly increasing across the list and lives in ``(0, u_T)``;
    ``y`` is SHOOTER-relative height within the map band; ``tol`` is the
    half-width of the allowed vertical band (world units, clamped up to
    :data:`MIN_WAYPOINT_TOL`); ``priority`` 1 = drop first when relaxing,
    higher survives longer (M5.5.5).
    """

    u: float
    y: float
    tol: float
    priority: int


@dataclass(frozen=True)
class WaypointPlan:
    """The validated M5.5.3 plan object."""

    target_id: str
    secondary_targets: tuple[str, ...] = ()
    branch_hint: str = "any"  # "over" | "under" | "any"
    waypoints: tuple[WaypointSpec, ...] = ()
    style: str | None = None
    rationale: str = ""


def _num(value: object, field: str) -> float:
    """A JSON number (bools are NOT numbers); else a schema error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanSchemaError(f"'{field}' must be a number")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise PlanSchemaError(f"'{field}' must be finite")
    return out


def _int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanSchemaError(f"'{field}' must be an integer")
    return int(value)


def _str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise PlanSchemaError(f"'{field}' must be a string")
    return value


def parse_plan(
    raw: object,
    *,
    target_u_T: Mapping[str, float],
    band_lo: float,
    band_hi: float,
    styles: Collection[str],
) -> WaypointPlan:
    """Validate one raw LLM JSON object into a :class:`WaypointPlan`.

    ``target_u_T`` maps EVERY living enemy id to its shooter-relative ``u_T``
    (waypoints live strictly inside ``(0, u_T[target_id])``); ``[band_lo,
    band_hi]`` is the playable band in SHOOTER-relative y (``map_y_bounds() -
    my``); ``styles`` are the known persona ids. Any violation raises
    :class:`PlanSchemaError` with a message addressed to the model.
    """
    if not isinstance(raw, dict):
        raise PlanSchemaError("the plan must be a single JSON object")
    unknown = set(raw) - {
        "target_id",
        "secondary_targets",
        "branch_hint",
        "waypoints",
        "style",
        "rationale",
    }
    if unknown:
        raise PlanSchemaError(f"unknown fields: {sorted(unknown)}")

    target_id = _str(raw.get("target_id"), "target_id")
    if target_id not in target_u_T:
        raise PlanSchemaError(f"dead target_id '{target_id}' — pick one of {sorted(target_u_T)}")
    u_T = target_u_T[target_id]

    raw_secondary = raw.get("secondary_targets", [])
    if raw_secondary is None:
        raw_secondary = []
    if not isinstance(raw_secondary, list):
        raise PlanSchemaError("'secondary_targets' must be a list of living enemy ids")
    secondary: list[str] = []
    for entry in raw_secondary:
        sid = _str(entry, "secondary_targets entry")
        if sid not in target_u_T:
            raise PlanSchemaError(
                f"dead secondary target '{sid}' — pick one of {sorted(target_u_T)}"
            )
        secondary.append(sid)

    branch_hint = raw.get("branch_hint", "any")
    if branch_hint is None:
        branch_hint = "any"
    branch_hint = _str(branch_hint, "branch_hint")
    if branch_hint not in _BRANCH_HINTS:
        raise PlanSchemaError(f"branch_hint must be one of {sorted(_BRANCH_HINTS)}")

    raw_waypoints = raw.get("waypoints", [])
    if raw_waypoints is None:
        raw_waypoints = []
    if not isinstance(raw_waypoints, list):
        raise PlanSchemaError("'waypoints' must be a list")
    if len(raw_waypoints) > MAX_WAYPOINTS:
        raise PlanSchemaError(f"at most {MAX_WAYPOINTS} waypoints (got {len(raw_waypoints)})")
    specs: list[WaypointSpec] = []
    prev_u: float | None = None
    for i, entry in enumerate(raw_waypoints):
        if not isinstance(entry, dict):
            raise PlanSchemaError(f"waypoint {i} must be an object")
        unknown_wp = set(entry) - {"u", "y", "tol", "priority"}
        if unknown_wp:
            raise PlanSchemaError(f"waypoint {i}: unknown fields {sorted(unknown_wp)}")
        u = _num(entry.get("u"), f"waypoint {i} u")
        y = _num(entry.get("y"), f"waypoint {i} y")
        if not 0.0 < u < u_T:
            raise PlanSchemaError(
                f"waypoint {i}: u={u:g} outside (0, u_T={u_T:g}) — endpoints are solver-owned"
            )
        if prev_u is not None and u <= prev_u:
            raise PlanSchemaError(
                f"waypoint {i}: u={u:g} not strictly greater than the previous u={prev_u:g}"
            )
        if not band_lo <= y <= band_hi:
            raise PlanSchemaError(
                f"waypoint {i}: y={y:g} outside the shooter-relative map band "
                f"[{band_lo:g}, {band_hi:g}]"
            )
        raw_tol = entry.get("tol", DEFAULT_WAYPOINT_TOL)
        tol = _num(raw_tol, f"waypoint {i} tol")
        tol = max(tol, MIN_WAYPOINT_TOL)  # clamped UP, never rejected (M5.5.8)
        raw_priority = entry.get("priority", 1)
        priority = _int(raw_priority, f"waypoint {i} priority")
        prev_u = u
        specs.append(WaypointSpec(u=u, y=y, tol=tol, priority=priority))

    raw_style = raw.get("style")
    style: str | None
    if raw_style is None:
        style = None
    else:
        style = _str(raw_style, "style")
        if style not in styles:
            raise PlanSchemaError(f"unknown style '{style}' — pick one of {sorted(styles)} or null")

    raw_rationale = raw.get("rationale", "")
    if raw_rationale is None:
        raw_rationale = ""
    rationale = _str(raw_rationale, "rationale")[:RATIONALE_MAX_CHARS]

    return WaypointPlan(
        target_id=target_id,
        secondary_targets=tuple(secondary),
        branch_hint=branch_hint,
        waypoints=tuple(specs),
        style=style,
        rationale=rationale,
    )


__all__ = [
    "MAX_WAYPOINTS",
    "MIN_WAYPOINT_TOL",
    "PlanSchemaError",
    "RATIONALE_MAX_CHARS",
    "WaypointPlan",
    "WaypointSpec",
    "parse_plan",
]
