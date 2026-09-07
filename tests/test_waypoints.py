"""Tests for M5.5 Slice D1: the waypoint plan schema.

The LLM's ONLY output is a :class:`WaypointPlan` (M5.5.1); validation is
strict per the bridge's reject list and cheap — a schema error never burns a
solver attempt (M5.5.3). ``tol`` below the hit radius is CLAMPED up, never
rejected (M5.5.8).
"""

from __future__ import annotations

import pytest

from agents.waypoints import (
    MAX_WAYPOINTS,
    MIN_WAYPOINT_TOL,
    RATIONALE_MAX_CHARS,
    PlanSchemaError,
    parse_plan,
)

# shooter-relative geometry for the tests: one living enemy at u_T = 20.
TARGET_U_T = {"enemy_0": 20.0, "enemy_1": 24.5}
BAND = (-14.6, 14.6)
STYLES = {"sniper", "howitzer", "serpent", "bodyguard", "professor", "master_magician"}


def _parse(raw: object) -> object:
    return parse_plan(raw, target_u_T=TARGET_U_T, band_lo=BAND[0], band_hi=BAND[1], styles=STYLES)


def _plan(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "target_id": "enemy_0",
        "waypoints": [{"u": 5.0, "y": 2.0, "tol": 2.0, "priority": 1}],
    }
    base.update(overrides)
    return base


def test_valid_plan_parses() -> None:
    plan = _plan(
        secondary_targets=["enemy_1"],
        branch_hint="over",
        style="howitzer",
        rationale="go over the ridge",
    )
    out = parse_plan(plan, target_u_T=TARGET_U_T, band_lo=BAND[0], band_hi=BAND[1], styles=STYLES)
    assert out.target_id == "enemy_0"
    assert out.secondary_targets == ("enemy_1",)
    assert out.branch_hint == "over"
    assert out.style == "howitzer"
    assert out.rationale == "go over the ridge"
    assert out.waypoints[0].u == 5.0


def test_empty_plan_is_valid() -> None:
    out = parse_plan(
        {"target_id": "enemy_0"},
        target_u_T=TARGET_U_T,
        band_lo=BAND[0],
        band_hi=BAND[1],
        styles=STYLES,
    )
    assert out.waypoints == ()
    assert out.branch_hint == "any"


def test_tol_below_hit_radius_is_clamped_not_rejected() -> None:
    out = parse_plan(
        _plan(waypoints=[{"u": 5.0, "y": 0.0, "tol": 0.01}]),
        target_u_T=TARGET_U_T,
        band_lo=BAND[0],
        band_hi=BAND[1],
        styles=STYLES,
    )
    assert out.waypoints[0].tol == MIN_WAYPOINT_TOL


def test_rejects_non_monotone_u() -> None:
    with pytest.raises(PlanSchemaError, match="strictly"):
        parse_plan(
            _plan(waypoints=[{"u": 5.0, "y": 0.0}, {"u": 5.0, "y": 1.0}]),
            target_u_T=TARGET_U_T,
            band_lo=BAND[0],
            band_hi=BAND[1],
            styles=STYLES,
        )


def test_rejects_u_outside_open_interval() -> None:
    with pytest.raises(PlanSchemaError, match="outside"):
        parse_plan(
            _plan(waypoints=[{"u": 0.0, "y": 0.0}]),
            target_u_T=TARGET_U_T,
            band_lo=BAND[0],
            band_hi=BAND[1],
            styles=STYLES,
        )
    with pytest.raises(PlanSchemaError, match="outside"):
        parse_plan(
            _plan(waypoints=[{"u": 20.0, "y": 0.0}]),
            target_u_T=TARGET_U_T,
            band_lo=BAND[0],
            band_hi=BAND[1],
            styles=STYLES,
        )


def test_u_is_validated_against_the_plan_s_own_target() -> None:
    # u=22 is legal for enemy_1 (u_T=24.5) but outside enemy_0's (20).
    out = parse_plan(
        _plan(target_id="enemy_1", waypoints=[{"u": 22.0, "y": 0.0}]),
        target_u_T=TARGET_U_T,
        band_lo=BAND[0],
        band_hi=BAND[1],
        styles=STYLES,
    )
    assert out.target_id == "enemy_1"
    with pytest.raises(PlanSchemaError, match="outside"):
        parse_plan(
            _plan(waypoints=[{"u": 22.0, "y": 0.0}]),
            target_u_T=TARGET_U_T,
            band_lo=BAND[0],
            band_hi=BAND[1],
            styles=STYLES,
        )


def test_rejects_dead_target_id() -> None:
    with pytest.raises(PlanSchemaError, match="dead target_id"):
        parse_plan(
            _plan(target_id="enemy_9"),
            target_u_T=TARGET_U_T,
            band_lo=BAND[0],
            band_hi=BAND[1],
            styles=STYLES,
        )


def test_rejects_more_than_five_waypoints() -> None:
    too_many = [{"u": float(2 * i + 1), "y": 0.0} for i in range(MAX_WAYPOINTS + 1)]
    with pytest.raises(PlanSchemaError, match="at most 5"):
        parse_plan(
            _plan(waypoints=too_many),
            target_u_T=TARGET_U_T,
            band_lo=BAND[0],
            band_hi=BAND[1],
            styles=STYLES,
        )


def test_rejects_y_outside_the_band() -> None:
    with pytest.raises(PlanSchemaError, match="map band"):
        parse_plan(
            _plan(waypoints=[{"u": 5.0, "y": 99.0}]),
            target_u_T=TARGET_U_T,
            band_lo=BAND[0],
            band_hi=BAND[1],
            styles=STYLES,
        )


def test_rejects_unknown_style() -> None:
    with pytest.raises(PlanSchemaError, match="unknown style"):
        parse_plan(
            _plan(style="gremlin"),
            target_u_T=TARGET_U_T,
            band_lo=BAND[0],
            band_hi=BAND[1],
            styles=STYLES,
        )


def test_rejects_unknown_branch_hint_and_unknown_fields() -> None:
    with pytest.raises(PlanSchemaError, match="branch_hint"):
        parse_plan(
            _plan(branch_hint="sideways"),
            target_u_T=TARGET_U_T,
            band_lo=BAND[0],
            band_hi=BAND[1],
            styles=STYLES,
        )
    with pytest.raises(PlanSchemaError, match="unknown fields"):
        parse_plan(
            _plan(expression="0.5*x^2"),
            target_u_T=TARGET_U_T,
            band_lo=BAND[0],
            band_hi=BAND[1],
            styles=STYLES,
        )


def test_rationale_is_truncated_not_rejected() -> None:
    long = "x" * (RATIONALE_MAX_CHARS + 50)
    out = parse_plan(
        _plan(rationale=long),
        target_u_T=TARGET_U_T,
        band_lo=BAND[0],
        band_hi=BAND[1],
        styles=STYLES,
    )
    assert len(out.rationale) == RATIONALE_MAX_CHARS


def test_schema_error_is_a_value_error_and_names_the_field() -> None:
    try:
        parse_plan(
            {"waypoints": []},
            target_u_T=TARGET_U_T,
            band_lo=BAND[0],
            band_hi=BAND[1],
            styles=STYLES,
        )
    except PlanSchemaError as exc:
        assert isinstance(exc, ValueError)
        assert "target_id" in str(exc)
    else:  # pragma: no cover
        pytest.fail("missing target_id must be rejected")
