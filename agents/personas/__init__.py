"""Persona registry (M5.4.0): the SIX canonical personas, manifest-enumerated.

A typed Python registry stands in for the bridge's ``manifest.yaml`` sketch
(no YAML dependency; the enumeration contract is what matters):

- the ``*_STYLE`` constants in :mod:`agents.personas.texts` are the single
  source of truth for the persona set — the manifest is BUILT from them, so
  adding a style text without metadata fails loudly here, and the tests
  additionally assert ``set(PERSONAS) == {discovered ids}`` (never write the
  persona count as a literal anywhere in the harness — bridge M5.4.0);
- Gremlin is deliberately EXCLUDED (designer call): ``personas/file.txt``
  has no verifier and a cross-turn rule ("never reuse the same family") a
  fresh-conversation-per-turn agent cannot honor. The file stays raw.

Verdict flow (M5.4.1): after each turn the agent runs the persona's verifier
on the emitted expression; ``CONSTRAINT_VIOLATION`` is logged with the
persona id and never crashes the round-robin, and ``MAGICIAN_PARTIAL`` is a
first-class outcome (M5.4.2). The leaderboard keeps "lost the round"
(outcome) separate from "did not play its own game" (violation counts).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import texts
from .verifiers import (
    VERIFIERS,
    VerdictKind,
    VerifierContext,
    VerifierFn,
    VerifierVerdict,
)


@dataclass(frozen=True)
class PersonaSpec:
    """One persona's wiring: prompt text + machine verifier.

    ``verifier`` names a function in :data:`agents.personas.verifiers.VERIFIERS`
    (string, like the bridge manifest's ``verifier: verify_multi_hit`` — the
    manifest test asserts every name resolves).

    ``degrades_to`` is the verdict kind used when full compliance is
    unattainable: ``MAGICIAN_PARTIAL`` for the master magician (a first-class
    partial outcome, M5.4.2); ``CONSTRAINT_VIOLATION`` for the five personas
    without a partial mechanism — their degraded turns are logged as
    violations, never excused.
    """

    id: str
    style_text: str
    constraint_type: str
    verifier: str
    degrades_to: str


# Per-persona metadata (constraint vocabulary from the bridge's table;
# everything else is derived from texts.py at import time).
_PERSONA_META: dict[str, tuple[str, str, str]] = {
    "sniper": ("simplest_rung", "verify_sniper", "CONSTRAINT_VIOLATION"),
    "howitzer": ("high_arc", "verify_howitzer", "CONSTRAINT_VIOLATION"),
    "serpent": ("gap_crossings", "verify_serpent", "CONSTRAINT_VIOLATION"),
    "bodyguard": ("teammate_bands", "verify_bodyguard", "CONSTRAINT_VIOLATION"),
    "professor": ("stated_constraints", "verify_professor", "CONSTRAINT_VIOLATION"),
    "master_magician": ("multi_hit", "verify_master_magician", "MAGICIAN_PARTIAL"),
}


def _style_ids() -> list[str]:
    """The persona ids discovered from ``texts.py`` — the enumeration the
    bridge mandates (no hardcoded count anywhere)."""
    return [
        name[: -len("_STYLE")].lower()
        for name in vars(texts)
        if name.isupper() and name.endswith("_STYLE")
    ]


def _build_manifest() -> dict[str, PersonaSpec]:
    manifest: dict[str, PersonaSpec] = {}
    for pid in _style_ids():
        meta = _PERSONA_META.get(pid)
        if meta is None:
            raise RuntimeError(
                f"persona style text '{pid}' has no _PERSONA_META entry — "
                "add one (constraint_type, verifier, degrades_to) to "
                "agents/personas/__init__.py"
            )
        constraint_type, verifier, degrades_to = meta
        manifest[pid] = PersonaSpec(
            id=pid,
            style_text=getattr(texts, f"{pid.upper()}_STYLE"),
            constraint_type=constraint_type,
            verifier=verifier,
            degrades_to=degrades_to,
        )
    orphan = sorted(set(_PERSONA_META) - set(manifest))
    if orphan:
        raise RuntimeError(f"_PERSONA_META entries without a style text in texts.py: {orphan}")
    return manifest


PERSONAS: dict[str, PersonaSpec] = _build_manifest()


def resolve_verifier(spec: PersonaSpec) -> VerifierFn:
    """The spec's verifier function (KeyError = manifest/registry drift,
    which the manifest test pins)."""
    return VERIFIERS[spec.verifier]


__all__ = [
    "PERSONAS",
    "PersonaSpec",
    "VerifierContext",
    "VerifierFn",
    "VerifierVerdict",
    "VerdictKind",
    "resolve_verifier",
]
