# personas/

Raw LLM system-prompt fragments ("prepend") and persona style prompts — the
SOURCE MATERIAL for the M5.4 persona harness. Kept as provenance only: the
wired code lives in `agents/personas/` (see
`planning/The_bridge_nobody_wrote_down.md` M5.4.0–M5.4.2 for the spec).

**Status: the six canonical personas are ADAPTED and WIRED.** The adapted
style texts live in `agents/personas/texts.py` (`*_STYLE` constants — the
manifest's enumeration source, bridge M5.4.0); each persona attaches through
`LLMAgent(..., persona=...)` with its machine verifier in
`agents/personas/verifiers.py`. Do not point new code at the files below.

| File | Persona / role | Status |
|------|----------------|--------|
| `Shared_core_prepend_to_every_bot_.txt` | Core shared prompt (frame + hard rules) | SUPERSEDED — `_SYSTEM_PROMPT` in `agents/llm_agent.py` is the adapted core |
| `Shared_core_prepend_to_every_bot_-2.txt` | Sniper | ADAPTED + wired (`agents/personas/`) |
| `Shared_core_prepend_to_every_bot_-3.txt` | Howitzer | ADAPTED + wired (`agents/personas/`) |
| `Shared_core_prepend_to_every_bot_-4.txt` | Serpent | ADAPTED + wired (`agents/personas/`) |
| `Shared_core_prepend_to_every_bot_-5.txt` | Bodyguard | ADAPTED + wired (`agents/personas/`) |
| `Shared_core_prepend_to_every_bot_-6.txt` | Professor | ADAPTED + wired (`agents/personas/`) |
| `The_Master_Magician.txt` | Master Magician | ADAPTED + wired (`agents/personas/`) |
| `file.txt` | Gremlin | RAW, deliberately EXCLUDED — no manifest entry (designer call: a cross-turn rule the fresh-conversation-per-turn agent cannot honor) |

The bridge spec defines SIX canonical personas (sniper, howitzer, serpent,
bodyguard, professor, master_magician); the Gremlin stays here as raw
material with no verifier and no manifest entry.
