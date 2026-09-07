# personas/

Shared LLM system-prompt fragments ("prepend") and persona style prompts for
agent-based play. The M5.4 persona harness (`planning/The_bridge_nobody_wrote_down.md`)
defines six canonical personas (sniper, howitzer, serpent, bodyguard,
professor, master_magician); `The_Master_Magician.txt` and `file.txt`
(Gremlin) are additional/alternate styles.

| File | Persona / role |
|------|----------------|
| `Shared_core_prepend_to_every_bot_.txt` | Core shared prompt (frame + hard rules) |
| `Shared_core_prepend_to_every_bot_-2.txt` | Sniper |
| `Shared_core_prepend_to_every_bot_-3.txt` | Howitzer |
| `Shared_core_prepend_to_every_bot_-4.txt` | Serpent |
| `Shared_core_prepend_to_every_bot_-5.txt` | Bodyguard |
| `Shared_core_prepend_to_every_bot_-6.txt` | Professor |
| `The_Master_Magician.txt` | Master Magician |
| `file.txt` | Gremlin |

Not yet wired into `agents/llm_agent.py` (personas are a separate workstream —
see `agents/llm_agent.py` header). If the M5.4 consolidation lands, these
migrate to `agents/personas/` per the bridge spec.
