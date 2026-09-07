# sketches/

Early design sketches and drafts, **superseded** by the real implementation
in `../graphwar_sim/`, `../agents/`, and `../tests/`. Kept for provenance and
design history; none of these are imported or run.

| File | What it sketched | Superseded by |
|------|------------------|---------------|
| `1._Ladder_dispatch_in_solver.solve_.py` | Ladder dispatch in `solver.solve()` | `graphwar_sim/solver.py` |
| `1._Ladder_dispatch_in_solver.solve_-2.py` | CCF contract types | `graphwar_sim/ccf.py` |
| `2._Property-test_scaffolding.py` | Property-test scaffolding | `tests/test_ccf_property.py` |
| `2._Property-test_scaffolding-2.py` | Property-test scaffolding (v2) | `tests/test_ccf_property.py` |
| `3._Give_it_a_simulate_tool.py` | Simulate tool (used `sympify`) | `agents/simulate_tool.py` |
| `Implementation.py` | CVXPY corridor solver | `graphwar_sim/ccf.py`, `graphwar_sim/corridor.py` |

Do not treat these as current behavior — the package code is authoritative.
