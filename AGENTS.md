# AGENTS.md

## Editable installs must point at the repo root

Run every `pip install -e` from this repo root — never from a `.kilo/worktrees/*`
worktree. A worktree-targeted editable install shadows the working tree with
stale code in any context that resolves imports off `sys.path` instead of CWD
(script files run by path, `multiprocessing` spawn workers), while `python -m`
and stdin runs still see the real tree — so different entry points silently
measure different code.

Symptom: two runs of the same solver on the same seeds disagree on outcomes.

Check first: `python3 -m pip show graphwar-sim` — `Editable project location`
must be `/Users/matthurindo/graph_war`. Fix: `python3 -m pip install -e . --no-deps`.
