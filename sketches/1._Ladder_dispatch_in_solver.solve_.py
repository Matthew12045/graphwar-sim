# solver.py — dispatch only, zero CCF logic

@dataclass(frozen=True)
class Rung:
    name: str
    fn: Callable[[Problem, Corridor], CCFResult]
    enabled: bool = True

LADDER: tuple[Rung, ...] = (
    Rung("direct",     solve_direct),
    Rung("ccf",        ccf.fit),        # ← new rung
    Rung("perturb",    solve_perturb),
    Rung("fallback",   solve_fallback),
)

def solve(problem: Problem, *, ladder: tuple[Rung, ...] = LADDER) -> Outcome:
    corridor = build_corridor(problem)

    # M5.1 pre-check runs FIRST. CCF must never see an unreachable corridor.
    if not corridor_is_reachable(corridor):
        return Outcome.unreachable(corridor)

    trace: list[tuple[str, CCFRejection]] = []
    for rung in ladder:
        if not rung.enabled:
            continue
        result = rung.fn(problem, corridor)
        if isinstance(result, CertifiedFit):
            return Outcome.solved(result, rung=rung.name, trace=trace)
        trace.append((rung.name, result))

    return Outcome.exhausted(trace=trace)