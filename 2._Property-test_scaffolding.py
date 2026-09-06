# --- P2: metamorphic — widening must never lose a certificate ----------

@given(corridors(), st.floats(0.1, 5.0))
@settings(max_examples=300, deadline=None)
def test_widening_is_monotone(corridor, pad):
    tight = fit(problem_stub(), corridor)
    wide  = fit(problem_stub(), corridor.widened(pad))
    if isinstance(tight, CertifiedFit):
        assert isinstance(wide, CertifiedFit), (
            "widening a certifiable corridor caused rejection — "
            "search is unstable, not the corridor"
        )

# --- P3: determinism (stated solver guarantee) -------------------------

@given(corridors())
def test_bitwise_determinism(corridor):
    a = fit(problem_stub(seed=7), corridor)
    b = fit(problem_stub(seed=7), corridor)
    assert serialize(a) == serialize(b)

# --- P4: degenerate corridors must reject, never crash -----------------

@given(corridors())
def test_inverted_corridor_always_rejects(corridor):
    inverted = Corridor(xs=corridor.xs, lo=corridor.hi, hi=corridor.lo)
    assert fit(problem_stub(), inverted) is CCFRejection.CORRIDOR_TOO_TIGHT