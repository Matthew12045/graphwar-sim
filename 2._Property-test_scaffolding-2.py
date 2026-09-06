# tests/test_ccf_properties.py
import numpy as np
from hypothesis import given, settings, strategies as st

from graphwar.ccf import fit, CertifiedFit, CCFRejection

# --- corridor strategy -------------------------------------------------

@st.composite
def corridors(draw, n_bands=st.integers(4, 24)):
    n = draw(n_bands)
    xs = np.linspace(0.0, draw(st.floats(1.0, 50.0)), n)
    lo = np.array(draw(st.lists(
        st.floats(-40, 40, allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n)))
    width = np.array(draw(st.lists(
        st.floats(0.01, 20.0, allow_nan=False),
        min_size=n, max_size=n)))
    return Corridor(xs=xs, lo=lo, hi=lo + width)

DENSE = 40  # resample factor vs. whatever CCF used internally

def _dense_offset_grid(corridor):
    """Offset by half-step so we never land on CCF's own sample points."""
    xs = corridor.xs
    fine = np.linspace(xs[0], xs[-1], len(xs) * DENSE)
    step = fine[1] - fine[0]
    return fine[:-1] + step / 2.0

# --- P1: the invariant that matters ------------------------------------

@given(corridors())
@settings(max_examples=500, deadline=None)
def test_certified_fit_never_escapes_corridor(corridor):
    result = fit(problem_stub(), corridor)
    if not isinstance(result, CertifiedFit):
        return  # rejection is always permissible

    x = _dense_offset_grid(corridor)
    y = evaluate(result.expr, x)
    lo, hi = corridor.interp_bounds(x)

    assert np.all(np.isfinite(y)), "certified fit produced non-finite values"
    assert np.all(y >= lo - TOL), f"escaped below at x={x[np.argmin(y - lo)]}"
    assert np.all(y <= hi + TOL), f"escaped above at x={x[np.argmax(y - hi)]}"