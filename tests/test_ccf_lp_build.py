"""The vectorized LP row builder must produce the exact same matrix as the
per-entry reference builder (same CSR data/indices/indptr and rhs)."""

from __future__ import annotations

import numpy as np
import pytest

from graphwar_sim.ccf import _add_sample_rows, _add_sample_rows_reference, _LPRows


@pytest.mark.parametrize("seed", range(8))
def test_vectorized_sample_rows_match_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    K, J = int(rng.integers(1, 60)), int(rng.integers(1, 12))
    phi = rng.normal(size=(K, J))
    phi[rng.random((K, J)) < 0.5] = 0.0  # sparse like the truncated basis
    us = rng.uniform(0, 40, size=K)
    Lrel = rng.normal(size=K)
    Hrel = Lrel + rng.uniform(0, 5, size=K)
    exempt = rng.random(K) < 0.2
    n_var = 2 * J + K + 5
    args = (phi, us, Lrel, Hrel, exempt, 0.37, 2 * J, 2 * J + 1, 2 * J + 2, 2 * J + 3)

    ref, fast = _LPRows(n_var), _LPRows(n_var)
    for rows in (ref, fast):
        rows.add([(0, 1.0), (J, -1.0)], 0.0)  # a fixed row first, like _build_lp
    _add_sample_rows_reference(ref, *args)
    _add_sample_rows(fast, *args)

    a, b = ref.matrix(), fast.matrix()
    a.sort_indices()
    b.sort_indices()
    assert ref.b == fast.b
    assert a.shape == b.shape
    assert np.array_equal(a.indptr, b.indptr)
    assert np.array_equal(a.indices, b.indices)
    assert np.array_equal(a.data, b.data)
