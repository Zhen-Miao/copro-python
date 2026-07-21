"""Tests for the centered whitened-Frobenius correlation normalizer."""

import numpy as np
from scipy import sparse

from copro.correlation import _whitened_frob_norm


def _explicit_norm(K, Rx=None, Ry=None):
    Kc = K - K.mean(axis=1, keepdims=True) - K.mean(axis=0, keepdims=True) + K.mean()
    if Rx is None or Ry is None:
        return np.linalg.norm(Kc, ord="fro")
    Rx = (Rx + Rx.T) / 2
    Ry = (Ry + Ry.T) / 2
    return np.sqrt(max(np.sum(((Rx @ Kc) @ Ry) * Kc), 0.0))


def test_dense_whitened_frobenius_matches_trace_identity():
    rng = np.random.default_rng(22)
    K = rng.uniform(size=(9, 7))
    A = rng.normal(size=(9, 9))
    B = rng.normal(size=(7, 7))
    Rx = A @ A.T
    Ry = B @ B.T
    np.testing.assert_allclose(
        _whitened_frob_norm(K, Rx, Ry),
        _explicit_norm(K, Rx, Ry),
        rtol=1e-12,
        atol=1e-12,
    )


def test_sparse_low_rank_centering_matches_dense_result():
    rng = np.random.default_rng(23)
    K = rng.uniform(size=(12, 10))
    K[K < 0.72] = 0
    Rx = rng.uniform(size=(12, 12))
    Rx[Rx < 0.75] = 0
    Rx = Rx @ Rx.T
    Ry = rng.uniform(size=(10, 10))
    Ry[Ry < 0.75] = 0
    Ry = Ry @ Ry.T
    expected = _explicit_norm(K, Rx, Ry)
    actual = _whitened_frob_norm(
        sparse.csr_matrix(K), sparse.csr_matrix(Rx), sparse.csr_matrix(Ry)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_unwhitened_fallback_uses_centered_frobenius_norm():
    rng = np.random.default_rng(24)
    K = rng.normal(size=(8, 6))
    expected = _explicit_norm(K)
    np.testing.assert_allclose(_whitened_frob_norm(K), expected, rtol=1e-12)
    np.testing.assert_allclose(
        _whitened_frob_norm(sparse.csr_matrix(K)), expected, rtol=1e-12
    )
