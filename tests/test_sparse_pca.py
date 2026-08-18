"""compute_pca() on sparse input must equal the dense path exactly.

Sparse blocks are standardized implicitly through a LinearOperator (the same
trick R's IRLBA uses with its center/scale. arguments), so the cells-by-genes
block is never densified. The decomposition is of the same matrix, so results
must agree to solver tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from copro.core import CoProMulti, CoProSingle, subset_data
from copro.pca import compute_pca


N_CELLS, N_GENES, N_PCA = 240, 20, 5


def _counts(seed: int = 0):
    """Sparse log-normalized-looking counts with the two guarded gene types."""
    rng = np.random.default_rng(seed)
    X = np.log1p(rng.poisson(0.4, size=(N_CELLS, N_GENES)).astype(float))
    X[:, 0] = 0.0                      # all-zero gene (zero variance guard)
    X[:, 1] = 0.0
    # Very-sparse gene: one non-zero per cell-type block (< 1% of that block),
    # so the non-zero-proportion guard fires for both cell types.
    X[[3, N_CELLS // 2 + 3], 1] = 4.0
    return X


def _locations():
    rng = np.random.default_rng(1)
    return pd.DataFrame({"x": rng.random(N_CELLS), "y": rng.random(N_CELLS)})


def _labels():
    labels = np.array(["A"] * N_CELLS, dtype=object)
    labels[N_CELLS // 2:] = "B"
    return labels


def _single(X):
    obj = CoProSingle(
        normalized_data=X,
        location_data=_locations(),
        meta_data=pd.DataFrame(index=np.arange(N_CELLS)),
        cell_types=_labels(),
    )
    return subset_data(obj, ["A", "B"])


def _multi(X):
    slide = np.where(np.arange(N_CELLS) % 2 == 0, "s1", "s2")
    obj = CoProMulti(
        normalized_data=X,
        location_data=_locations(),
        meta_data=pd.DataFrame({"slideID": slide}),
        cell_types=_labels(),
    )
    return subset_data(obj, ["A", "B"])


def _assert_pca_equal(dense_pca, sparse_pca):
    np.testing.assert_allclose(dense_pca["sdev"], sparse_pca["sdev"], rtol=1e-9, atol=1e-10)
    # An SVD is only defined up to sign, but svd_flip() pins the convention, so
    # the two paths must also agree on orientation — downstream CCA scores would
    # otherwise flip when the input happens to be stored sparse.
    signs = np.sign(np.sum(dense_pca["components"] * sparse_pca["components"], axis=1))
    assert np.all(signs > 0), "sparse path flipped a component relative to dense"
    np.testing.assert_allclose(
        dense_pca["components"], signs[:, None] * sparse_pca["components"],
        rtol=1e-7, atol=1e-9,
    )
    np.testing.assert_allclose(
        dense_pca["scores"], signs[None, :] * sparse_pca["scores"],
        rtol=1e-7, atol=1e-9,
    )


@pytest.mark.parametrize("center,scale", [(True, True), (True, False), (False, True), (False, False)])
def test_single_slide_sparse_matches_dense(center, scale):
    X = _counts()
    dense = compute_pca(_single(X.copy()), n_pca=N_PCA, center=center, scale=scale)
    spars = compute_pca(_single(sparse.csr_matrix(X)), n_pca=N_PCA, center=center, scale=scale)
    for ct in ("A", "B"):
        _assert_pca_equal(dense.pca_global[ct], spars.pca_global[ct])


@pytest.mark.parametrize("center,scale", [(True, True), (True, False), (False, True), (False, False)])
def test_multi_slide_sparse_matches_dense(center, scale):
    X = _counts()
    dense = compute_pca(_multi(X.copy()), n_pca=N_PCA, center=center, scale=scale)
    spars = compute_pca(_multi(sparse.csr_matrix(X)), n_pca=N_PCA, center=center, scale=scale)
    for ct in ("A", "B"):
        d, s = dense.pca_global[ct], spars.pca_global[ct]
        _assert_pca_equal(d, s)
        np.testing.assert_allclose(d["col_means"], s["col_means"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(d["col_stds"], s["col_stds"], rtol=1e-9, atol=1e-12)
        signs = np.sign(np.sum(d["components"] * s["components"], axis=1))
        for slide in dense.slide_list:
            np.testing.assert_allclose(
                dense.pca_results[slide][ct],
                signs[None, :] * spars.pca_results[slide][ct],
                rtol=1e-7, atol=1e-9,
            )


def test_sparse_input_is_never_densified(monkeypatch):
    """Guard against a regression that reintroduces a dense cells-by-genes copy."""
    X = sparse.csr_matrix(_counts())

    def _fail(self, *args, **kwargs):
        raise AssertionError("sparse PCA input was densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", _fail, raising=True)
    monkeypatch.setattr(sparse.csr_matrix, "todense", _fail, raising=True)
    compute_pca(_single(X), n_pca=N_PCA)
    compute_pca(_multi(X), n_pca=N_PCA)


def test_guarded_genes_are_left_unscaled():
    """Zero-variance and very-sparse genes keep a scale factor of 1 on the
    sparse path, matching R's .sparse_pca_parameters()."""
    X = _counts()
    obj = compute_pca(_multi(sparse.csr_matrix(X)), n_pca=N_PCA)
    col_stds = obj.pca_global["A"]["col_stds"]
    assert col_stds[0] == 1.0   # all-zero gene
    assert col_stds[1] == 1.0   # non-zero proportion < 1%
