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
from scipy.sparse._compressed import _cs_matrix

from copro.core import CoProMulti, CoProSingle, subset_data
from copro.pca import compute_pca


N_CELLS, N_GENES, N_PCA = 240, 20, 5

# compute_pca() accepts any of these unchanged (_as_svd_ready_sparse only
# converts formats outside csr/csc), so each needs its own coverage.
SPARSE_FORMATS = (sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array)


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


@pytest.mark.parametrize("fmt", SPARSE_FORMATS)
@pytest.mark.parametrize("center,scale", [(True, True), (True, False), (False, True), (False, False)])
def test_single_slide_sparse_matches_dense(center, scale, fmt):
    X = _counts()
    dense = compute_pca(_single(X.copy()), n_pca=N_PCA, center=center, scale=scale)
    spars = compute_pca(_single(fmt(X)), n_pca=N_PCA, center=center, scale=scale)
    for ct in ("A", "B"):
        _assert_pca_equal(dense.pca_global[ct], spars.pca_global[ct])


# Standardization arrays stored per preprocessing mode, and the tolerance used
# to compare them. The dense path computes the variance two-pass and the sparse
# path one-pass (mirroring R), so these agree only as well as that formula pair
# does: ~1e-14 relative here because _counts() keeps gene means below ~1, but
# the gap grows as (mean/sd)^2 and would exceed rtol=1e-9 for a gene with, say,
# mean 1e3 and sd 1e-2. Keep the fixture's means small.
_STANDARDIZATION_KEYS = {
    "within_slide": ("slide_centers", "slide_scales"),
    "pooled": ("col_means", "col_stds"),
}


@pytest.mark.parametrize("center_per_slide", [True, False])
@pytest.mark.parametrize("center,scale", [(True, True), (True, False), (False, True), (False, False)])
def test_multi_slide_sparse_matches_dense(center, scale, center_per_slide):
    X = _counts()
    kwargs = dict(n_pca=N_PCA, center=center, scale=scale, center_per_slide=center_per_slide)
    dense = compute_pca(_multi(X.copy()), **kwargs)
    spars = compute_pca(_multi(sparse.csr_matrix(X)), **kwargs)
    mode = "within_slide" if center_per_slide else "pooled"
    for ct in ("A", "B"):
        d, s = dense.pca_global[ct], spars.pca_global[ct]
        assert d["preprocessing"] == s["preprocessing"] == mode
        _assert_pca_equal(d, s)
        for key in _STANDARDIZATION_KEYS[mode]:
            np.testing.assert_allclose(d[key], s[key], rtol=1e-9, atol=1e-12)
        signs = np.sign(np.sum(d["components"] * s["components"], axis=1))
        for slide in dense.slide_list:
            np.testing.assert_allclose(
                dense.pca_results[slide][ct],
                signs[None, :] * spars.pca_results[slide][ct],
                rtol=1e-7, atol=1e-9,
            )


@pytest.mark.parametrize("center,scale", [(True, True), (False, True)])
def test_within_slide_blocks_are_standardized_per_slide(center, scale):
    """Each slide's block must be centered/scaled against its own statistics."""
    X = _counts()
    obj = compute_pca(_multi(sparse.csr_matrix(X)), n_pca=N_PCA,
                      center=center, scale=scale, center_per_slide=True)
    for ct in ("A", "B"):
        pca = obj.pca_global[ct]
        centers, scales = pca["slide_centers"], pca["slide_scales"]
        assert centers.shape == scales.shape == (len(obj.slide_list), X.shape[1])
        if not center:
            assert np.all(centers == 0.0)
        # A gene guarded on any one slide is guarded on all of them.
        guarded = (scales == 1.0).any(axis=0)
        assert np.array_equal(scales[:, guarded], np.ones_like(scales[:, guarded]))
    # Per-slide scores are rows of the global score matrix, so stacking the
    # per-slide blocks back together must reproduce it exactly.
    for ct in ("A", "B"):
        stacked = np.vstack([obj.pca_results[s][ct] for s in obj.slide_list])
        assert stacked.shape == obj.pca_global[ct]["scores"].shape


def test_within_slide_is_the_default():
    X = _counts()
    obj = compute_pca(_multi(sparse.csr_matrix(X)), n_pca=N_PCA)
    assert obj.pca_global["A"]["preprocessing"] == "within_slide"


def test_within_slide_removes_a_pure_slide_offset():
    """A constant per-slide shift is batch effect, not signal: within-slide
    preprocessing must absorb it, while the pooled path does not."""
    X = _counts()
    obj_plain = _multi(X.copy())
    shifted = X.copy()
    # slide "s2" is every odd row (see _multi); add a fixed offset to it.
    shifted[1::2, 2:] += 3.0

    within = compute_pca(_multi(shifted.copy()), n_pca=N_PCA, center_per_slide=True)
    pooled = compute_pca(_multi(shifted.copy()), n_pca=N_PCA, center_per_slide=False)
    baseline = compute_pca(obj_plain, n_pca=N_PCA, center_per_slide=True)

    for ct in ("A", "B"):
        # Within-slide standardization is invariant to the offset.
        np.testing.assert_allclose(
            within.pca_global[ct]["sdev"], baseline.pca_global[ct]["sdev"],
            rtol=1e-6, atol=1e-8,
        )
        # The pooled path lets the offset dominate the leading component.
        assert pooled.pca_global[ct]["sdev"][0] > 2 * within.pca_global[ct]["sdev"][0]


@pytest.mark.parametrize("fmt", SPARSE_FORMATS)
def test_sparse_input_is_never_densified(monkeypatch, fmt):
    """Guard against a regression that reintroduces a dense cells-by-genes copy.

    Patch ``_cs_matrix``, not ``csr_matrix``: ``toarray``/``todense`` are defined
    on that shared base, so patching the leaf class shadows them for csr_matrix
    only and leaves csc_matrix / csr_array able to densify undetected.
    """
    X = fmt(_counts())

    def _fail(self, *args, **kwargs):
        raise AssertionError("sparse PCA input was densified")

    monkeypatch.setattr(_cs_matrix, "toarray", _fail, raising=True)
    monkeypatch.setattr(_cs_matrix, "todense", _fail, raising=True)
    compute_pca(_single(X), n_pca=N_PCA)
    compute_pca(_multi(X), n_pca=N_PCA)


def test_densification_guard_covers_every_supported_format():
    """The guard above is only meaningful if it actually trips on each format."""
    def _fail(self, *args, **kwargs):
        raise AssertionError("densified")

    original = _cs_matrix.toarray
    _cs_matrix.toarray = _fail
    try:
        for fmt in SPARSE_FORMATS:
            with pytest.raises(AssertionError, match="densified"):
                fmt(_counts()).toarray()
    finally:
        _cs_matrix.toarray = original


@pytest.mark.parametrize("center_per_slide", [True, False])
@pytest.mark.parametrize("as_sparse", [True, False])
def test_guarded_genes_are_left_unscaled(center_per_slide, as_sparse):
    """Zero-variance and very-sparse genes keep a scale factor of 1.

    Matches R's .sparse_pca_parameters() / .withinSlidePCAParameters(), which
    apply this guard on both the sparse and the dense path.
    """
    X = _counts()
    obj = compute_pca(_multi(sparse.csr_matrix(X) if as_sparse else X),
                      n_pca=N_PCA, center_per_slide=center_per_slide)
    for ct in ("A", "B"):
        pca = obj.pca_global[ct]
        scales = pca["slide_scales"] if center_per_slide else pca["col_stds"][None, :]
        assert np.all(scales[:, 0] == 1.0)   # all-zero gene
        assert np.all(scales[:, 1] == 1.0)   # non-zero proportion < 1%
        assert np.all(scales > 0)            # nothing is ever divided by zero
