"""Regression tests for low-severity gene-space and PCA consistency fixes.

Covers:
  * The gene-space sign convention: the single-cell-type positivity flip only
    controls the objective sign for exactly 2 cell types; for >2 cell types the
    orientation / global sign is intentionally left undefined (no ineffective
    flip). See copro/gene_space.py.
  * Scale-only PCA (center=False, scale=True): the single-slide path now uses
    the same tiny-variance / very-sparse safeguard as the multi-slide path and
    utils.center_scale_matrix, so both agree. See copro/pca.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from copro.core import CoProMulti, CoProSingle, subset_data
from copro.gene_space import (
    compute_genespace_objective,
    optimize_genespace_avg_corr,
)
from copro.pca import compute_pca


# ---------------------------------------------------------------------------
# FIX 1 — gene-space sign convention
# ---------------------------------------------------------------------------

def _two_type_negative_problem():
    """One gene, one slide, a single strongly negative A-B pair."""
    slides = ["s1"]
    cell_types = ["A", "B"]
    self_cov = {"s1": {ct: np.eye(1) for ct in cell_types}}
    cross_cov = {"s1": {"A-B": np.array([[-0.9]])}}
    return self_cov, cross_cov, slides, cell_types


def _three_type_negative_bc_problem():
    """One gene, one slide; the B-C pair (not involving cell_types[0]) dominates
    negatively, while A is only weakly coupled to B and C.

    A single-type flip of cell_types[0] (= "A") can only negate the A-B and A-C
    terms, so it cannot control the sign of the dominant B-C term.
    """
    slides = ["s1"]
    cell_types = ["A", "B", "C"]
    self_cov = {"s1": {ct: np.eye(1) for ct in cell_types}}
    cross_cov = {
        "s1": {
            "A-B": np.array([[0.05]]),
            "A-C": np.array([[0.07]]),
            "B-C": np.array([[-0.9]]),
        }
    }
    return self_cov, cross_cov, slides, cell_types


def test_two_celltype_objective_is_non_negative_after_convention():
    """With exactly 2 cell types the single-type flip must restore a
    non-negative objective regardless of the random initialization."""
    self_cov, cross_cov, slides, cell_types = _two_type_negative_problem()
    for seed in range(20):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            weights = optimize_genespace_avg_corr(
                self_cov, cross_cov, slides, cell_types,
                max_iter=500, tol=1e-12, verbose=False, random_state=seed,
            )
        objective = compute_genespace_objective(
            weights, self_cov, cross_cov, slides, cell_types
        )
        assert objective >= -1e-12, (
            f"2-type objective must be non-negative, got {objective} (seed {seed})"
        )


def test_three_celltype_orientation_sign_is_undefined_not_falsely_flipped():
    """With >2 cell types the convention does NOT force a non-negative
    objective: a converged negative objective is returned as-is, because the
    single-type flip cannot fix a dominant pair that excludes cell_types[0]."""
    self_cov, cross_cov, slides, cell_types = _three_type_negative_bc_problem()

    objectives = []
    for seed in range(20):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            weights = optimize_genespace_avg_corr(
                self_cov, cross_cov, slides, cell_types,
                max_iter=500, tol=1e-12, verbose=False, random_state=seed,
            )
        objectives.append(
            compute_genespace_objective(
                weights, self_cov, cross_cov, slides, cell_types
            )
        )

    # Documented behavior: the global sign is undefined for >2 cell types, so
    # both signs occur across initializations and negatives are NOT flipped.
    assert any(o < 0 for o in objectives), (
        "expected at least one negative converged objective for >2 cell types; "
        "a false positivity flip would have removed them"
    )

    # For a negative case, confirm that R's single-type flip could not have
    # restored positivity anyway — which is exactly why it is skipped here.
    neg_seed = int(np.argmin(objectives))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        w_neg = optimize_genespace_avg_corr(
            self_cov, cross_cov, slides, cell_types,
            max_iter=500, tol=1e-12, verbose=False, random_state=neg_seed,
        )
    obj_neg = compute_genespace_objective(
        w_neg, self_cov, cross_cov, slides, cell_types
    )
    assert obj_neg < 0
    flipped = {ct: w_neg[ct].copy() for ct in cell_types}
    flipped[cell_types[0]] = -flipped[cell_types[0]]
    obj_flipped = compute_genespace_objective(
        flipped, self_cov, cross_cov, slides, cell_types
    )
    assert obj_flipped < 0, (
        "flipping cell_types[0] should not restore positivity when a "
        "non-cell_types[0] pair dominates negatively"
    )


# ---------------------------------------------------------------------------
# FIX 3 — scale-only PCA parity between single- and multi-slide paths
# ---------------------------------------------------------------------------

def _scale_only_pca_dataset(seed=3, n_cells=300, n_genes=8):
    """Dense data with one near-constant gene (tiny variance) and one
    very-sparse gene (few non-zeros), which exercise both parts of the
    (std < 1e-3) | (nonzero_proportion < 0.01) safeguard."""
    rng = np.random.default_rng(seed)
    X = np.abs(rng.normal(size=(n_cells, n_genes))) + 1.0
    # Near-constant gene: scale ~1e-4 (between 1e-10 and 1e-3), all non-zero.
    # It is centered on zero because scale-only PCA divides by the *uncentered*
    # root mean square — R's scale(center = FALSE, scale = TRUE) — so that is
    # the quantity the guard tests.
    #
    # The divisor and the guard come from two different places in R. The
    # uncentered-RMS formula is what R's dense scale() call uses; the guard is
    # taken from R's sparse .sparse_pca_parameters(), which applies it for every
    # scale=TRUE case. R's own dense scale-only branch has no guard and divides
    # an all-zero gene by zero, so we deliberately follow the sparse convention
    # on both Python paths.
    X[:, n_genes - 2] = 1e-4 * rng.normal(size=n_cells)
    # Very-sparse gene: scale >> 1e-3 but non-zero proportion < 0.01.
    X[:, n_genes - 1] = 0.0
    X[[10, 200], n_genes - 1] = 10.0
    return X


def test_scale_only_pca_matches_single_and_multi_slide():
    """Single-slide scale-only PCA must match the multi-slide path for
    tiny-variance / very-sparse genes (same guard, same sdev/scaling)."""
    X = _scale_only_pca_dataset()
    n_cells, n_genes = X.shape

    # Sanity: the two special genes actually trigger the two guard branches.
    # center=False, so the scale factor is the uncentered root mean square.
    rms = np.sqrt((X ** 2).sum(axis=0) / (n_cells - 1))
    sparse_nz = np.mean(X[:, n_genes - 1] != 0)
    assert 1e-10 < rms[n_genes - 2] < 1e-3
    assert rms[n_genes - 1] > 1e-3 and sparse_nz < 0.01

    loc = pd.DataFrame({
        "x": np.linspace(0, 1, n_cells),
        "y": np.linspace(1, 0, n_cells),
    })
    labels = np.array(["A"] * n_cells, dtype=object)

    single = CoProSingle(
        normalized_data=X.copy(), location_data=loc.copy(),
        meta_data=pd.DataFrame(index=np.arange(n_cells)),
        cell_types=labels.copy(),
    )
    single = subset_data(single, ["A"])
    single = compute_pca(single, n_pca=5, center=False, scale=True)

    multi = CoProMulti(
        normalized_data=X.copy(), location_data=loc.copy(),
        meta_data=pd.DataFrame({"slideID": ["s1"] * n_cells}),
        cell_types=labels.copy(),
    )
    multi = subset_data(multi, ["A"])
    # There is one slide here, so within-slide and pooled preprocessing are the
    # same computation; run both to pin that they agree with the single-slide
    # path and with each other.
    multi = compute_pca(multi, n_pca=5, center=False, scale=True)
    multi_pooled = compute_pca(
        subset_data(
            CoProMulti(
                normalized_data=X.copy(), location_data=loc.copy(),
                meta_data=pd.DataFrame({"slideID": ["s1"] * n_cells}),
                cell_types=labels.copy(),
            ),
            ["A"],
        ),
        n_pca=5, center=False, scale=True, center_per_slide=False,
    )

    single_sdev = single.pca_global["A"]["sdev"]
    multi_sdev = multi.pca_global["A"]["sdev"]

    # Both multi-slide modes store the applied scaling; the two guarded genes
    # must be left unscaled (factor 1.0) on every path.
    within_scales = multi.pca_global["A"]["slide_scales"][0]
    pooled_scales = multi_pooled.pca_global["A"]["col_stds"]
    for scales in (within_scales, pooled_scales):
        assert scales[n_genes - 2] == 1.0
        assert scales[n_genes - 1] == 1.0

    np.testing.assert_allclose(single_sdev, multi_sdev, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        multi_pooled.pca_global["A"]["sdev"], multi_sdev, rtol=1e-6, atol=1e-8
    )
