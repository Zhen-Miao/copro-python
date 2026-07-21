"""Regression tests for the multi-slide design/response alignment in
``copro.test_gene_scores``.

Historically the CoProMulti branch of ``test_gene_scores`` built the design
matrix ``X`` (and covariates / LMM group) in *whole-dataset* cell order while
building the response ``cell_score`` by concatenating per-slide cell-score
arrays in ``obj.slide_list`` order.  Those two orderings agree only when each
cell type's cells happen to be contiguous-by-slide in the original ordering.
For interleaved multi-slide input the rows of ``X`` were silently permuted
relative to ``cell_score``, so every gene was regressed against a scrambled
response.

These tests build a CoProMulti object whose cells are *interleaved* across two
slides (original order is not contiguous-by-slide) and check that:

1. the GLM per-gene estimates / p-values match a hand-aligned OLS built in the
   corrected per-slide order (proving X and cell_score align row-for-row);
2. a gene that is genuinely associated with a CC axis is recovered as
   significant (the misaligned/buggy ordering destroys the association);
3. results are invariant to how the cells are interleaved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.api import OLS, add_constant

import copro as cp
from copro.core import CoProMulti

SIGMA = 0.3
N_CC = 2
N_PCA = 5
N_PER = 25          # cells per (slide, cell-type) block
SLIDES = ["s1", "s2"]
TYPES = ["A", "B"]
LOADINGS = np.array([0.50, -0.35, 0.25, 0.15, -0.10, 0.08])
N_GENES = len(LOADINGS)


# --------------------------------------------------------------------------- #
# Data construction
# --------------------------------------------------------------------------- #

def _make_cells(seed: int):
    """Build a flat list of per-cell records for a 2-slide, 2-type dataset."""
    rng = np.random.default_rng(seed)
    cells = []
    for slide_index, slide in enumerate(SLIDES):
        for type_index, ct in enumerate(TYPES):
            position = np.linspace(0.04, 0.96, N_PER)
            common = np.sin(2 * np.pi * position)
            noise = rng.normal(scale=0.10, size=(N_PER, N_GENES))
            block = np.exp(
                1.0
                + 0.08 * slide_index
                + common[:, None] * LOADINGS[None, :]
                + noise
            )
            for k in range(N_PER):
                cells.append(
                    {
                        "expr": block[k],
                        "x": float(position[k] + 0.012 * type_index),
                        "y": float(0.2 * np.cos(2 * np.pi * position[k])
                                   + 0.03 * type_index),
                        "slideID": slide,
                        "ct": ct,
                    }
                )
    return cells


def _assemble(cells, order):
    """Assemble a CoProMulti object with rows placed in the given order."""
    expr = np.vstack([cells[i]["expr"] for i in order])
    location = pd.DataFrame(
        {
            "x": [cells[i]["x"] for i in order],
            "y": [cells[i]["y"] for i in order],
        }
    )
    meta = pd.DataFrame({"slideID": [cells[i]["slideID"] for i in order]})
    cell_types = np.array([cells[i]["ct"] for i in order], dtype=object)

    obj = CoProMulti(
        normalized_data=expr,
        location_data=location,
        meta_data=meta,
        cell_types=cell_types,
        slide_list=list(SLIDES),
    )
    obj.gene_names = [f"g{i}" for i in range(N_GENES)]
    return obj


def _run_pipeline(obj):
    obj = cp.subset_data(obj, list(TYPES))
    obj = cp.compute_pca(obj, n_pca=N_PCA)
    obj = cp.compute_distance(obj, normalize=True, truncate=True)
    obj = cp.compute_kernel_matrix(
        obj, [SIGMA], method="dense", min_ave_cell_neighbor=1
    )
    obj = cp.run_skr_cca(obj, scale_pcs=True, n_cc=N_CC, max_iter=500, verbose=False)
    obj = cp.compute_gene_and_cell_scores(obj)
    return obj


def _interleaved_order(cells, seed=0):
    """A random permutation — guarantees cells are NOT contiguous-by-slide."""
    return np.random.default_rng(seed).permutation(len(cells))


# --------------------------------------------------------------------------- #
# Ordering helpers used to build the *ground-truth* (correctly aligned) design
# --------------------------------------------------------------------------- #

def _cell_score_concat(obj, ct, cc):
    """Response: per-slide cell scores concatenated in slide_list order."""
    parts = []
    for slide in obj.slide_list:
        key = f"cellScores|sigma{SIGMA}|{slide}|{ct}"
        if key in obj.cell_scores:
            parts.append(obj.cell_scores[key][:, cc])
    return np.concatenate(parts)


def _correct_X(obj, ct):
    """Design assembled per-slide in slide_list order (aligns with response)."""
    slide_ids = obj.meta_data_sub["slideID"].values
    parts = []
    for slide in obj.slide_list:
        key = f"cellScores|sigma{SIGMA}|{slide}|{ct}"
        if key not in obj.cell_scores:
            continue
        mask_s = (obj.cell_types_sub == ct) & (slide_ids == slide)
        parts.append(np.asarray(obj.normalized_data_sub[mask_s], dtype=float))
    return np.vstack(parts)


def _buggy_X(obj, ct):
    """Design assembled in whole-dataset cell order (the historical bug)."""
    mask = obj.cell_types_sub == ct
    return np.asarray(obj.normalized_data_sub[mask], dtype=float)


def _ols_per_gene(X, y):
    """Mirror ``_test_gene_glm`` (no covariates): estimate + p-value per gene."""
    y = np.asarray(y, dtype=float)
    n_genes = X.shape[1]
    est = np.full(n_genes, np.nan)
    pval = np.full(n_genes, np.nan)
    for g in range(n_genes):
        Xg = add_constant(X[:, g].astype(float).reshape(-1, 1))
        result = OLS(y, Xg).fit()
        est[g] = result.params[1]
        pval[g] = result.pvalues[1]
    return est, pval


def _per_cell_score_vector(obj, ct, cc):
    """Cell scores placed at each cell's *whole-subset* row position."""
    slide_ids = obj.meta_data_sub["slideID"].values
    n = obj.normalized_data_sub.shape[0]
    out = np.full(n, np.nan)
    for slide in obj.slide_list:
        key = f"cellScores|sigma{SIGMA}|{slide}|{ct}"
        if key not in obj.cell_scores:
            continue
        mask_s = (obj.cell_types_sub == ct) & (slide_ids == slide)
        out[mask_s] = obj.cell_scores[key][:, cc]
    return out


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def interleaved_obj():
    cells = _make_cells(seed=17)
    obj = _assemble(cells, _interleaved_order(cells, seed=3))
    return _run_pipeline(obj)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_input_is_actually_interleaved(interleaved_obj):
    """Sanity: for at least one cell type the slide labels are not
    contiguous-by-slide in the subset order (otherwise the bug is masked)."""
    slide_ids = interleaved_obj.meta_data_sub["slideID"].values
    changed = False
    for ct in TYPES:
        seq = slide_ids[interleaved_obj.cell_types_sub == ct]
        # number of slide-label transitions; >1 means interleaved
        transitions = int(np.sum(seq[1:] != seq[:-1]))
        if transitions > 1:
            changed = True
    assert changed, "Test data is contiguous-by-slide; bug would be masked."


def test_glm_matches_hand_aligned_ols(interleaved_obj):
    """Function GLM output must equal a hand-aligned OLS built in the corrected
    per-slide order, for every cell type and every CC."""
    obj = interleaved_obj
    result = cp.test_gene_scores(
        obj, sigma=SIGMA, test_type="glm",
        gene_names=obj.gene_names, verbose=False,
    )
    for ct in TYPES:
        for cc in range(N_CC):
            cc_name = f"CC_{cc + 1}"
            df = result.gene_score_test[ct][cc_name]

            X = _correct_X(obj, ct)
            y = _cell_score_concat(obj, ct, cc)
            assert X.shape[0] == y.shape[0], "internal: correct design misaligned"

            est, pval = _ols_per_gene(X, y)
            np.testing.assert_allclose(
                df["estimate"].to_numpy(), est, rtol=1e-6, atol=1e-9,
                err_msg=f"estimate mismatch for {ct} {cc_name}",
            )
            np.testing.assert_allclose(
                df["pvalue"].to_numpy(), pval, rtol=1e-6, atol=1e-12,
                err_msg=f"pvalue mismatch for {ct} {cc_name}",
            )


def test_associated_gene_recovered_as_significant(interleaved_obj):
    """A gene built to equal the (correctly aligned) CC1 cell score must come
    out highly significant with the fix — and would be non-significant under
    the historical whole-order (buggy) alignment."""
    obj = interleaved_obj

    # Build a synthetic gene equal to each cell's CC1 score (+ tiny noise),
    # placed at the cell's whole-subset row position, then append it as a new
    # gene column.  Because it is stored in whole-subset order, the design must
    # be re-assembled per-slide (the fix) for the association to survive.
    rng = np.random.default_rng(123)
    n = obj.normalized_data_sub.shape[0]
    assoc = np.zeros(n)
    for ct in TYPES:
        pcs = _per_cell_score_vector(obj, ct, cc=0)
        m = ~np.isnan(pcs)
        assoc[m] = pcs[m]
    assoc = assoc + rng.normal(scale=1e-3, size=n)

    obj.normalized_data_sub = np.hstack(
        [np.asarray(obj.normalized_data_sub, dtype=float), assoc[:, None]]
    )
    gene_names = list(obj.gene_names) + ["assoc"]
    g_assoc = len(gene_names) - 1

    result = cp.test_gene_scores(
        obj, sigma=SIGMA, test_type="glm",
        gene_names=gene_names, verbose=False,
    )

    for ct in TYPES:
        df = result.gene_score_test[ct]["CC_1"]
        # Function == correctly-aligned OLS for the synthetic gene.
        X_corr = _correct_X(obj, ct)
        y = _cell_score_concat(obj, ct, cc=0)
        est_corr, p_corr = _ols_per_gene(X_corr[:, [g_assoc]], y)

        func_est = df.loc[df["gene"] == "assoc", "estimate"].to_numpy()[0]
        func_p = df.loc[df["gene"] == "assoc", "pvalue"].to_numpy()[0]
        assert np.isclose(func_est, est_corr[0], rtol=1e-6, atol=1e-9)
        assert np.isclose(func_p, p_corr[0], rtol=1e-6, atol=1e-12)

        # Correct alignment: slope ~= 1 and strongly significant.
        assert abs(est_corr[0] - 1.0) < 0.05, f"{ct}: slope {est_corr[0]} != ~1"
        assert p_corr[0] < 1e-6, f"{ct}: correct p={p_corr[0]} not significant"

        # Buggy (whole-order) alignment: the association is destroyed.
        X_bug = _buggy_X(obj, ct)
        est_bug, p_bug = _ols_per_gene(X_bug[:, [g_assoc]], y)
        assert not np.isclose(est_bug[0], est_corr[0], rtol=1e-2), (
            f"{ct}: buggy and correct estimates coincide — interleaving too weak"
        )
        assert p_bug[0] > 0.05, (
            f"{ct}: buggy p={p_bug[0]} unexpectedly significant "
            f"(interleaving did not scramble the response)"
        )


def test_results_invariant_to_slide_interleaving():
    """Two different row-interleavings of the *same* cells must yield identical
    GLM p-values.  Built by re-indexing an already-fitted object so the check
    isolates the design/response assembly from CCA nondeterminism."""
    cells = _make_cells(seed=42)
    obj = _run_pipeline(_assemble(cells, _interleaved_order(cells, seed=1)))

    perm = np.random.default_rng(7).permutation(
        obj.normalized_data_sub.shape[0]
    )
    reindexed = _reindex_subset(obj, perm)

    res_a = cp.test_gene_scores(
        obj, sigma=SIGMA, test_type="glm",
        gene_names=obj.gene_names, verbose=False,
    )
    res_b = cp.test_gene_scores(
        reindexed, sigma=SIGMA, test_type="glm",
        gene_names=reindexed.gene_names, verbose=False,
    )

    for ct in TYPES:
        for cc in range(N_CC):
            cc_name = f"CC_{cc + 1}"
            pa = res_a.gene_score_test[ct][cc_name]["pvalue"].to_numpy()
            pb = res_b.gene_score_test[ct][cc_name]["pvalue"].to_numpy()
            np.testing.assert_allclose(
                pa, pb, rtol=1e-8, atol=1e-12,
                err_msg=f"p-values differ across interleavings for {ct} {cc_name}",
            )


# --------------------------------------------------------------------------- #
# Re-indexing helper for the invariance test
# --------------------------------------------------------------------------- #

def _reindex_subset(obj, perm):
    """Return a copy of *obj* with all subset rows permuted by *perm*, with the
    per-slide cell scores re-split to match the new within-slide ordering.  The
    underlying cells are unchanged, so gene-test results must be identical."""
    import copy

    new = copy.copy(obj)
    new.normalized_data_sub = np.asarray(obj.normalized_data_sub)[perm]
    new.cell_types_sub = np.asarray(obj.cell_types_sub)[perm]
    new.meta_data_sub = obj.meta_data_sub.iloc[perm].reset_index(drop=True)
    new.cell_scores = dict(obj.cell_scores)
    new.gene_names = list(obj.gene_names)

    slide_ids_old = obj.meta_data_sub["slideID"].values
    new_slide_ids = new.meta_data_sub["slideID"].values
    n = obj.normalized_data_sub.shape[0]

    for ct in obj.cell_types_of_interest:
        # Reconstruct the full per-cell score matrix in OLD subset order.
        n_cc = None
        for slide in obj.slide_list:
            key = f"cellScores|sigma{SIGMA}|{slide}|{ct}"
            if key in obj.cell_scores:
                n_cc = obj.cell_scores[key].shape[1]
                break
        if n_cc is None:
            continue
        full = np.full((n, n_cc), np.nan)
        for slide in obj.slide_list:
            key = f"cellScores|sigma{SIGMA}|{slide}|{ct}"
            if key not in obj.cell_scores:
                continue
            mask_old = (obj.cell_types_sub == ct) & (slide_ids_old == slide)
            full[mask_old] = obj.cell_scores[key]

        full_new = full[perm]
        for slide in obj.slide_list:
            key = f"cellScores|sigma{SIGMA}|{slide}|{ct}"
            if key not in obj.cell_scores:
                continue
            mask_new = (new.cell_types_sub == ct) & (new_slide_ids == slide)
            new.cell_scores[key] = full_new[mask_new]

    return new
