"""Regression tests for deflation consistency and the within-type refinement.

Covers two defects fixed in ``copro/optimization.py``:

DEFECT 1 -- deflation scheme for >2 cell types.  ``run_skr_cca`` maps
``scale_pcs=True`` to ``sdev2_dict=None`` (whitened PCs) and ``scale_pcs=False``
to a weighted metric.  Since ``scale_pcs`` is only a reparametrization
(``X~ = X diag(sdev)**-1``, ``w~ = diag(sdev) w``), the two settings must yield
the SAME canonical subspace on every axis.  This requires the projection
deflation used for >2 types to be applied under the weighted metric as well, so
both paths deflate the *same* operator.  The tests below assert this on all
axes for both the single-slide and multi-slide optimizers, and pin R's
projection output for the whitened path.

DEFECT 2 -- single-slide within-type refinement.  Maximizing ``w' Y w`` depends
only on the symmetric part of ``Y``.  With an asymmetric (e.g. row/col
normalized) self-kernel the refinement must iterate on ``(Y + Y')/2`` to reach
the true optimum, matching ``_initialize_next_component`` and the multi-slide
direct solve.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from copro.optimization import (
    _compute_Y_resi,
    optimize_bilinear,
    optimize_bilinear_multi_slides,
    optimize_bilinear_n,
    optimize_bilinear_n_multi_slides,
)


REFERENCE = Path(__file__).parent / "r_reference" / "modern_parity.csv"


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def _three_type_problem(seed=11, n_features=4):
    """Raw PC scores, per-PC sdev, whitened scores, and shared cross-kernels."""
    rng = np.random.default_rng(seed)
    cell_types = ["A", "B", "C"]
    n_cells = {"A": 30, "B": 26, "C": 34}
    sigma = 0.3

    X_raw = {ct: rng.normal(size=(n_cells[ct], n_features)) for ct in cell_types}
    # Distinct, strictly positive per-PC standard deviations per cell type.
    sdev = {
        ct: np.linspace(0.6, 2.1, n_features) + 0.15 * i
        for i, ct in enumerate(cell_types)
    }
    sdev2 = {ct: sdev[ct] ** 2 for ct in cell_types}
    X_white = {ct: X_raw[ct] / sdev[ct] for ct in cell_types}

    # Kernels live on cells, so they are identical for both parametrizations.
    kernels = {}
    for i, ci in enumerate(cell_types):
        for cj in cell_types[i + 1:]:
            kernels[f"kernel|sigma{sigma}|{ci}|{cj}"] = rng.normal(
                size=(n_cells[ci], n_cells[cj])
            )
    return cell_types, sigma, X_raw, X_white, sdev, sdev2, kernels


def _assert_same_subspace(cell_types, sdev, w_white, w_raw, n_cc):
    """|cos| == 1 on every axis between w~ (whitened) and diag(sdev) w (raw).

    ``w_raw`` is normalized under the weighted metric, so ``diag(sdev) w_raw``
    is the whitened-space representation of the same direction.  A single global
    sign per axis must align every cell type simultaneously.
    """
    for ax in range(n_cc):
        ref = cell_types[0]
        mapped_ref = sdev[ref] * w_raw[ref][:, ax]
        sign = np.sign(float(w_white[ref][:, ax] @ mapped_ref))
        assert sign != 0
        for ct in cell_types:
            wt = w_white[ct][:, ax]
            mapped = sdev[ct] * w_raw[ct][:, ax]
            cos = float(wt @ mapped) / (
                np.linalg.norm(wt) * np.linalg.norm(mapped)
            )
            assert abs(cos) > 1 - 1e-8, (
                f"axis {ax} {ct}: |cos|={abs(cos):.2e} (subspaces differ)"
            )
            np.testing.assert_allclose(
                wt, sign * mapped, atol=1e-6,
                err_msg=f"axis {ax} {ct}: sign-aligned weights differ",
            )


# ---------------------------------------------------------------------------
# DEFECT 1 -- cross-scale consistency (single slide)
# ---------------------------------------------------------------------------

def test_single_slide_deflation_same_subspace_across_scale_pcs():
    cell_types, sigma, X_raw, X_white, sdev, sdev2, kernels = (
        _three_type_problem()
    )
    n_cc = 3
    kw = dict(verbose=False, max_iter=5000, tol=1e-11)

    w1_white = optimize_bilinear(X_white, kernels, sigma, **kw)
    w_white = optimize_bilinear_n(
        X_white, kernels, sigma, w1_white, cell_types, n_cc=n_cc, **kw
    )

    w1_raw = optimize_bilinear(X_raw, kernels, sigma, sdev2_dict=sdev2, **kw)
    w_raw = optimize_bilinear_n(
        X_raw, kernels, sigma, w1_raw, cell_types, n_cc=n_cc,
        sdev2_dict=sdev2, **kw
    )

    for ct in cell_types:
        assert w_white[ct].shape[1] == n_cc
        assert w_raw[ct].shape[1] == n_cc
    _assert_same_subspace(cell_types, sdev, w_white, w_raw, n_cc)


# ---------------------------------------------------------------------------
# DEFECT 1 -- cross-scale consistency (multi-slide, twin deflation site)
# ---------------------------------------------------------------------------

def test_multi_slide_deflation_same_subspace_across_scale_pcs():
    cell_types, sigma, X_raw, X_white, sdev, sdev2, _ = _three_type_problem(
        seed=5
    )
    rng = np.random.default_rng(99)
    slides = ["s1", "s2"]
    n_cells = {ct: X_raw[ct].shape[0] for ct in cell_types}
    n_cc = 3

    # Two slides that reuse the same PC scores but carry different kernels.
    X_white_all = {s: {ct: X_white[ct] for ct in cell_types} for s in slides}
    X_raw_all = {s: {ct: X_raw[ct] for ct in cell_types} for s in slides}
    kernels = {}
    for s in slides:
        for i, ci in enumerate(cell_types):
            for cj in cell_types[i + 1:]:
                kernels[f"kernel|sigma{sigma}|{s}|{ci}|{cj}"] = rng.normal(
                    size=(n_cells[ci], n_cells[cj])
                )

    kw = dict(verbose=False, max_iter=5000, tol=1e-11)

    w1_white = optimize_bilinear_multi_slides(
        X_white_all, kernels, sigma, slides, **kw
    )
    w_white = optimize_bilinear_n_multi_slides(
        X_white_all, kernels, sigma, slides, w1_white, cell_types,
        n_cc=n_cc, **kw
    )

    w1_raw = optimize_bilinear_multi_slides(
        X_raw_all, kernels, sigma, slides, sdev2_dict=sdev2, **kw
    )
    w_raw = optimize_bilinear_n_multi_slides(
        X_raw_all, kernels, sigma, slides, w1_raw, cell_types,
        n_cc=n_cc, sdev2_dict=sdev2, **kw
    )

    _assert_same_subspace(cell_types, sdev, w_white, w_raw, n_cc)


# ---------------------------------------------------------------------------
# DEFECT 2 -- within-type refinement uses the symmetric part of Y
# ---------------------------------------------------------------------------

def test_within_type_asymmetric_self_kernel_uses_symmetric_optimum():
    rng = np.random.default_rng(3)
    n, p = 40, 5
    sigma = 0.3
    X = rng.normal(size=(n, p))

    # Realistic self-kernel: symmetric positive-definite part (so the top
    # eigenvector of the symmetric part is the true maximizer of w' Y w) plus a
    # strong antisymmetric perturbation that makes the raw self-kernel -- and
    # hence Y = X' K X -- asymmetric.
    C = rng.normal(size=(n, n))
    K_sym = C @ C.T / n + 0.5 * np.eye(n)
    K_anti = rng.normal(size=(n, n))
    K_anti = K_anti - K_anti.T
    K = K_sym + 0.8 * K_anti

    kernels = {f"kernel|sigma{sigma}|A|A": K}

    Y = _compute_Y_resi({"A": X}, kernels, sigma, ["A"])["A"]["A"]
    # The problem is genuinely asymmetric.
    assert np.max(np.abs(Y - Y.T)) > 1.0
    Y_sym = (Y + Y.T) / 2
    eigvals, eigvecs = np.linalg.eigh(Y_sym)
    top = eigvecs[:, -1]
    # Symmetric part is positive definite, so its top eigenvector is the
    # unconstrained maximizer of the Rayleigh quotient.
    assert eigvals[-1] > 0

    w = optimize_bilinear(
        {"A": X}, kernels, sigma, verbose=False, max_iter=5000, tol=1e-12
    )["A"][:, 0]

    cos = abs(float(w @ top) / (np.linalg.norm(w) * np.linalg.norm(top)))
    assert cos > 1 - 1e-8, f"refined weight is not the symmetric optimum: |cos|={cos:.3e}"

    # It maximizes w' Y w = w' Y_sym w (weight is unit-norm here).
    rayleigh = float(w @ Y_sym @ w) / float(w @ w)
    np.testing.assert_allclose(rayleigh, eigvals[-1], rtol=1e-8)

    # Guard that the fix matters: iterating on the raw asymmetric Y would
    # converge elsewhere, so the symmetric optimum is NOT the dominant
    # eigenvector of Y itself.
    raw_vals, raw_vecs = np.linalg.eig(Y)
    raw_dom = np.real(raw_vecs[:, int(np.argmax(np.abs(raw_vals)))])
    raw_cos = abs(
        float(raw_dom @ top) / (np.linalg.norm(raw_dom) * np.linalg.norm(top))
    )
    assert raw_cos < 0.99


# ---------------------------------------------------------------------------
# DEFECT 1 -- R parity for the >2-type projection (whitened / scale_pcs=True)
# ---------------------------------------------------------------------------

def _r_reference_weights():
    frame = pd.read_csv(REFERENCE)
    lookup = dict(zip(frame["metric"], frame["value"], strict=True))
    weights = {}
    for ct in ("A", "B", "C"):
        cols = []
        for ax in (1, 2):
            cols.append(
                np.array([lookup[f"mset_{ct}_cc{ax}_{k}"] for k in range(3)])
            )
        weights[ct] = np.column_stack(cols)
    return weights


# Inputs are hard-coded identically here and in
# tests/r_reference/generate_modern_parity.R so both languages see byte-for-byte
# the same problem.
_MSET_X = {
    "A": np.array([[0.5, -1.2, 0.3], [1.1, 0.4, -0.7],
                   [-0.9, 0.8, 1.0], [0.2, -0.5, 0.6]]),
    "B": np.array([[-0.3, 1.0, 0.5], [0.7, -0.6, 0.9],
                   [1.2, 0.1, -0.4], [-0.8, 0.5, 0.2]]),
    "C": np.array([[0.6, 0.2, -1.1], [-0.4, 0.9, 0.7],
                   [0.3, -0.8, 0.4], [1.0, 0.5, -0.2]]),
}
_MSET_KERNELS = {
    "kernel|sigma0.5|A|B": np.array([[1.0, 0.2, 0.1, 0.0], [0.1, 0.9, 0.3, 0.2],
                                     [0.0, 0.4, 1.1, 0.1], [0.3, 0.1, 0.2, 0.8]]),
    "kernel|sigma0.5|A|C": np.array([[0.9, 0.1, 0.2, 0.1], [0.2, 1.0, 0.1, 0.0],
                                     [0.1, 0.3, 0.8, 0.2], [0.0, 0.2, 0.1, 1.1]]),
    "kernel|sigma0.5|B|C": np.array([[1.1, 0.0, 0.1, 0.2], [0.1, 0.8, 0.2, 0.1],
                                     [0.3, 0.1, 1.0, 0.0], [0.2, 0.2, 0.1, 0.9]]),
}


def test_multiset_projection_matches_r_fixture():
    cell_types = ["A", "B", "C"]
    sigma = 0.5
    kw = dict(verbose=False, max_iter=20000, tol=1e-11)

    w1 = optimize_bilinear(_MSET_X, _MSET_KERNELS, sigma, **kw)
    fitted = optimize_bilinear_n(
        _MSET_X, _MSET_KERNELS, sigma, w1, cell_types, n_cc=2, **kw
    )
    expected = _r_reference_weights()

    for ax in range(2):
        ref = "A"
        sign = np.sign(float(fitted[ref][:, ax] @ expected[ref][:, ax]))
        assert sign != 0
        for ct in cell_types:
            np.testing.assert_allclose(
                fitted[ct][:, ax], sign * expected[ct][:, ax], atol=1e-6,
                err_msg=f"axis {ax} {ct}: Python does not match R projection",
            )
