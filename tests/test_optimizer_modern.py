"""Regression tests for the exact-SVD and projection-deflation optimizer."""

import numpy as np

from copro.optimization import (
    _apply_deflation,
    _compute_Y_resi,
    _solve_two_type_svd,
    optimize_bilinear,
    optimize_bilinear_n,
)


def _two_type_problem(seed=1):
    rng = np.random.default_rng(seed)
    X = {"A": rng.normal(size=(28, 6)), "B": rng.normal(size=(31, 6))}
    K = rng.normal(size=(28, 31))
    kernels = {"kernel|sigma0.2|A|B": K}
    return X, kernels


def test_two_type_all_axes_equal_one_exact_svd():
    X, kernels = _two_type_problem()
    first = optimize_bilinear(X, kernels, 0.2, verbose=False)
    fitted = optimize_bilinear_n(
        X, kernels, 0.2, first, ["A", "B"], n_cc=4, verbose=False
    )
    expected = _solve_two_type_svd(
        _compute_Y_resi(X, kernels, 0.2, ["A", "B"]), ["A", "B"], 4
    )
    for ct in ("A", "B"):
        np.testing.assert_allclose(
            np.abs(fitted[ct]), np.abs(expected[ct]), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            fitted[ct].T @ fitted[ct], np.eye(4), rtol=1e-12, atol=1e-12
        )


def test_weighted_two_type_svd_satisfies_cca_metric():
    X, kernels = _two_type_problem()
    metric = {
        "A": np.linspace(0.5, 2.0, 6) ** 2,
        "B": np.linspace(0.8, 1.8, 6) ** 2,
    }
    Y = _compute_Y_resi(X, kernels, 0.2, ["A", "B"])
    fitted = _solve_two_type_svd(Y, ["A", "B"], 4, metric)
    for ct in ("A", "B"):
        gram = fitted[ct].T @ (metric[ct][:, None] * fitted[ct])
        np.testing.assert_allclose(gram, np.eye(4), rtol=1e-12, atol=1e-12)


def test_projection_deflation_removes_both_selected_directions():
    rng = np.random.default_rng(7)
    cell_types = ["A", "B", "C"]
    Y = {ct: {} for ct in cell_types}
    weights = {}
    for ct in cell_types:
        w = rng.normal(size=(5, 1))
        weights[ct] = w / np.linalg.norm(w)
    for i, ct_i in enumerate(cell_types):
        for ct_j in cell_types[i + 1 :]:
            block = rng.normal(size=(5, 5))
            Y[ct_i][ct_j] = block
            Y[ct_j][ct_i] = block.T

    residual = _apply_deflation(
        Y, weights, 0, cell_types, method="projection"
    )
    for i, ct_i in enumerate(cell_types):
        for ct_j in cell_types[i + 1 :]:
            block = residual[ct_i][ct_j]
            np.testing.assert_allclose(weights[ct_i].T @ block, 0, atol=1e-12)
            np.testing.assert_allclose(block @ weights[ct_j], 0, atol=1e-12)
