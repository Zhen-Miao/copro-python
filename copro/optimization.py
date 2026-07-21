"""Exact two-type and iterative multi-set skrCCA optimization."""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
from scipy.sparse.linalg import svds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_vec(v: np.ndarray) -> np.ndarray:
    """Normalize to unit length → (n, 1) column vector."""
    norm = float(np.sqrt(np.sum(v ** 2)))
    if norm < 1e-12:
        warnings.warn("Near-zero vector encountered during normalization.")
        return np.zeros((v.size, 1), dtype=float)
    return (v / norm).reshape(-1, 1)


def _normalize_vec_weighted(v: np.ndarray, sdev2: np.ndarray | None = None) -> np.ndarray:
    """Normalize a vector under ``w' diag(sdev2) w = 1``."""
    if sdev2 is None:
        return _normalize_vec(v)
    values = np.asarray(v, dtype=float).ravel()
    metric = np.asarray(sdev2, dtype=float).ravel()
    if values.size != metric.size or np.any(metric <= 0) or not np.all(np.isfinite(metric)):
        raise ValueError("sdev2 must contain one finite positive value per feature.")
    norm = float(np.sqrt(np.sum(values**2 * metric)))
    if norm < 1e-12:
        warnings.warn("Near-zero vector encountered during weighted normalization.")
        return np.zeros((values.size, 1), dtype=float)
    return (values / norm).reshape(-1, 1)


def _normalize_gradient_weighted(
    v: np.ndarray, sdev2: np.ndarray | None = None
) -> np.ndarray:
    """Apply the diagonal inverse metric, then weighted-normalize."""
    if sdev2 is None:
        return _normalize_vec(v)
    metric = np.asarray(sdev2, dtype=float).ravel()
    return _normalize_vec_weighted(np.asarray(v).ravel() / metric, metric)


def _get_kernel_matrix(flat_kernels: dict, sigma: float, ct_i: str, ct_j: str) -> np.ndarray:
    """Retrieve kernel matrix, trying both (i,j) and transposed (j,i) orderings."""
    name = f"kernel|sigma{sigma}|{ct_i}|{ct_j}"
    if name in flat_kernels:
        return flat_kernels[name]
    name_sym = f"kernel|sigma{sigma}|{ct_j}|{ct_i}"
    if name_sym in flat_kernels:
        return flat_kernels[name_sym].T
    raise KeyError(f"Kernel not found for pair ({ct_i}, {ct_j}) sigma={sigma}")


def _get_kernel_matrix_flat_multi(flat_kernels: dict, sigma: float, ct_i: str, ct_j: str, slide: str) -> np.ndarray:
    """Retrieve kernel for a specific slide, trying both orderings."""
    name = f"kernel|sigma{sigma}|{slide}|{ct_i}|{ct_j}"
    if name in flat_kernels:
        return flat_kernels[name]
    name_sym = f"kernel|sigma{sigma}|{slide}|{ct_j}|{ct_i}"
    if name_sym in flat_kernels:
        return flat_kernels[name_sym].T
    raise KeyError(f"Kernel not found for slide={slide} ({ct_i},{ct_j}) sigma={sigma}")


def _initialize_weights_svd(X_dict: dict, cell_types: list) -> dict:
    """Initialize weights as first right singular vector of each X matrix."""
    w_dict = {}
    for ct in cell_types:
        X = X_dict[ct]
        # Use scipy svds for consistency with R irlba
        k = min(1, min(X.shape) - 1)
        if k < 1:
            # Fallback for tiny matrices
            w_dict[ct] = _normalize_vec(X[0])
        else:
            try:
                _, _, Vt = svds(X.astype(float), k=1)
                w_dict[ct] = Vt[0].reshape(-1, 1)
            except Exception:
                _, _, Vt = np.linalg.svd(X, full_matrices=False)
                w_dict[ct] = Vt[0].reshape(-1, 1)
    return w_dict


def _check_convergence(w_new: dict, w_old: dict, cell_types: list) -> float:
    """Return sign-invariant max absolute difference across weight vectors."""
    max_diff = 0.0
    for ct in cell_types:
        diff_fwd = float(np.max(np.abs(w_new[ct] - w_old[ct])))
        diff_flip = float(np.max(np.abs(w_new[ct] + w_old[ct])))
        diff = min(diff_fwd, diff_flip)
        if np.isnan(diff):
            diff = 0.0
        max_diff = max(max_diff, diff)
    return max_diff


# ---------------------------------------------------------------------------
# First component
# ---------------------------------------------------------------------------

def optimize_bilinear(
    X_dict: dict,
    flat_kernels: dict,
    sigma: float,
    max_iter: int = 1000,
    tol: float = 1e-5,
    verbose: bool = True,
    step_size: float = 1.0,
    sdev2_dict: dict | None = None,
) -> dict:
    """Compute the first skrCCA component from cached PC-space operators.

    Parameters
    ----------
    X_dict : dict {ct: ndarray (n_cells, n_features)}
    flat_kernels : dict of kernel matrices
    sigma : float
    max_iter, tol : convergence parameters

    Returns
    -------
    w_dict : dict {ct: ndarray (n_features, 1)}
    """
    cell_types = list(X_dict.keys())
    n_features = X_dict[cell_types[0]].shape[1]
    if not 0 < step_size <= 1:
        raise ValueError("step_size must be in (0, 1].")

    # Build each small PC-space operator only once.  With two types the
    # constrained problem is exactly the singular-vector variational problem.
    Y = _compute_Y_resi(X_dict, flat_kernels, sigma, cell_types)
    if len(cell_types) == 2:
        return _solve_two_type_svd(Y, cell_types, 1, sdev2_dict)

    w_dict = _initialize_weights_svd(X_dict, cell_types)
    w_dict = _bilinear_from_Y_resi(
        w_dict, Y, n_features, max_iter, tol, verbose,
        step_size=step_size, sdev2_dict=sdev2_dict,
    )

    # Ensure (n_features, 1) shape
    for ct in cell_types:
        w_dict[ct] = w_dict[ct].reshape(-1, 1)
    return w_dict


# ---------------------------------------------------------------------------
# Subsequent components via deflation
# ---------------------------------------------------------------------------

def _compute_Y_resi(
    X_dict: dict,
    flat_kernels: dict,
    sigma: float,
    cell_types: list,
) -> dict:
    """Precompute Y[i][j] = X_i^T K_ij X_j for all pairs."""
    is_within = len(cell_types) == 1
    Y = {ct: {} for ct in cell_types}

    if is_within:
        ct = cell_types[0]
        X = X_dict[ct]
        K = _get_kernel_matrix(flat_kernels, sigma, ct, ct)
        Y[ct][ct] = X.T @ (K @ X)
    else:
        for ct_i, ct_j in combinations(cell_types, 2):
            X_i = X_dict[ct_i]
            X_j = X_dict[ct_j]
            K = _get_kernel_matrix(flat_kernels, sigma, ct_i, ct_j)
            Y_ij = X_i.T @ (K @ X_j)
            Y[ct_i][ct_j] = Y_ij
            Y[ct_j][ct_i] = Y_ij.T

    return Y


def _solve_two_type_svd(
    Y: dict,
    cell_types: list,
    n_cc: int = 1,
    sdev2_dict: dict | None = None,
) -> dict:
    """Solve every ordinary two-type skrCCA axis with one exact SVD."""
    if len(cell_types) != 2:
        raise ValueError("The exact SVD solver requires exactly two cell types.")
    ct1, ct2 = cell_types
    Y12 = np.asarray(Y[ct1][ct2], dtype=float)
    max_axes = min(Y12.shape)
    if n_cc < 1 or n_cc > max_axes:
        raise ValueError(
            f"n_cc must be between 1 and {max_axes} for this PC-space operator."
        )

    inv1 = inv2 = None
    if sdev2_dict is not None:
        inv1 = 1.0 / np.sqrt(np.asarray(sdev2_dict[ct1], dtype=float))
        inv2 = 1.0 / np.sqrt(np.asarray(sdev2_dict[ct2], dtype=float))
        Y12 = inv1[:, None] * Y12 * inv2[None, :]

    U, _, Vt = np.linalg.svd(Y12, full_matrices=False)
    W1 = U[:, :n_cc]
    W2 = Vt.T[:, :n_cc]
    if sdev2_dict is not None:
        W1 = inv1[:, None] * W1
        W2 = inv2[:, None] * W2
    return {ct1: W1, ct2: W2}


def _matches_two_type_first_axis(
    supplied: dict,
    exact: dict,
    cell_types: list,
    sdev2_dict: dict | None = None,
    tolerance: float = 1e-4,
) -> bool:
    """Whether a supplied first axis is the leading exact singular pair."""
    cosines = []
    for ct in cell_types:
        left = np.asarray(supplied[ct])[:, 0]
        right = np.asarray(exact[ct])[:, 0]
        if sdev2_dict is None:
            num = float(left @ right)
            den = float(np.linalg.norm(left) * np.linalg.norm(right))
        else:
            metric = np.asarray(sdev2_dict[ct], dtype=float)
            num = float(np.sum(metric * left * right))
            den = float(
                np.sqrt(np.sum(metric * left**2) * np.sum(metric * right**2))
            )
        cosines.append(num / den if den > 0 else np.nan)
    return bool(
        np.all(np.isfinite(cosines))
        and np.all(np.abs(np.abs(cosines) - 1.0) <= tolerance)
        and np.prod(np.sign(cosines)) > 0
    )


def _apply_deflation(
    Y: dict,
    w_dict: dict,
    qq: int,
    cell_types: list,
    sdev2_dict: dict | None = None,
    method: str = "rank1",
) -> dict:
    """Deflate PC-space operators with rank-one or two-sided projection.

    ``method="projection"`` supports both the unweighted metric (``sdev2_dict``
    is ``None``; whitened / ``scale_pcs=True`` PCs) and the weighted CCA metric
    (``sdev2_dict`` set; raw / ``scale_pcs=False`` PCs).  The weighted form is
    the exact image of the whitened orthogonal projection under the change of
    variables ``X~ = X diag(sdev)**-1`` and ``w~ = diag(sdev) w``, so the
    deflated operator is identical in both parametrizations.  This is what keeps
    every canonical axis of a >2-type problem the same whether or not PCs are
    scaled -- ``scale_pcs`` remains a pure reparametrization on all axes.
    """
    if method not in {"rank1", "projection"}:
        raise ValueError("method must be 'rank1' or 'projection'.")
    is_within = len(cell_types) == 1

    def project(Y1, w1, w2, d1=None, d2=None):
        # Two-sided projection deflation (I - p1 q1^T) Y (I - q2 p2^T) where q is
        # the weight normalized under the (possibly weighted) metric and p = D q.
        # With no metric (d=None) p == q and this reduces to the orthogonal
        # projection (I - u u^T) Y (I - v v^T).  With D = diag(sdev**2) it is the
        # oblique projection (I - D1 q1 q1^T) Y (I - q2 q2^T D2), i.e. the exact
        # image of the whitened orthogonal projection under w~ = diag(sdev) w.
        if d1 is None:
            q1 = w1 / np.linalg.norm(w1)
            p1 = q1
        else:
            d1 = np.asarray(d1, dtype=float).reshape(-1, 1)
            q1 = w1 / float(np.sqrt((w1 * d1 * w1).sum()))
            p1 = d1 * q1
        if d2 is None:
            q2 = w2 / np.linalg.norm(w2)
            p2 = q2
        else:
            d2 = np.asarray(d2, dtype=float).reshape(-1, 1)
            q2 = w2 / float(np.sqrt((w2 * d2 * w2).sum()))
            p2 = d2 * q2
        lam = float((q1.T @ Y1 @ q2).item())
        return Y1 - p1 @ (q1.T @ Y1) - (Y1 @ q2) @ p2.T + lam * (p1 @ p2.T)

    if is_within:
        ct = cell_types[0]
        Y1 = Y[ct][ct]
        w1 = w_dict[ct][:, qq : qq + 1]
        d_ct = None if sdev2_dict is None else sdev2_dict[ct]
        if method == "projection":
            Y[ct][ct] = project(Y1, w1, w1, d_ct, d_ct)
        else:
            lam = float((w1.T @ Y1 @ w1).flat[0])
            if sdev2_dict is None:
                left = right = w1
            else:
                left = right = w1 * np.asarray(sdev2_dict[ct])[:, None]
            Y[ct][ct] = Y1 - lam * (left @ right.T)
    else:
        for ct_i, ct_j in combinations(cell_types, 2):
            w1 = w_dict[ct_i][:, qq : qq + 1]
            w2 = w_dict[ct_j][:, qq : qq + 1]
            Y1 = Y[ct_i][ct_j]
            if method == "projection":
                d_i = None if sdev2_dict is None else sdev2_dict[ct_i]
                d_j = None if sdev2_dict is None else sdev2_dict[ct_j]
                Y[ct_i][ct_j] = project(Y1, w1, w2, d_i, d_j)
            else:
                lam = float((w1.T @ Y1 @ w2).flat[0])
                if sdev2_dict is None:
                    left, right = w1, w2
                else:
                    left = w1 * np.asarray(sdev2_dict[ct_i])[:, None]
                    right = w2 * np.asarray(sdev2_dict[ct_j])[:, None]
                Y[ct_i][ct_j] = Y1 - lam * (left @ right.T)
            Y[ct_j][ct_i] = Y[ct_i][ct_j].T

    return Y


def _initialize_next_component(Y: dict, cell_types: list) -> dict:
    """Initialize weights for the next component from deflated Y matrices."""
    is_within = len(cell_types) == 1
    w_new = {}

    if is_within:
        ct = cell_types[0]
        Y_sym = Y[ct][ct]
        Y_sym = (Y_sym + Y_sym.T) / 2  # enforce symmetry for numerical stability
        # Largest eigenvector of symmetric matrix
        eigenvalues, eigenvectors = np.linalg.eigh(Y_sym)
        # eigh returns ascending order — take last (largest)
        w_new[ct] = eigenvectors[:, -1:].copy()
    else:
        ct_0 = cell_types[0]
        # For Y[ct_0][ct]: left sv (U[:,0]) → ct_0, right sv (Vt[0]) → ct_j
        Y0j = Y[ct_0][cell_types[1]]
        try:
            U, _, Vt = svds(Y0j.astype(float), k=1)
            w_new[ct_0] = U[:, 0].reshape(-1, 1)
        except Exception:
            U, _, Vt = np.linalg.svd(Y0j, full_matrices=False)
            w_new[ct_0] = U[:, 0].reshape(-1, 1)

        for ct in cell_types[1:]:
            Y_i = Y[ct_0][ct]
            try:
                _, _, Vt = svds(Y_i.astype(float), k=1)
                w_new[ct] = Vt[0].reshape(-1, 1)
            except Exception:
                _, _, Vt = np.linalg.svd(Y_i, full_matrices=False)
                w_new[ct] = Vt[0].reshape(-1, 1)

    return w_new


def _bilinear_from_Y_resi(
    w_new: dict,
    Y: dict,
    n_features: int,
    max_iter: int,
    tol: float,
    verbose: bool = True,
    step_size: float = 1.0,
    sdev2_dict: dict | None = None,
) -> dict:
    """Iterative refinement using precomputed (deflated) Y matrices."""
    cell_types = list(w_new.keys())
    is_within = len(cell_types) == 1

    for iteration in range(max_iter):
        w_old = {ct: w_new[ct].copy() for ct in cell_types}

        if is_within:
            ct = cell_types[0]
            # Maximizing w^T Y w depends only on the symmetric part of Y, so
            # iterate on (Y + Y^T)/2.  For a symmetric self-kernel this is a
            # no-op; for an asymmetric (row/col-normalized) self-kernel it makes
            # this single-slide refinement agree with the symmetric operator
            # already used by _initialize_next_component and the multi-slide
            # direct solve.
            Y_ct = Y[ct][ct]
            Y_sym = (Y_ct + Y_ct.T) * 0.5
            update = _normalize_gradient_weighted(
                Y_sym @ w_new[ct],
                None if sdev2_dict is None else sdev2_dict[ct],
            )
            if step_size < 1:
                update = _normalize_vec_weighted(
                    (1 - step_size) * w_old[ct] + step_size * update,
                    None if sdev2_dict is None else sdev2_dict[ct],
                )
            w_new[ct] = update
        else:
            for ct_i in cell_types:
                update = np.zeros((n_features, 1), dtype=float)
                for ct_j in cell_types:
                    if ct_i == ct_j:
                        continue
                    Y_ij = Y[ct_i].get(ct_j)
                    if Y_ij is None:
                        raise ValueError(f"Missing Y_resi for ({ct_i}, {ct_j})")
                    update += Y_ij @ w_new[ct_j]
                metric = None if sdev2_dict is None else sdev2_dict[ct_i]
                update = _normalize_gradient_weighted(update, metric)
                if step_size < 1:
                    update = _normalize_vec_weighted(
                        (1 - step_size) * w_old[ct_i] + step_size * update,
                        metric,
                    )
                w_new[ct_i] = update

        diff = _check_convergence(w_new, w_old, cell_types)
        if diff <= tol:
            if verbose:
                print(f"  Component convergence at iteration {iteration} (max_diff={diff:.3e})")
            break
    else:
        warnings.warn("_bilinear_from_Y_resi: max_iter reached without convergence.")

    return w_new


def optimize_bilinear_n(
    X_dict: dict,
    flat_kernels: dict,
    sigma: float,
    w_dict: dict,
    cell_types: list,
    n_cc: int = 2,
    max_iter: int = 1000,
    tol: float = 1e-5,
    verbose: bool = True,
    step_size: float = 1.0,
    sdev2_dict: dict | None = None,
) -> dict:
    """Compute components 2 … n_cc via deflation.

    Parameters
    ----------
    w_dict : dict {ct: ndarray (n_features, k_start)} — already has first component(s)

    Returns
    -------
    w_dict : dict {ct: ndarray (n_features, n_cc)}
    """
    n_features = X_dict[cell_types[0]].shape[1]
    k_start = w_dict[cell_types[0]].shape[1]

    if n_cc <= k_start:
        raise ValueError(f"n_cc ({n_cc}) must be > existing components ({k_start}).")

    Y = _compute_Y_resi(X_dict, flat_kernels, sigma, cell_types)

    if len(cell_types) == 2:
        exact = _solve_two_type_svd(Y, cell_types, n_cc, sdev2_dict)
        if _matches_two_type_first_axis(
            w_dict, exact, cell_types, sdev2_dict=sdev2_dict
        ):
            return exact

    for qq in range(k_start - 1, n_cc - 1):
        # Deflate using component qq.  For >2 cell types the stationary vectors
        # are not pairwise singular vectors, so full projection (not rank-1) is
        # required to keep later axes orthogonal within every cell type.  We use
        # projection for both metrics: _apply_deflation implements the weighted
        # projection so scale_pcs=True/False yield the same axes (see DEFECT 1).
        # Two-type / within-type deflation keeps the established rank-1 rule.
        method = "projection" if len(cell_types) > 2 else "rank1"
        Y = _apply_deflation(
            Y, w_dict, qq, cell_types,
            sdev2_dict=sdev2_dict, method=method,
        )

        # Initialize next component
        w_new = _initialize_next_component(Y, cell_types)

        # Refine
        w_new = _bilinear_from_Y_resi(
            w_new, Y, n_features, max_iter, tol, verbose=verbose,
            step_size=step_size, sdev2_dict=sdev2_dict,
        )

        # Append to w_dict
        for ct in cell_types:
            w_dict[ct] = np.hstack([w_dict[ct], w_new[ct]])

    return w_dict


# ---------------------------------------------------------------------------
# Multi-slide optimization
# ---------------------------------------------------------------------------

def _compute_Y_multi_slides(
    X_list_all: dict,
    flat_kernels: dict,
    sigma: float,
    slides: list,
    cell_types: list,
) -> dict:
    """Sum PC-space operators over slides without stacking cells or kernels."""
    n_features = X_list_all[slides[0]][cell_types[0]].shape[1]
    Y = {ct: {} for ct in cell_types}
    if len(cell_types) == 1:
        ct = cell_types[0]
        total = np.zeros((n_features, n_features), dtype=float)
        for slide in slides:
            X = X_list_all.get(slide, {}).get(ct)
            if X is None:
                continue
            try:
                K = _get_kernel_matrix_flat_multi(
                    flat_kernels, sigma, ct, ct, slide
                )
            except KeyError:
                continue
            total += X.T @ (K @ X)
        Y[ct][ct] = total
        return Y

    for ct_i, ct_j in combinations(cell_types, 2):
        total = np.zeros((n_features, n_features), dtype=float)
        found = False
        for slide in slides:
            X_i = X_list_all.get(slide, {}).get(ct_i)
            X_j = X_list_all.get(slide, {}).get(ct_j)
            if X_i is None or X_j is None:
                continue
            try:
                K = _get_kernel_matrix_flat_multi(
                    flat_kernels, sigma, ct_i, ct_j, slide
                )
            except KeyError:
                continue
            total += X_i.T @ (K @ X_j)
            found = True
        if not found:
            raise KeyError(
                f"No kernel blocks found for pair ({ct_i}, {ct_j}) at sigma={sigma}."
            )
        Y[ct_i][ct_j] = total
        Y[ct_j][ct_i] = total.T
    return Y

def optimize_bilinear_multi_slides(
    X_list_all: dict,   # {slide: {ct: ndarray}}
    flat_kernels: dict,
    sigma: float,
    slides: list,
    max_iter: int = 1000,
    tol: float = 1e-5,
    verbose: bool = True,
    step_size: float = 1.0,
    sdev2_dict: dict | None = None,
) -> dict:
    """SkrCCA first component, multi-slide. Shared weights, sums contributions across slides.

    Objective: Maximize Σ_q w_i^T X_{i,q}^T K_{ij,q} X_{j,q} w_j
    """
    cell_types = list(X_list_all[slides[0]].keys())
    n_features = X_list_all[slides[0]][cell_types[0]].shape[1]
    Y = _compute_Y_multi_slides(
        X_list_all, flat_kernels, sigma, slides, cell_types
    )

    if len(cell_types) == 2:
        return _solve_two_type_svd(Y, cell_types, 1, sdev2_dict)

    if len(cell_types) == 1:
        ct = cell_types[0]
        Y_sym = (Y[ct][ct] + Y[ct][ct].T) * 0.5
        metric = None if sdev2_dict is None else np.asarray(sdev2_dict[ct])
        if metric is not None:
            inv = 1.0 / np.sqrt(metric)
            Y_sym = inv[:, None] * Y_sym * inv[None, :]
        values, vectors = np.linalg.eigh(Y_sym)
        w = vectors[:, -1:]
        if metric is not None:
            w = _normalize_vec_weighted(inv[:, None] * w, metric)
        return {ct: w}

    # Initialize weights from stacked data across all slides
    w_dict = {}
    for ct in cell_types:
        X_stacked = np.vstack([X_list_all[slide][ct] for slide in slides if ct in X_list_all[slide]])
        k = min(1, min(X_stacked.shape) - 1)
        if k < 1:
            w_dict[ct] = _normalize_vec(X_stacked[0])
        else:
            try:
                _, _, Vt = svds(X_stacked.astype(float), k=1)
                w_dict[ct] = Vt[0].reshape(-1, 1)
            except Exception:
                _, _, Vt = np.linalg.svd(X_stacked, full_matrices=False)
                w_dict[ct] = Vt[0].reshape(-1, 1)

    w_dict = _bilinear_from_Y_resi(
        w_dict, Y, n_features, max_iter, tol, verbose,
        step_size=step_size, sdev2_dict=sdev2_dict,
    )

    for ct in cell_types:
        w_dict[ct] = w_dict[ct].reshape(-1, 1)
    return w_dict


def optimize_bilinear_n_multi_slides(
    X_list_all: dict,
    flat_kernels: dict,
    sigma: float,
    slides: list,
    w_dict: dict,
    cell_types: list,
    n_cc: int = 2,
    max_iter: int = 1000,
    tol: float = 1e-5,
    verbose: bool = True,
    step_size: float = 1.0,
    sdev2_dict: dict | None = None,
) -> dict:
    """Compute components 2..n_cc for multi-slide via deflation."""
    n_features = X_list_all[slides[0]][cell_types[0]].shape[1]
    k_start = w_dict[cell_types[0]].shape[1]

    if n_cc <= k_start:
        raise ValueError(f"n_cc ({n_cc}) must be > existing components ({k_start}).")

    Y = _compute_Y_multi_slides(
        X_list_all, flat_kernels, sigma, slides, cell_types
    )

    if len(cell_types) == 2:
        exact = _solve_two_type_svd(Y, cell_types, n_cc, sdev2_dict)
        if _matches_two_type_first_axis(
            w_dict, exact, cell_types, sdev2_dict=sdev2_dict
        ):
            return exact

    for qq in range(k_start - 1, n_cc - 1):
        # >2 cell types require full projection deflation for later axes; the
        # weighted projection in _apply_deflation makes scale_pcs=True/False
        # produce the same axes (see DEFECT 1).  Two-type / within-type deflation
        # keeps the established rank-1 rule.
        method = "projection" if len(cell_types) > 2 else "rank1"
        Y = _apply_deflation(
            Y, w_dict, qq, cell_types,
            sdev2_dict=sdev2_dict, method=method,
        )
        w_new = _initialize_next_component(Y, cell_types)
        w_new = _bilinear_from_Y_resi(
            w_new, Y, n_features, max_iter, tol, verbose=verbose,
            step_size=step_size, sdev2_dict=sdev2_dict,
        )
        for ct in cell_types:
            w_dict[ct] = np.hstack([w_dict[ct], w_new[ct]])

    return w_dict
