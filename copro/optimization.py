"""SkrCCA power-method optimization — mirrors 04_optimization_function_refactored.R exactly."""

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
    """Return max absolute difference across all weight vectors."""
    max_diff = 0.0
    for ct in cell_types:
        diff = float(np.max(np.abs(w_new[ct] - w_old[ct])))
        if np.isnan(diff):
            diff = 0.0
        max_diff = max(max_diff, diff)
    return max_diff


def _compute_update_standard(
    ct_i: str,
    cell_types: list,
    X_dict: dict,
    flat_kernels: dict,
    sigma: float,
    w_dict: dict,
    n_features: int,
) -> np.ndarray:
    """Update vector for ct_i: sum_j≠i  X_i^T K_ij X_j w_j."""
    update = np.zeros((n_features, 1), dtype=float)
    X_i = X_dict[ct_i]
    for ct_j in cell_types:
        if ct_i == ct_j:
            continue
        K = _get_kernel_matrix(flat_kernels, sigma, ct_i, ct_j)
        X_j = X_dict[ct_j]
        w_j = w_dict[ct_j]
        # X_i^T (K (X_j w_j))
        v = X_j @ w_j          # (n_cells_j, 1)
        kv = K @ v              # (n_cells_i, 1)
        update += X_i.T @ kv   # (n_features, 1)
    return update


def _compute_update_within(X: np.ndarray, K: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Update vector for within-type: X^T K X w."""
    return X.T @ (K @ (X @ w))


# ---------------------------------------------------------------------------
# First component
# ---------------------------------------------------------------------------

def optimize_bilinear(
    X_dict: dict,
    flat_kernels: dict,
    sigma: float,
    max_iter: int = 1000,
    tol: float = 1e-5,
) -> dict:
    """SkrCCA power method — first component.

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
    is_within = len(cell_types) == 1

    w_dict = _initialize_weights_svd(X_dict, cell_types)

    for iteration in range(max_iter + 1):
        w_old = {ct: w_dict[ct].copy() for ct in cell_types}

        if is_within:
            ct = cell_types[0]
            X = X_dict[ct]
            K = _get_kernel_matrix(flat_kernels, sigma, ct, ct)
            update = _compute_update_within(X, K, w_dict[ct])
            w_dict[ct] = _normalize_vec(update)
        else:
            for ct_i in cell_types:
                update = _compute_update_standard(
                    ct_i, cell_types, X_dict, flat_kernels, sigma, w_dict, n_features
                )
                w_dict[ct_i] = _normalize_vec(update)

        diff = _check_convergence(w_dict, w_old, cell_types)
        if diff <= tol:
            print(f"Convergence reached at iteration {iteration} (max_diff={diff:.3e})")
            break
    else:
        warnings.warn(f"optimize_bilinear: max_iter={max_iter} reached without convergence.")

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


def _apply_deflation(Y: dict, w_dict: dict, qq: int, cell_types: list) -> dict:
    """Deflate Y matrices using component qq (0-indexed column in w_dict[ct])."""
    is_within = len(cell_types) == 1

    if is_within:
        ct = cell_types[0]
        Y1 = Y[ct][ct]
        w1 = w_dict[ct][:, qq : qq + 1]
        lam = float((w1.T @ Y1 @ w1).flat[0])
        Y[ct][ct] = Y1 - lam * (w1 @ w1.T)
    else:
        for ct_i, ct_j in combinations(cell_types, 2):
            w1 = w_dict[ct_i][:, qq : qq + 1]
            w2 = w_dict[ct_j][:, qq : qq + 1]
            Y1 = Y[ct_i][ct_j]
            lam = float((w1.T @ Y1 @ w2).flat[0])
            Y[ct_i][ct_j] = Y1 - lam * (w1 @ w2.T)
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
) -> dict:
    """Iterative refinement using precomputed (deflated) Y matrices."""
    cell_types = list(w_new.keys())
    is_within = len(cell_types) == 1

    for iteration in range(max_iter):
        w_old = {ct: w_new[ct].copy() for ct in cell_types}

        if is_within:
            ct = cell_types[0]
            w_new[ct] = _normalize_vec(Y[ct][ct] @ w_new[ct])
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
                w_new[ct_i] = _normalize_vec(update)

        diff = _check_convergence(w_new, w_old, cell_types)
        if diff <= tol:
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

    for qq in range(k_start - 1, n_cc - 1):
        # Deflate using component qq
        Y = _apply_deflation(Y, w_dict, qq, cell_types)

        # Initialize next component
        w_new = _initialize_next_component(Y, cell_types)

        # Refine
        w_new = _bilinear_from_Y_resi(w_new, Y, n_features, max_iter, tol)

        # Append to w_dict
        for ct in cell_types:
            w_dict[ct] = np.hstack([w_dict[ct], w_new[ct]])

    return w_dict


# ---------------------------------------------------------------------------
# Multi-slide optimization
# ---------------------------------------------------------------------------

def optimize_bilinear_multi_slides(
    X_list_all: dict,   # {slide: {ct: ndarray}}
    flat_kernels: dict,
    sigma: float,
    slides: list,
    max_iter: int = 1000,
    tol: float = 1e-5,
) -> dict:
    """SkrCCA first component, multi-slide. Shared weights, sums contributions across slides.

    Objective: Maximize Σ_q w_i^T X_{i,q}^T K_{ij,q} X_{j,q} w_j
    """
    cell_types = list(X_list_all[slides[0]].keys())
    n_features = X_list_all[slides[0]][cell_types[0]].shape[1]
    is_within = len(cell_types) == 1

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

    for iteration in range(max_iter + 1):
        w_old = {ct: w_dict[ct].copy() for ct in cell_types}

        if is_within:
            ct = cell_types[0]
            update = np.zeros((n_features, 1), dtype=float)
            for slide in slides:
                X = X_list_all[slide].get(ct)
                if X is None:
                    continue
                try:
                    K = _get_kernel_matrix_flat_multi(flat_kernels, sigma, ct, ct, slide)
                except KeyError:
                    continue
                update += X.T @ (K @ (X @ w_dict[ct]))
            w_dict[ct] = _normalize_vec(update)
        else:
            for ct_i in cell_types:
                update = np.zeros((n_features, 1), dtype=float)
                for ct_j in cell_types:
                    if ct_i == ct_j:
                        continue
                    w_j = w_dict[ct_j]
                    for slide in slides:
                        try:
                            X_i = X_list_all[slide][ct_i]
                            X_j = X_list_all[slide][ct_j]
                            K = _get_kernel_matrix_flat_multi(flat_kernels, sigma, ct_i, ct_j, slide)
                        except KeyError:
                            continue
                        update += X_i.T @ (K @ (X_j @ w_j))
                w_dict[ct_i] = _normalize_vec(update)

        diff = _check_convergence(w_dict, w_old, cell_types)
        if diff <= tol:
            print(f"Convergence reached at iteration {iteration} (max_diff={diff:.3e})")
            break
    else:
        warnings.warn(f"optimize_bilinear_multi_slides: max_iter={max_iter} reached without convergence.")

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
) -> dict:
    """Compute components 2..n_cc for multi-slide via deflation."""
    n_features = X_list_all[slides[0]][cell_types[0]].shape[1]
    k_start = w_dict[cell_types[0]].shape[1]

    if n_cc <= k_start:
        raise ValueError(f"n_cc ({n_cc}) must be > existing components ({k_start}).")

    def _make_Y_aggregate():
        """Compute Y_aggregate[ct_i][ct_j] = Σ_q X_{i,q}^T K_{ij,q} X_{j,q}"""
        is_within = len(cell_types) == 1
        Y = {ct: {} for ct in cell_types}
        if is_within:
            ct = cell_types[0]
            Y_sum = np.zeros((n_features, n_features))
            for slide in slides:
                try:
                    X = X_list_all[slide].get(ct)
                    if X is None:
                        continue
                    K = _get_kernel_matrix_flat_multi(flat_kernels, sigma, ct, ct, slide)
                    Y_sum += X.T @ (K @ X)
                except KeyError:
                    pass
            Y[ct][ct] = Y_sum
        else:
            for ct_i, ct_j in combinations(cell_types, 2):
                Y_sum = np.zeros((n_features, n_features))
                for slide in slides:
                    try:
                        X_i = X_list_all[slide].get(ct_i)
                        X_j = X_list_all[slide].get(ct_j)
                        if X_i is None or X_j is None:
                            continue
                        K = _get_kernel_matrix_flat_multi(flat_kernels, sigma, ct_i, ct_j, slide)
                        Y_sum += X_i.T @ (K @ X_j)
                    except KeyError:
                        pass
                Y[ct_i][ct_j] = Y_sum
                Y[ct_j][ct_i] = Y_sum.T
        return Y

    Y = _make_Y_aggregate()

    for qq in range(k_start - 1, n_cc - 1):
        Y = _apply_deflation(Y, w_dict, qq, cell_types)
        w_new = _initialize_next_component(Y, cell_types)
        w_new = _bilinear_from_Y_resi(w_new, Y, n_features, max_iter, tol)
        for ct in cell_types:
            w_dict[ct] = np.hstack([w_dict[ct], w_new[ct]])

    return w_dict
