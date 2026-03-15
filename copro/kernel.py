"""compute_kernel_matrix() — Gaussian RBF kernel from distance matrices."""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np

from .core import CoProSingle


def _kernel_flat_name(sigma: float, ct_i: str, ct_j: str) -> str:
    return f"kernel|sigma{sigma}|{ct_i}|{ct_j}"


def _dist_flat_name(ct_i: str, ct_j: str) -> str:
    return f"dist|{ct_i}|{ct_j}"


def _kernel_from_distance(dist_mat: np.ndarray, sigma: float, lower_limit: float) -> np.ndarray:
    """K(x,y) = exp(-0.5 * (d/σ)^2), zeros below lower_limit."""
    K = np.exp(-0.5 * (dist_mat / sigma) ** 2)
    K[K < lower_limit] = 0.0
    return K


def _process_kernel(
    K: np.ndarray,
    lower_limit: float,
    upper_quantile: float,
    row_normalize: bool = False,
    col_normalize: bool = False,
) -> np.ndarray:
    """Clip upper quantile, optionally row/col-normalize, then zero out below lower_limit."""
    valid = K[K >= lower_limit]
    if len(valid) == 0:
        return K
    upper_clip = float(np.quantile(valid, upper_quantile))
    K = K.copy()
    K[K >= upper_clip] = upper_clip

    if row_normalize and not col_normalize:
        # Each non-zero row sums to 1  (matches R rowNormalizeKernel=TRUE)
        row_sums = K.sum(axis=1)
        nz = row_sums > 1e-4
        K[nz] = K[nz] / row_sums[nz, np.newaxis]
    elif col_normalize and not row_normalize:
        col_sums = K.sum(axis=0)
        nz = col_sums > 1e-4
        K[:, nz] = K[:, nz] / col_sums[np.newaxis, nz]

    K[K < lower_limit] = 0.0
    return K


def _should_remove_sigma(
    K: np.ndarray,
    lower_limit: float,
    sigma: float,
    ct_i: str,
    ct_j: str,
    min_ave_cell_neighbor: float,
    all_sigmas: list,
) -> bool:
    n1, n2 = K.shape
    min_prop_zero = min_ave_cell_neighbor * min(n1, n2) / (n1 * n2)
    prop_nonzero = np.mean(K > lower_limit)

    if prop_nonzero < min_prop_zero:
        warnings.warn(
            f"Kernel for {ct_i}-{ct_j} sigma={sigma}: too sparse "
            f"({prop_nonzero:.4f} < {min_prop_zero:.4f}). "
            + ("Dropping this sigma." if len(all_sigmas) > 1 else "Only sigma available!")
        )
        if len(all_sigmas) == 1:
            raise ValueError(
                f"Only sigma={sigma} provided but kernel is too sparse. Use a larger sigma."
            )
        return True

    if np.all(np.isnan(K)):
        warnings.warn(f"Kernel for {ct_i}-{ct_j} sigma={sigma}: all NaN. Dropping sigma.")
        if len(all_sigmas) == 1:
            raise ValueError(f"Only sigma={sigma} provided and kernel is all NaN.")
        return True

    return False


def compute_kernel_matrix(
    obj,
    sigma_values: list,
    lower_limit: float = 1e-7,
    upper_quantile: float = 0.85,
    min_ave_cell_neighbor: float = 2.0,
    row_normalize_kernel: bool = False,
    col_normalize_kernel: bool = False,
):
    """Compute Gaussian kernel matrices for each sigma and cell-type pair.

    Dispatches to multi-slide version for CoProMulti objects.

    Single-slide: stores under 'kernel|sigma{s}|ct_i|ct_j'.
    Multi-slide: stores under 'kernel|sigma{s}|{slide}|ct_i|ct_j'.
    Removes sigma values that produce overly-sparse kernels.
    """
    from .core import CoProMulti
    if isinstance(obj, CoProMulti):
        return _compute_kernel_matrix_multi(
            obj, sigma_values, lower_limit, upper_quantile,
            min_ave_cell_neighbor, row_normalize_kernel, col_normalize_kernel
        )

    # --- Single-slide path ---
    cts = obj.cell_types_of_interest
    if not cts:
        raise ValueError("No cell types of interest. Run subset_data() first.")
    if not obj.distances:
        raise ValueError("Distance matrices missing. Run compute_distance() first.")

    sigma_values = list(sigma_values)
    obj.sigma_values = sigma_values
    kernel_matrices = {}
    sigmas_to_remove = set()

    if len(cts) == 1:
        pairs = [(cts[0], cts[0])]
    else:
        pairs = list(combinations(cts, 2))

    for sigma in sigma_values:
        sigma_bad = False

        for ct_i, ct_j in pairs:
            # Look up distance matrix (try both orderings)
            flat_name = f"dist|{ct_i}|{ct_j}"
            if flat_name not in obj.distances:
                flat_name = f"dist|{ct_j}|{ct_i}"
            if flat_name not in obj.distances:
                warnings.warn(f"No distance matrix for {ct_i}-{ct_j}. Skipping.")
                continue

            dist_mat = obj.distances[flat_name]

            # Handle Inf in within-type diagonals
            dist_use = dist_mat.copy()
            dist_use[~np.isfinite(dist_use)] = np.nanmax(dist_use[np.isfinite(dist_use)])

            K = _kernel_from_distance(dist_use, sigma, lower_limit)

            if _should_remove_sigma(K, lower_limit, sigma, ct_i, ct_j, min_ave_cell_neighbor, sigma_values):
                sigma_bad = True
                continue

            K = _process_kernel(K, lower_limit, upper_quantile, row_normalize_kernel, col_normalize_kernel)
            kernel_matrices[_kernel_flat_name(sigma, ct_i, ct_j)] = K

        if sigma_bad:
            sigmas_to_remove.add(sigma)

    # Remove bad sigma values
    for sigma in sigmas_to_remove:
        for ct_i, ct_j in pairs:
            key = _kernel_flat_name(sigma, ct_i, ct_j)
            kernel_matrices.pop(key, None)

    obj.sigma_values = [s for s in sigma_values if s not in sigmas_to_remove]
    obj.kernel_matrices = kernel_matrices
    return obj


def _compute_kernel_matrix_multi(
    obj,
    sigma_values,
    lower_limit=1e-7,
    upper_quantile=0.85,
    min_ave_cell_neighbor=2.0,
    row_normalize_kernel=False,
    col_normalize_kernel=False,
):
    """Multi-slide kernel computation. Keys: 'kernel|sigma{s}|{slide}|ct_i|ct_j'."""
    cts = obj.cell_types_of_interest
    slides = obj.slide_list
    sigma_values = list(sigma_values)
    obj.sigma_values = sigma_values
    kernel_matrices = {}
    sigmas_to_remove = set()

    if len(cts) == 1:
        pairs = [(cts[0], cts[0])]
    else:
        pairs = list(combinations(cts, 2))

    for sigma in sigma_values:
        sigma_bad = False
        for slide in slides:
            for ct_i, ct_j in pairs:
                flat_dist = f"dist|{slide}|{ct_i}|{ct_j}"
                if flat_dist not in obj.distances:
                    flat_dist = f"dist|{slide}|{ct_j}|{ct_i}"
                if flat_dist not in obj.distances:
                    continue

                dist_mat = obj.distances[flat_dist]
                dist_use = dist_mat.copy()
                dist_use[~np.isfinite(dist_use)] = np.nanmax(dist_use[np.isfinite(dist_use)])

                K = _kernel_from_distance(dist_use, sigma, lower_limit)

                if _should_remove_sigma(K, lower_limit, sigma, ct_i, ct_j, min_ave_cell_neighbor, sigma_values):
                    sigma_bad = True
                    continue

                K = _process_kernel(K, lower_limit, upper_quantile, row_normalize_kernel, col_normalize_kernel)
                flat_kernel = f"kernel|sigma{sigma}|{slide}|{ct_i}|{ct_j}"
                kernel_matrices[flat_kernel] = K

        if sigma_bad:
            sigmas_to_remove.add(sigma)

    for sigma in sigmas_to_remove:
        for slide in slides:
            for ct_i, ct_j in pairs:
                key = f"kernel|sigma{sigma}|{slide}|{ct_i}|{ct_j}"
                kernel_matrices.pop(key, None)

    obj.sigma_values = [s for s in sigma_values if s not in sigmas_to_remove]
    obj.kernel_matrices = kernel_matrices
    return obj
