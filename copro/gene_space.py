"""Batch-robust multi-slide CCA directly in gene space.

This module ports the R package's ``runGeneSpaceCCA`` workflow.  Unlike the
ordinary skrCCA pipeline, it does not use PCA.  It standardizes expression for
each slide/cell-type block, builds per-slide gene covariance operators, and
maximizes the average per-slide canonical correlation with a frozen-sigma
Jacobi iteration.

The public entry points are intentionally kept in this standalone module until
they are re-exported from :mod:`copro`.
"""

from __future__ import annotations

from itertools import combinations
from numbers import Integral, Real
from typing import Mapping
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist


MIN_CELLS_PER_SLIDE = 10


def _as_rng(random_state=None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _as_expression_matrix(matrix):
    """Coerce expression data without densifying a sparse input."""
    if sparse.issparse(matrix):
        out = matrix.astype(float, copy=False).tocsr()
    elif isinstance(matrix, pd.DataFrame):
        out = matrix.to_numpy(dtype=float)
    else:
        out = np.asarray(matrix, dtype=float)
    if out.ndim != 2:
        raise ValueError("Expression inputs must be two-dimensional.")
    return out


def _as_covariance_matrix(matrix):
    """Coerce dense covariance inputs while preserving sparse operators."""
    if sparse.issparse(matrix):
        out = matrix.astype(float, copy=False)
    elif isinstance(matrix, pd.DataFrame):
        out = matrix.to_numpy(dtype=float)
    else:
        out = np.asarray(matrix, dtype=float)
    if out.ndim != 2:
        raise ValueError("Covariance inputs must be two-dimensional.")
    return out


def _get_gene_names(obj, n_genes: int) -> np.ndarray:
    for attr in ("gene_names", "gene_list"):
        value = getattr(obj, attr, None)
        if value is not None and len(value) == n_genes:
            return np.asarray(value, dtype=object)
    data = getattr(obj, "normalized_data_sub", None)
    if isinstance(data, pd.DataFrame) and data.shape[1] == n_genes:
        return data.columns.to_numpy(dtype=object)
    return np.asarray([f"gene_{i}" for i in range(n_genes)], dtype=object)


def _get_cross_covariance(cross_for_slide: Mapping, ct_i: str, ct_j: str):
    forward = f"{ct_i}-{ct_j}"
    reverse = f"{ct_j}-{ct_i}"
    if forward in cross_for_slide:
        return _as_covariance_matrix(cross_for_slide[forward])
    if reverse in cross_for_slide:
        return _as_covariance_matrix(cross_for_slide[reverse]).T
    raise KeyError(f"Cross-covariance not found for pair: {ct_i} - {ct_j}")


def _check_convergence(weights_new: Mapping, weights_old: Mapping,
                       cell_types: list[str]) -> float:
    """Sign-invariant infinity-norm convergence check used by the R code."""
    max_diff = 0.0
    for ct in cell_types:
        new = np.asarray(weights_new[ct])
        old = np.asarray(weights_old[ct])
        forward = float(np.max(np.abs(new - old)))
        flipped = float(np.max(np.abs(new + old)))
        diff = min(forward, flipped)
        if np.isnan(diff):
            diff = 0.0
        max_diff = max(max_diff, diff)
    return max_diff


def _per_slide_sigmas(weights: Mapping, self_covariances: Mapping,
                       slides: list[str], cell_types: list[str]) -> dict:
    sigmas = {}
    for slide in slides:
        sigmas[slide] = {}
        for ct in cell_types:
            w = np.asarray(weights[ct], dtype=float).reshape(-1, 1)
            cov = _as_covariance_matrix(self_covariances[slide][ct])
            value = float((w.T @ cov @ w).item())
            sigmas[slide][ct] = max(np.sqrt(max(value, 0.0)), 1e-12)
    return sigmas


def compute_genespace_objective(
    weights: Mapping,
    self_covariances: Mapping,
    cross_covariances: Mapping,
    slides: list[str],
    cell_types: list[str],
) -> float:
    """Return the average per-slide gene-space CCA objective."""
    if not slides:
        raise ValueError("At least one slide is required.")
    sigmas = _per_slide_sigmas(weights, self_covariances, slides, cell_types)
    objective = 0.0
    for slide in slides:
        for ct_i, ct_j in combinations(cell_types, 2):
            cross = _get_cross_covariance(
                cross_covariances[slide], ct_i, ct_j
            )
            w_i = np.asarray(weights[ct_i], dtype=float).reshape(-1, 1)
            w_j = np.asarray(weights[ct_j], dtype=float).reshape(-1, 1)
            numerator = float((w_i.T @ cross @ w_j).item())
            objective += numerator / (
                sigmas[slide][ct_i] * sigmas[slide][ct_j]
            )
    return objective / len(slides)


def _validate_optimizer_inputs(self_covariances, cross_covariances, slides,
                               cell_types, max_iter, tol) -> tuple[int, int]:
    cell_types = list(cell_types)
    slides = list(slides)
    if len(cell_types) < 2:
        raise ValueError(
            "Gene-space CCA requires at least 2 cell types. Found: "
            + ", ".join(cell_types)
        )
    if not isinstance(max_iter, Integral) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not isinstance(tol, Real) or not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a positive finite scalar.")
    if not slides:
        raise ValueError("At least one slide is required.")

    try:
        first = _as_covariance_matrix(
            self_covariances[slides[0]][cell_types[0]]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed per-slide self-covariance structure.") from exc
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("Self-covariance matrices must be square.")
    n_genes = first.shape[0]

    for slide in slides:
        if slide not in self_covariances or slide not in cross_covariances:
            raise ValueError(f"Missing covariance data for slide '{slide}'.")
        for ct in cell_types:
            cov = _as_covariance_matrix(self_covariances[slide][ct])
            if cov.shape != (n_genes, n_genes):
                raise ValueError(
                    f"Self-covariance shape mismatch for {slide}/{ct}: {cov.shape}."
                )
        for ct_i, ct_j in combinations(cell_types, 2):
            cross = _get_cross_covariance(
                cross_covariances[slide], ct_i, ct_j
            )
            if cross.shape != (n_genes, n_genes):
                raise ValueError(
                    f"Cross-covariance shape mismatch for {slide}/{ct_i}-{ct_j}: "
                    f"{cross.shape}."
                )
    return n_genes, len(slides)


def optimize_genespace_avg_corr(
    self_covariances: Mapping,
    cross_covariances: Mapping,
    slides,
    cell_types,
    max_iter: int = 3000,
    tol: float = 1e-6,
    verbose: bool = True,
    random_state=0,
) -> dict[str, np.ndarray]:
    """Find the first gene-space canonical component.

    The update is a simultaneous (Jacobi) frozen-sigma sweep: all cell-type
    updates use weights and per-slide score variances from the previous
    iterate.  ``random_state`` may be an integer or ``numpy.random.Generator``.
    """
    cell_types = list(cell_types)
    slides = list(slides)
    n_genes, n_slides = _validate_optimizer_inputs(
        self_covariances, cross_covariances, slides, cell_types, max_iter, tol
    )
    rng = _as_rng(random_state)
    weights = {}
    for ct in cell_types:
        vector = rng.normal(size=(n_genes, 1))
        weights[ct] = vector / np.linalg.norm(vector)

    max_diff = np.inf
    for iteration in range(1, max_iter + 1):
        old = {ct: weights[ct].copy() for ct in cell_types}
        sigmas = _per_slide_sigmas(old, self_covariances, slides, cell_types)
        updated = {}
        for ct_i in cell_types:
            update = np.zeros((n_genes, 1), dtype=float)
            for slide in slides:
                sigma_i = sigmas[slide][ct_i]
                for ct_j in cell_types:
                    if ct_j == ct_i:
                        continue
                    sigma_j = sigmas[slide][ct_j]
                    cross = _get_cross_covariance(
                        cross_covariances[slide], ct_i, ct_j
                    )
                    update += (cross @ (old[ct_j] / sigma_j)) / sigma_i
            update /= n_slides
            norm = float(np.linalg.norm(update))
            if norm > 0:
                updated[ct_i] = update / norm
            else:
                warnings.warn(
                    f"Zero gradient norm for cell type '{ct_i}' at iter "
                    f"{iteration}; keeping previous weight.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                updated[ct_i] = old[ct_i]
        weights = updated
        max_diff = _check_convergence(weights, old, cell_types)

        if verbose and (iteration == 1 or iteration % 500 == 0):
            objective = compute_genespace_objective(
                weights, self_covariances, cross_covariances, slides, cell_types
            )
            print(
                f"  Iter {iteration}: max_diff = {max_diff:.2e}, "
                f"objective = {objective:.4f}"
            )
        if max_diff <= tol:
            if verbose:
                objective = compute_genespace_objective(
                    weights, self_covariances, cross_covariances,
                    slides, cell_types
                )
                print(
                    f"  Converged at iteration {iteration} "
                    f"(max_diff = {max_diff:.2e}, objective = {objective:.4f})"
                )
            break

    if max_diff > tol:
        warnings.warn(
            f"Did not converge after {max_iter} iterations "
            f"(max_diff = {max_diff:.2e}).",
            RuntimeWarning,
            stacklevel=2,
        )

    objective = compute_genespace_objective(
        weights, self_covariances, cross_covariances, slides, cell_types
    )
    if objective < 0:
        weights[cell_types[0]] = -weights[cell_types[0]]
    return weights


def optimize_genespace_avg_corr_n(
    self_covariances: Mapping,
    cross_covariances: Mapping,
    slides,
    cell_types,
    weights: Mapping,
    n_cc: int = 2,
    max_iter: int = 3000,
    tol: float = 1e-6,
    verbose: bool = True,
    random_state=0,
) -> dict[str, np.ndarray]:
    """Append components through ``n_cc`` using weight-space Gram-Schmidt."""
    cell_types = list(cell_types)
    slides = list(slides)
    n_genes, n_slides = _validate_optimizer_inputs(
        self_covariances, cross_covariances, slides, cell_types, max_iter, tol
    )
    if not isinstance(n_cc, Integral) or isinstance(n_cc, bool) or n_cc < 1:
        raise ValueError("n_cc must be a positive integer.")

    out = {
        ct: np.asarray(weights[ct], dtype=float).copy()
        for ct in cell_types
    }
    for ct in cell_types:
        if out[ct].ndim == 1:
            out[ct] = out[ct][:, None]
        if out[ct].shape[0] != n_genes:
            raise ValueError(f"Weight length mismatch for cell type '{ct}'.")
    k_start = out[cell_types[0]].shape[1]
    if any(out[ct].shape[1] != k_start for ct in cell_types):
        raise ValueError("All cell types must have the same number of components.")
    if n_cc <= k_start:
        raise ValueError(
            f"n_cc ({n_cc}) must be greater than existing components ({k_start})."
        )

    rng = _as_rng(random_state)
    for component in range(k_start + 1, n_cc + 1):
        if verbose:
            print(f"  Finding CC {component} ...")
        current = {}
        for ct in cell_types:
            vector = rng.normal(size=(n_genes, 1))
            current[ct] = vector / np.linalg.norm(vector)

        max_diff = np.inf
        for iteration in range(1, max_iter + 1):
            old = {ct: current[ct].copy() for ct in cell_types}
            sigmas = _per_slide_sigmas(
                old, self_covariances, slides, cell_types
            )
            updated = {}
            for ct_i in cell_types:
                update = np.zeros((n_genes, 1), dtype=float)
                for slide in slides:
                    sigma_i = sigmas[slide][ct_i]
                    for ct_j in cell_types:
                        if ct_j == ct_i:
                            continue
                        sigma_j = sigmas[slide][ct_j]
                        cross = _get_cross_covariance(
                            cross_covariances[slide], ct_i, ct_j
                        )
                        update += (cross @ (old[ct_j] / sigma_j)) / sigma_i
                update /= n_slides

                for previous in range(component - 1):
                    prior = out[ct_i][:, previous:previous + 1]
                    update -= float((update.T @ prior).item()) * prior

                norm = float(np.linalg.norm(update))
                if norm > 0:
                    updated[ct_i] = update / norm
                else:
                    warnings.warn(
                        "Zero norm after Gram-Schmidt deflation for cell type "
                        f"'{ct_i}' at CC {component}; signal subspace likely "
                        "exhausted.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    updated[ct_i] = np.zeros((n_genes, 1), dtype=float)
            current = updated
            max_diff = _check_convergence(current, old, cell_types)
            if verbose and (iteration == 1 or iteration % 500 == 0):
                print(f"    Iter {iteration}: max_diff = {max_diff:.2e}")
            if max_diff <= tol:
                if verbose:
                    print(
                        f"    CC {component} converged at iteration {iteration}"
                    )
                break

        if max_diff > tol:
            warnings.warn(
                f"CC {component} did not converge after {max_iter} iterations "
                f"(max_diff = {max_diff:.2e}).",
                RuntimeWarning,
                stacklevel=2,
            )
        objective = compute_genespace_objective(
            current, self_covariances, cross_covariances, slides, cell_types
        )
        if objective < 0:
            current[cell_types[0]] = -current[cell_types[0]]
        for ct in cell_types:
            out[ct] = np.column_stack((out[ct], current[ct]))
    return out


def _prepare_gene_space_data(
    obj,
    clip="quantile",
    min_prevalence: float = 0.008,
    min_cells: int = 20,
    cell_types=None,
    slides=None,
    min_cells_per_slide: int = MIN_CELLS_PER_SLIDE,
    materialize_z: bool = True,
) -> dict:
    expression_value = getattr(obj, "normalized_data_sub", None)
    cell_type_value = getattr(obj, "cell_types_sub", None)
    metadata = getattr(obj, "meta_data_sub", None)
    if expression_value is None or cell_type_value is None or metadata is None:
        raise ValueError("Run subset_data() before run_gene_space_cca().")
    if "slideID" not in metadata.columns:
        raise ValueError("meta_data_sub must contain a 'slideID' column.")

    expression = _as_expression_matrix(expression_value)
    cell_type_values = np.asarray(cell_type_value, dtype=object)
    slide_values = metadata["slideID"].astype(str).to_numpy()
    if len(cell_type_values) != expression.shape[0] or len(slide_values) != expression.shape[0]:
        raise ValueError("Subset expression, cell types, and metadata are misaligned.")

    cell_types = list(cell_types if cell_types is not None else
                      getattr(obj, "cell_types_of_interest", []))
    slides = list(slides if slides is not None else getattr(obj, "slide_list", []))
    slides = [str(slide) for slide in slides]
    if not cell_types:
        cell_types = list(dict.fromkeys(cell_type_values.tolist()))
    if not slides:
        slides = list(dict.fromkeys(slide_values.tolist()))

    if not isinstance(min_prevalence, Real) or not np.isfinite(min_prevalence) or not 0 <= min_prevalence <= 1:
        raise ValueError("min_prevalence must be between 0 and 1.")
    if not isinstance(min_cells, Integral) or isinstance(min_cells, bool) or min_cells < 0:
        raise ValueError("min_cells must be a non-negative integer.")
    if (not isinstance(min_cells_per_slide, Integral)
            or isinstance(min_cells_per_slide, bool)
            or min_cells_per_slide < 1):
        raise ValueError("min_cells_per_slide must be a positive integer.")

    if sparse.issparse(expression):
        expressed_count = np.asarray((expression > 0).sum(axis=0)).ravel()
        prevalence = expressed_count / expression.shape[0]
    else:
        prevalence = np.mean(expression > 0, axis=0)
        expressed_count = np.sum(expression > 0, axis=0)
    keep = (prevalence >= min_prevalence) & (expressed_count >= min_cells)
    if not np.any(keep):
        raise ValueError(
            "No genes pass the prevalence filter. Lower min_prevalence or min_cells."
        )
    all_genes = _get_gene_names(obj, expression.shape[1])
    genes = all_genes[keep]
    gene_indices = np.flatnonzero(keep)
    expression = expression[:, keep].copy()

    if isinstance(clip, str) and clip == "quantile":
        if sparse.issparse(expression):
            positive = expression.data[expression.data > 0]
        else:
            positive = expression[expression > 0]
        if positive.size == 0:
            raise ValueError("No positive expression values remain after filtering.")
        clip_value = float(np.quantile(positive, 0.98))
    elif isinstance(clip, Real) and not isinstance(clip, bool) and np.isfinite(clip):
        clip_value = float(clip)
    else:
        raise ValueError('clip must be "quantile" or a single numeric threshold.')
    if sparse.issparse(expression):
        expression.data[expression.data > clip_value] = clip_value
    else:
        expression[expression > clip_value] = clip_value

    dropped = {}
    for slide in slides:
        slide_mask = slide_values == slide
        missing = [
            ct for ct in cell_types
            if int(np.sum(slide_mask & (cell_type_values == ct)))
            < min_cells_per_slide
        ]
        if missing:
            dropped[slide] = missing

    for slide, missing in dropped.items():
        warnings.warn(
            f"Slide '{slide}' dropped: cell type(s) {', '.join(missing)} "
            f"have fewer than {min_cells_per_slide} cells.",
            RuntimeWarning,
            stacklevel=2,
        )
    valid_slides = [slide for slide in slides if slide not in dropped]
    if not valid_slides:
        raise ValueError("No slides have all cell types present after filtering.")

    prepared = {
        "expression": expression,
        "cell_type_values": cell_type_values,
        "slide_values": slide_values,
    }
    if materialize_z:
        z_by_slide = {
            slide: {
                ct: _standardize_prepared_block(prepared, slide, ct)
                for ct in cell_types
            }
            for slide in valid_slides
        }
    else:
        z_by_slide = None

    return {
        "z_by_slide": z_by_slide,
        "genes": genes,
        "gene_indices": gene_indices,
        "slides": valid_slides,
        "clip_value": clip_value,
        **prepared,
    }


def _standardize_prepared_block(prepared: Mapping, slide: str, ct: str):
    """Materialize and z-score one retained slide/cell-type block."""
    mask = (
        (prepared["slide_values"] == slide)
        & (prepared["cell_type_values"] == ct)
    )
    block = prepared["expression"][mask]
    if sparse.issparse(block):
        block = block.toarray()
    else:
        block = np.asarray(block, dtype=float)
    means = block.mean(axis=0)
    stds = block.std(axis=0, ddof=1)
    z = np.divide(
        block - means,
        stds,
        out=np.zeros_like(block, dtype=float),
        where=np.isfinite(stds) & (stds != 0),
    )
    z[~np.isfinite(z)] = 0.0
    return z


def _get_kernel(flat_kernels: Mapping, sigma: float, ct_i: str, ct_j: str,
                slide: str):
    forward = f"kernel|sigma{sigma}|{slide}|{ct_i}|{ct_j}"
    reverse = f"kernel|sigma{sigma}|{slide}|{ct_j}|{ct_i}"
    if forward in flat_kernels:
        return flat_kernels[forward]
    if reverse in flat_kernels:
        return flat_kernels[reverse].T
    raise KeyError(
        f"Kernel not found for slide={slide}, pair=({ct_i}, {ct_j}), "
        f"sigma={sigma}."
    )


def _precompute_covariances(z_by_slide: Mapping, flat_kernels: Mapping,
                            sigma: float, slides: list[str],
                            cell_types: list[str]) -> tuple[dict, dict]:
    self_covariances = {}
    cross_covariances = {}
    for slide in slides:
        self_covariances[slide] = {}
        cross_covariances[slide] = {}
        for ct in cell_types:
            z = z_by_slide[slide][ct]
            self_covariances[slide][ct] = (z.T @ z) / z.shape[0]
        for ct_i, ct_j in combinations(cell_types, 2):
            z_i = z_by_slide[slide][ct_i]
            z_j = z_by_slide[slide][ct_j]
            kernel = _get_kernel(flat_kernels, sigma, ct_i, ct_j, slide)
            if kernel.shape != (z_i.shape[0], z_j.shape[0]):
                raise ValueError(
                    f"Kernel shape {kernel.shape} does not match standardized "
                    f"expression blocks {(z_i.shape[0], z_j.shape[0])} for "
                    f"{slide}/{ct_i}-{ct_j}."
                )
            cross_covariances[slide][f"{ct_i}-{ct_j}"] = (
                z_i.T @ (kernel @ z_j)
            ) / np.sqrt(z_i.shape[0] * z_j.shape[0])
    return self_covariances, cross_covariances


_DISTANCE_DEFAULTS = {
    "dist_type": "Euclidean2D",
    "x_dist_scale": 1.0,
    "y_dist_scale": 1.0,
    "z_dist_scale": 1.0,
    "normalize_distance": True,
    "normalize_target": 0.01,
    "normalization_scope": "global",
    "truncate_low_dist": True,
    "knn_k": 10,
    "geodesic_threshold": 10,
    "geodesic_cutoff": 7,
}
_KERNEL_DEFAULTS = {
    "lower_limit": 1e-7,
    "upper_quantile": 0.85,
    "normalize_kernel": False,
    "min_ave_cell_neighbor": 2.0,
    "row_normalize_kernel": False,
    "col_normalize_kernel": False,
}
_DISTANCE_ALIASES = {
    "distType": "dist_type", "xDistScale": "x_dist_scale",
    "yDistScale": "y_dist_scale", "zDistScale": "z_dist_scale",
    "normalizeDistance": "normalize_distance",
    "normalizeTarget": "normalize_target",
    "normalizationScope": "normalization_scope",
    "truncateLowDist": "truncate_low_dist",
}
_KERNEL_ALIASES = {
    "lowerLimit": "lower_limit", "upperQuantile": "upper_quantile",
    "normalizeKernel": "normalize_kernel",
    "minAveCellNeighor": "min_ave_cell_neighbor",
    "rowNormalizeKernel": "row_normalize_kernel",
    "colNormalizeKernel": "col_normalize_kernel",
}


def _resolve_options(values, defaults, aliases, label):
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    resolved = dict(defaults)
    for key, value in values.items():
        canonical = aliases.get(key, key)
        if canonical not in defaults:
            raise ValueError(
                f"Unknown {label} option '{key}'. Allowed: "
                + ", ".join(defaults)
            )
        resolved[canonical] = value
    return resolved


def _resolve_streaming_options(distance_args=None, kernel_args=None):
    distance = _resolve_options(
        distance_args, _DISTANCE_DEFAULTS, _DISTANCE_ALIASES, "distance_args"
    )
    kernel = _resolve_options(
        kernel_args, _KERNEL_DEFAULTS, _KERNEL_ALIASES, "kernel_args"
    )
    if distance["dist_type"] not in ("Euclidean2D", "Euclidean3D"):
        raise NotImplementedError(
            "Streaming gene-space CCA currently supports Euclidean2D and "
            "Euclidean3D distances."
        )
    if distance["normalization_scope"] not in ("global", "per_slide"):
        raise ValueError("normalization_scope must be 'global' or 'per_slide'.")
    scales = [distance["x_dist_scale"], distance["y_dist_scale"],
              distance["z_dist_scale"]]
    if any(not isinstance(value, Real) or not np.isfinite(value) or value <= 0
           for value in scales):
        raise ValueError("Distance scales must be positive finite scalars.")
    target = distance["normalize_target"]
    if not isinstance(target, Real) or not np.isfinite(target) or target <= 0:
        raise ValueError("normalize_target must be a positive finite scalar.")
    if kernel["row_normalize_kernel"] and kernel["col_normalize_kernel"]:
        raise ValueError("Cannot do both row-wise and column-wise normalization.")
    if not 0 < kernel["lower_limit"] < 1:
        raise ValueError("lower_limit must be in (0, 1).")
    if not 0 < kernel["upper_quantile"] < 1:
        raise ValueError("upper_quantile must be in (0, 1).")
    return distance, kernel


def _pair_distance(obj, slide: str, ct_i: str, ct_j: str, distance: Mapping):
    metadata = obj.meta_data_sub
    slide_values = metadata["slideID"].astype(str).to_numpy()
    cell_types = np.asarray(obj.cell_types_sub, dtype=object)
    coordinates = obj.location_data_sub
    columns = ["x", "y"]
    if distance["dist_type"] == "Euclidean3D":
        columns.append("z")
    missing = [column for column in columns if column not in coordinates.columns]
    if missing:
        raise ValueError(
            f"location_data_sub is missing coordinate column(s): {', '.join(missing)}."
        )
    scales = np.asarray([
        distance["x_dist_scale"], distance["y_dist_scale"],
        *([distance["z_dist_scale"]] if len(columns) == 3 else []),
    ])
    mask_i = (slide_values == slide) & (cell_types == ct_i)
    mask_j = (slide_values == slide) & (cell_types == ct_j)
    coords_i = coordinates.loc[mask_i, columns].to_numpy(dtype=float) * scales
    coords_j = coordinates.loc[mask_j, columns].to_numpy(dtype=float) * scales
    return cdist(coords_i, coords_j)


def _process_distance(distance_matrix: np.ndarray, truncate: bool):
    matrix = np.asarray(distance_matrix, dtype=float).copy()
    zeros = matrix == 0
    if np.any(zeros):
        positive = matrix[np.isfinite(matrix) & (matrix > 0)]
        if positive.size == 0:
            raise ValueError("No finite non-zero distances found.")
        matrix[zeros] = positive.min()
        warnings.warn(
            "Zero distances detected; replaced with the smallest non-zero distance.",
            RuntimeWarning,
            stacklevel=2,
        )
    finite = matrix[np.isfinite(matrix) & (matrix != 0)]
    if finite.size == 0:
        raise ValueError("No finite non-zero distances found.")
    probability = min(1e-3, 2.0 / max(matrix.shape))
    percentile = float(np.quantile(finite, probability))
    if truncate:
        matrix[(matrix < percentile) & np.isfinite(matrix)] = percentile
    return matrix, percentile


def _process_kernel(kernel_matrix: np.ndarray, options: Mapping):
    kernel = np.asarray(kernel_matrix, dtype=float).copy()
    valid = kernel[(kernel >= options["lower_limit"]) & np.isfinite(kernel)]
    if valid.size == 0:
        kernel[np.isfinite(kernel)] = 0.0
        return kernel
    upper_clip = float(np.quantile(valid, options["upper_quantile"]))
    kernel[(kernel >= upper_clip) & np.isfinite(kernel)] = upper_clip
    if options["normalize_kernel"] and not options["row_normalize_kernel"] and not options["col_normalize_kernel"]:
        row_sums = np.nansum(kernel, axis=1)
        useful = row_sums[row_sums > 1e-5]
        if useful.size:
            kernel /= float(np.median(useful))
    elif options["row_normalize_kernel"]:
        row_sums = np.nansum(kernel, axis=1)
        keep = row_sums > 1e-4
        kernel[keep] /= row_sums[keep, None]
    elif options["col_normalize_kernel"]:
        col_sums = np.nansum(kernel, axis=0)
        keep = col_sums > 1e-4
        kernel[:, keep] /= col_sums[None, keep]
    kernel[(kernel < options["lower_limit"]) & np.isfinite(kernel)] = 0.0
    return kernel


def _streaming_covariances(obj, prepared: Mapping, sigma: float,
                           slides: list[str], cell_types: list[str],
                           distance_args=None, kernel_args=None,
                           verbose: bool = True):
    distance, kernel_options = _resolve_streaming_options(
        distance_args, kernel_args
    )
    pairs = list(combinations(cell_types, 2))
    self_covariances = {}
    percentiles = {slide: [] for slide in slides}
    if distance["normalize_distance"]:
        if verbose and distance["normalization_scope"] == "global":
            print("  Streaming phase 1: per-pair percentiles (global scope)...")
        for slide in slides:
            for ct_i, ct_j in pairs:
                raw = _pair_distance(obj, slide, ct_i, ct_j, distance)
                _, percentile = _process_distance(
                    raw, bool(distance["truncate_low_dist"])
                )
                percentiles[slide].append(percentile)
        all_percentiles = [value for values in percentiles.values()
                           for value in values if np.isfinite(value)]
        if not all_percentiles:
            raise ValueError("Streaming: no valid distance percentile across slides.")
        global_scale = distance["normalize_target"] / min(all_percentiles)
        if verbose and distance["normalization_scope"] == "global":
            print(f"  Streaming GLOBAL scaling factor = {global_scale:g}")
    else:
        global_scale = 1.0

    cross_covariances = {}
    for slide in slides:
        if verbose:
            print(f"  Streaming slide: {slide}")
        # Standardized expression remains bounded to one slide. In
        # particular, a sparse cells-by-genes input is never converted to one
        # complete dense array in streaming mode.
        z_for_slide = {
            ct: _standardize_prepared_block(prepared, slide, ct)
            for ct in cell_types
        }
        self_covariances[slide] = {
            ct: (z.T @ z) / z.shape[0]
            for ct, z in z_for_slide.items()
        }
        if distance["normalize_distance"] and distance["normalization_scope"] == "per_slide":
            finite = [value for value in percentiles[slide] if np.isfinite(value)]
            if not finite:
                raise ValueError(
                    f"Streaming: no valid distance percentile for slide '{slide}'."
                )
            scale = distance["normalize_target"] / min(finite)
        else:
            scale = global_scale
        cross_covariances[slide] = {}
        for ct_i, ct_j in pairs:
            raw = _pair_distance(obj, slide, ct_i, ct_j, distance)
            processed, _ = _process_distance(
                raw, bool(distance["truncate_low_dist"])
            )
            processed *= scale
            kernel_matrix = np.exp(-0.5 * (processed / sigma) ** 2)
            kernel_matrix[kernel_matrix < kernel_options["lower_limit"]] = 0.0
            kernel_matrix = _process_kernel(kernel_matrix, kernel_options)
            z_i = z_for_slide[ct_i]
            z_j = z_for_slide[ct_j]
            cross_covariances[slide][f"{ct_i}-{ct_j}"] = (
                z_i.T @ (kernel_matrix @ z_j)
            ) / np.sqrt(z_i.shape[0] * z_j.shape[0])
    return self_covariances, cross_covariances


def _store_results(obj, weights: Mapping, z_by_slide,
                   sigma: float, cell_types: list[str], n_cc: int,
                   genes: np.ndarray, gene_indices: np.ndarray,
                   slides: list[str], prepared: Mapping | None = None):
    sigma_name = f"sigma_{sigma}"
    gscca_name = f"gscca_{sigma_name}"
    obj.skr_cca_out[gscca_name] = {
        ct: np.asarray(weights[ct], dtype=float).copy() for ct in cell_types
    }
    obj.n_cc = n_cc
    for ct in cell_types:
        obj.gene_scores[f"geneScores|sigma{sigma}|{ct}"] = np.asarray(
            weights[ct], dtype=float
        ).copy()

    cell_type_values = np.asarray(obj.cell_types_sub, dtype=object)
    slide_values = obj.meta_data_sub["slideID"].astype(str).to_numpy()
    for ct in cell_types:
        ct_mask = cell_type_values == ct
        scores = np.full((int(ct_mask.sum()), n_cc), np.nan, dtype=float)
        ct_positions = np.flatnonzero(ct_mask)
        position_lookup = {global_position: local_position
                           for local_position, global_position in enumerate(ct_positions)}
        for slide in slides:
            block_mask = ct_mask & (slide_values == slide)
            block_positions = np.flatnonzero(block_mask)
            if z_by_slide is None:
                if prepared is None:
                    raise ValueError(
                        "Prepared expression is required for streaming scores."
                    )
                z = _standardize_prepared_block(prepared, slide, ct)
            else:
                z = z_by_slide[slide][ct]
            raw = z @ weights[ct]
            means = raw.mean(axis=0)
            stds = raw.std(axis=0, ddof=1)
            normalized = raw.copy()
            variable = np.isfinite(stds) & (stds > 0)
            normalized[:, variable] = (
                raw[:, variable] - means[variable]
            ) / stds[variable]
            local_positions = [position_lookup[position] for position in block_positions]
            scores[local_positions] = normalized
        obj.cell_scores[f"cellScores|sigma{sigma}|{ct}"] = scores
        for component in range(n_cc):
            column = f"cellScore_{sigma_name}_cc_index_{component + 1}"
            if column not in obj.meta_data_sub.columns:
                obj.meta_data_sub[column] = np.nan
            obj.meta_data_sub.loc[ct_mask, column] = scores[:, component]

    # Explicit metadata is needed because current Python score arrays do not
    # carry R-style row names.
    obj.gene_space_genes = list(genes)
    obj.gene_space_gene_indices = np.asarray(gene_indices, dtype=int)
    obj.gene_space_slides = list(slides)
    return obj


def run_gene_space_cca(
    obj,
    sigma: float,
    n_cc: int = 2,
    clip="quantile",
    min_prevalence: float = 0.008,
    min_cells: int = 20,
    max_iter: int = 3000,
    tol: float = 1e-6,
    streaming: bool = False,
    distance_args=None,
    kernel_args=None,
    verbose: bool = True,
    random_state=0,
):
    """Run batch-robust gene-space CCA on a ``CoProMulti`` object.

    Slot-based mode consumes existing per-slide kernels.  Streaming mode
    computes distances and kernels pair-by-pair, defaults to one global
    distance-normalization factor across slides, and leaves the object's
    distance/kernel caches untouched.
    """
    from .core import CoProMulti

    if not isinstance(obj, CoProMulti):
        raise TypeError(
            "run_gene_space_cca requires a CoProMulti object (multi-slide data)."
        )
    if not isinstance(streaming, (bool, np.bool_)):
        raise TypeError("streaming must be a boolean value.")
    if not isinstance(sigma, Real) or isinstance(sigma, bool) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive finite scalar.")
    if not isinstance(n_cc, Integral) or isinstance(n_cc, bool) or n_cc < 1:
        raise ValueError("n_cc must be a positive integer.")
    if not isinstance(max_iter, Integral) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not isinstance(tol, Real) or not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a positive finite scalar.")
    if not streaming:
        if not obj.kernel_matrices:
            raise ValueError(
                "Kernel matrices not found. Run compute_kernel_matrix() first, "
                "or call with streaming=True."
            )
        if sigma not in obj.sigma_values:
            raise ValueError(
                f"sigma = {sigma:g} not found in obj.sigma_values. "
                f"Available: {obj.sigma_values}."
            )
        if distance_args or kernel_args:
            warnings.warn(
                "distance_args / kernel_args are ignored when streaming=False.",
                UserWarning,
                stacklevel=2,
            )

    cell_types = list(obj.cell_types_of_interest)
    if not cell_types and obj.cell_types_sub is not None:
        cell_types = list(dict.fromkeys(np.asarray(obj.cell_types_sub).tolist()))
    if len(cell_types) < 2:
        raise ValueError(
            "run_gene_space_cca requires at least 2 cell types. Found: "
            + ", ".join(cell_types)
        )
    slides = [str(slide) for slide in obj.slide_list]

    if verbose:
        print("=== Gene-Space CCA ===")
        print("Step 1: Preparing gene-space data...")
    prepared = _prepare_gene_space_data(
        obj, clip=clip, min_prevalence=min_prevalence, min_cells=min_cells,
        cell_types=cell_types, slides=slides, materialize_z=not streaming,
    )
    if n_cc > len(prepared["genes"]):
        raise ValueError(
            f"n_cc ({n_cc}) exceeds number of genes after filtering "
            f"({len(prepared['genes'])})."
        )
    if verbose:
        print(
            f"  Genes: {len(prepared['genes'])}, "
            f"Slides: {len(prepared['slides'])}, "
            f"Cell types: {', '.join(cell_types)}"
        )

    if streaming:
        if verbose:
            print("Step 2: Streaming covariance precompute...")
        self_covariances, cross_covariances = _streaming_covariances(
            obj, prepared, sigma, prepared["slides"],
            cell_types, distance_args=distance_args, kernel_args=kernel_args,
            verbose=verbose
        )
        if sigma not in obj.sigma_values:
            obj.sigma_values.append(sigma)
    else:
        if verbose:
            print("Step 2: Precomputing covariance matrices...")
        self_covariances, cross_covariances = _precompute_covariances(
            prepared["z_by_slide"], obj.kernel_matrices, sigma,
            prepared["slides"], cell_types
        )

    if verbose:
        print("Step 3: Power iteration for canonical components...")
        print("  Finding CC 1 ...")
    rng = _as_rng(random_state)
    weights = optimize_genespace_avg_corr(
        self_covariances, cross_covariances, prepared["slides"], cell_types,
        max_iter=max_iter, tol=tol, verbose=verbose, random_state=rng
    )
    if n_cc > 1:
        weights = optimize_genespace_avg_corr_n(
            self_covariances, cross_covariances, prepared["slides"],
            cell_types, weights, n_cc=n_cc, max_iter=max_iter, tol=tol,
            verbose=verbose, random_state=rng
        )

    if verbose:
        print("Step 4: Storing results...")
    result = _store_results(
        obj, weights, prepared["z_by_slide"], sigma, cell_types, n_cc,
        prepared["genes"], prepared["gene_indices"], prepared["slides"],
        prepared=prepared,
    )
    if verbose:
        print("Done.")
    return result


__all__ = [
    "compute_genespace_objective",
    "optimize_genespace_avg_corr",
    "optimize_genespace_avg_corr_n",
    "run_gene_space_cca",
]
