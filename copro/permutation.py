"""Permutation inference for spatial kernel-restricted CCA.

The legacy fixed-sigma workflow remains available through
``run_skr_cca_permu`` followed by ``compute_normalized_correlation_permu``.
Modern workflows add fair selection over kernel bandwidths and a conditional
step-down test over canonical axes.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from .core import CoProSingle
from .correlation import _get_kernel_for_pair, _kernel_normalizer
from .optimization import (
    _apply_deflation,
    _bilinear_from_Y_resi,
    _compute_Y_resi,
    _initialize_next_component,
    _solve_two_type_svd,
    optimize_bilinear,
    optimize_bilinear_n,
)
from .skrcca import _prepare_pc_matrices


# ---------------------------------------------------------------------------
# Spatial resampling
# ---------------------------------------------------------------------------


def _prepare_spatial_resampling(
    location_data: pd.DataFrame,
    num_bins_x: int = 10,
    num_bins_y: int = 10,
) -> dict:
    """Cache bin membership and neighbor lookups for repeated draws."""
    if not isinstance(location_data, pd.DataFrame):
        location_data = pd.DataFrame(location_data)
    if not {"x", "y"}.issubset(location_data.columns):
        raise ValueError("location_data must contain 'x' and 'y' columns.")
    if int(num_bins_x) < 2 or int(num_bins_y) < 2:
        raise ValueError("num_bins_x and num_bins_y must be at least 2.")
    if len(location_data) == 0:
        raise ValueError("location_data must contain at least one row.")

    num_bins_x = int(num_bins_x)
    num_bins_y = int(num_bins_y)
    x = location_data["x"].to_numpy(dtype=float)
    y = location_data["y"].to_numpy(dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Spatial coordinates must be finite.")

    x_bin = np.asarray(pd.cut(x, bins=num_bins_x, labels=False), dtype=float)
    y_bin = np.asarray(pd.cut(y, bins=num_bins_y, labels=False), dtype=float)
    x_bin = np.where(np.isnan(x_bin), 0, x_bin).astype(int)
    y_bin = np.where(np.isnan(y_bin), 0, y_bin).astype(int)
    bin_id = x_bin * num_bins_y + y_bin
    unique_bins = np.unique(bin_id)
    members = {b: np.flatnonzero(bin_id == b) for b in unique_bins}

    neighbors = {}
    for b in unique_bins:
        bx, by = divmod(int(b), num_bins_y)
        adjacent = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = bx + dx, by + dy
                if 0 <= nx < num_bins_x and 0 <= ny < num_bins_y:
                    adjacent.append(nx * num_bins_y + ny)
        neighbors[b] = tuple(adjacent)

    return {
        "x": x,
        "y": y,
        "bin_id": bin_id,
        "unique_bins": unique_bins,
        "members": members,
        "neighbors": neighbors,
        "num_bins_x": num_bins_x,
        "num_bins_y": num_bins_y,
        "n_cells": len(location_data),
    }


def _draw_spatial_permutation(
    prepared: dict,
    match_quantile: bool = False,
    rng=None,
) -> np.ndarray:
    """Draw one index permutation from a prepared spatial resampling setup."""
    rng = np.random if rng is None else rng
    unique_bins = prepared["unique_bins"]
    shuffled = rng.permutation(unique_bins)
    mapping = dict(zip(unique_bins, shuffled))
    members = prepared["members"]
    perm = np.empty(prepared["n_cells"], dtype=int)

    for original_bin in unique_bins:
        target_bin = mapping[original_bin]
        original_idx = members[original_bin]
        candidate_parts = [members[target_bin]]

        if len(candidate_parts[0]) < len(original_idx):
            for neighbor in prepared["neighbors"][target_bin]:
                if neighbor == target_bin or neighbor not in members:
                    continue
                candidate_parts.append(members[neighbor])
                if sum(len(part) for part in candidate_parts) >= len(original_idx):
                    break

        candidates = np.concatenate(candidate_parts)
        if match_quantile and len(candidates) >= len(original_idx):
            selected_local = _match_by_quantile(
                prepared["x"][original_idx],
                prepared["y"][original_idx],
                prepared["x"][candidates],
                prepared["y"][candidates],
                len(original_idx),
            )
        else:
            selected_local = rng.choice(
                len(candidates),
                size=len(original_idx),
                replace=len(candidates) < len(original_idx),
            )
        perm[original_idx] = candidates[selected_local]

    return perm


def resample_spatial(
    location_data: pd.DataFrame,
    num_bins_x: int = 10,
    num_bins_y: int = 10,
    match_quantile: bool = False,
) -> np.ndarray:
    """Draw a bin-wise spatial resampling as zero-based row indices."""
    prepared = _prepare_spatial_resampling(
        location_data, num_bins_x=num_bins_x, num_bins_y=num_bins_y
    )
    return _draw_spatial_permutation(prepared, match_quantile=match_quantile)


def _match_by_quantile(orig_x, orig_y, cand_x, cand_y, n_points):
    """Greedily match cells by their within-tile quantile positions."""
    from scipy.stats import rankdata

    n_cand = len(cand_x)
    if len(orig_x) == 1:
        oqx, oqy = np.array([0.5]), np.array([0.5])
    else:
        oqx = rankdata(orig_x, method="average") / (len(orig_x) + 1)
        oqy = rankdata(orig_y, method="average") / (len(orig_y) + 1)

    if n_cand == 1:
        cqx, cqy = np.array([0.5]), np.array([0.5])
    else:
        cqx = rankdata(cand_x, method="average") / (n_cand + 1)
        cqy = rankdata(cand_y, method="average") / (n_cand + 1)

    dist = (oqx[:, None] - cqx[None, :]) ** 2 + (
        oqy[:, None] - cqy[None, :]
    ) ** 2
    sampled = np.empty(n_points, dtype=int)
    for index in range(n_points):
        best = int(np.argmin(dist[index]))
        sampled[index] = best
        dist[:, best] = np.inf
    return sampled


def generate_toroidal_permutations(
    x: np.ndarray,
    y: np.ndarray,
    n_permu: int,
    rng=None,
) -> np.ndarray:
    """Generate toroidal-shift permutation indices."""
    rng = np.random if rng is None else rng
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_width = x_max - x_min
    y_width = y_max - y_min

    if x_width < 1e-10 or y_width < 1e-10:
        warnings.warn("Spatial extent very small; toroidal shift may not work well.")
        return np.tile(np.arange(n), (n_permu, 1)).T

    original_order = np.lexsort((y, x))
    inverse_original = np.empty(n, dtype=int)
    inverse_original[original_order] = np.arange(n)
    permutations = np.empty((n, n_permu), dtype=int)
    for draw in range(n_permu):
        shift_x = rng.uniform(x_width * 0.1, x_width * 0.9)
        shift_y = rng.uniform(y_width * 0.1, y_width * 0.9)
        new_x = ((x - x_min + shift_x) % x_width) + x_min
        new_y = ((y - y_min + shift_y) % y_width) + y_min
        new_order = np.lexsort((new_y, new_x))
        permutations[:, draw] = new_order[inverse_original]
    return permutations


def _get_cell_permu(
    obj: CoProSingle,
    permu_method: str,
    n_permu: int,
    cts: list,
    permu_which: str,
    num_bins_x: int,
    num_bins_y: int,
    match_quantile: bool,
    rng=None,
) -> dict:
    """Generate or configure all permutation draws for each cell type."""
    if permu_method not in {"bin", "global", "toroidal", "pc"}:
        raise ValueError(
            "permu_method must be 'bin', 'global', 'toroidal', or 'pc'."
        )
    if permu_which not in {"second_only", "both", "first_only"}:
        raise ValueError(
            "permu_which must be 'second_only', 'both', or 'first_only'."
        )
    rng = np.random if rng is None else rng

    def should_permute(index: int) -> bool:
        if permu_which == "second_only":
            return index > 0
        if permu_which == "first_only":
            return index == 0
        return True

    cell_permu = {}
    for index, ct in enumerate(cts):
        mask = obj.cell_types_sub == ct
        n_cell = int(mask.sum())
        if not should_permute(index):
            cell_permu[ct] = np.tile(np.arange(n_cell), (n_permu, 1)).T
            continue

        if permu_method == "global":
            values = np.empty((n_cell, n_permu), dtype=int)
            for draw in range(n_permu):
                values[:, draw] = rng.permutation(n_cell)
            cell_permu[ct] = values
        elif permu_method == "bin":
            loc = obj.location_data_sub.loc[mask].reset_index(drop=True)
            prepared = _prepare_spatial_resampling(
                loc, num_bins_x=num_bins_x, num_bins_y=num_bins_y
            )
            values = np.empty((n_cell, n_permu), dtype=int)
            for draw in range(n_permu):
                values[:, draw] = _draw_spatial_permutation(
                    prepared, match_quantile=match_quantile, rng=rng
                )
            cell_permu[ct] = values
        elif permu_method == "toroidal":
            loc = obj.location_data_sub.loc[mask]
            cell_permu[ct] = generate_toroidal_permutations(
                loc["x"].to_numpy(dtype=float),
                loc["y"].to_numpy(dtype=float),
                n_permu,
                rng=rng,
            )
        else:
            cell_permu[ct] = {
                "type": "pc_permute",
                "seeds": rng.randint(0, 2**31 - 1, size=n_permu),
                "n_cell": n_cell,
            }
    return cell_permu


def _permute_pc_matrix(pc_mat: np.ndarray, seed: int) -> np.ndarray:
    """Shuffle values independently within each PC column."""
    rng = np.random.RandomState(seed)
    out = np.asarray(pc_mat).copy()
    for column in range(out.shape[1]):
        rng.shuffle(out[:, column])
    return out


def _permuted_pc_matrices(
    pc_mats: dict,
    cell_permu: dict,
    draw: int,
) -> dict:
    result = {}
    for ct, matrix in pc_mats.items():
        permutation = cell_permu[ct]
        if isinstance(permutation, dict) and permutation.get("type") == "pc_permute":
            result[ct] = _permute_pc_matrix(
                matrix, int(permutation["seeds"][draw])
            )
        else:
            result[ct] = matrix[permutation[:, draw]]
    return result


# ---------------------------------------------------------------------------
# Sigma-aware bin sizing
# ---------------------------------------------------------------------------


def _recover_distance_scale_factor(obj) -> float:
    """Return the CROSS-type mapping from raw to normalized spatial distance.

    This deliberately reads ``obj.distance_scale_factor`` (the CROSS-type factor
    set by :func:`compute_distance`), never ``self_distance_scale_factor``. The
    cross-type permutation correlation — and therefore the sigma-aware patch
    grid in :func:`_sigma_aware_bins` — is defined in CROSS-normalized units, so
    binning must use the cross factor. This matches R's
    ``.recoverDistanceScaleFactor`` (C_resampling_function.R), which reads
    ``@distanceScaleFactor``; R's ``computeSelfDistance`` never overwrites that
    slot with the within-type factor.
    """
    scale = getattr(obj, "distance_scale_factor", None)
    if scale is None:
        return np.nan
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        return np.nan
    return scale if np.isfinite(scale) and scale > 0 else np.nan


def _sigma_aware_bins(
    obj,
    sigma: float,
    min_bins: int = 2,
    verbose: bool = True,
) -> dict:
    """Choose a spatial patch grid whose raw side is about ``2 * sigma``."""
    loc = obj.location_data_sub
    extent_x = float(loc["x"].max() - loc["x"].min())
    extent_y = float(loc["y"].max() - loc["y"].min())
    scale = _recover_distance_scale_factor(obj)
    sigma = float(sigma)

    valid = (
        np.isfinite(scale)
        and scale > 0
        and np.isfinite(sigma)
        and sigma > 0
        and np.isfinite(extent_x)
        and np.isfinite(extent_y)
        and extent_x > 0
        and extent_y > 0
    )
    if not valid:
        warnings.warn(
            "sigma-aware bins: could not recover distance scale; falling back "
            "to a 10 x 10 grid. Pass num_bins_x/num_bins_y explicitly to override."
        )
        return {"num_bins_x": 10, "num_bins_y": 10, "scale_factor": np.nan}

    patch_raw = (2.0 * sigma) / scale
    num_bins_x = max(int(min_bins), int(np.floor(extent_x / patch_raw)))
    num_bins_y = max(int(min_bins), int(np.floor(extent_y / patch_raw)))
    side_x = (extent_x / num_bins_x) * scale
    side_y = (extent_y / num_bins_y) * scale

    for axis, side in (("x", side_x), ("y", side_y)):
        if side < sigma:
            warnings.warn(
                f"sigma-aware bins: {axis} patch side ({side:.3g}) < sigma "
                f"({sigma:.3g}); patches may over-shuffle within-type structure."
            )
        elif side > 4 * sigma:
            warnings.warn(
                f"sigma-aware bins: {axis} patch side ({side:.3g}) > 4*sigma "
                f"({4 * sigma:.3g}); patches may be too large."
            )

    if verbose:
        print(
            f"sigma-aware bins (sigma={sigma:g}): {num_bins_x} x {num_bins_y} "
            f"(normalized patch side ~ {side_x:.3g} x {side_y:.3g})"
        )
    return {
        "num_bins_x": num_bins_x,
        "num_bins_y": num_bins_y,
        "scale_factor": scale,
    }


def _resolve_bins(
    obj,
    sigma: float,
    permu_method: str,
    num_bins_x: int | None,
    num_bins_y: int | None,
    verbose: bool,
) -> tuple[int | None, int | None]:
    if permu_method == "bin" and (num_bins_x is None or num_bins_y is None):
        bins = _sigma_aware_bins(obj, sigma=sigma, verbose=verbose)
        return bins["num_bins_x"], bins["num_bins_y"]
    return num_bins_x, num_bins_y


# ---------------------------------------------------------------------------
# Shared scoring and conditional-axis kernels
# ---------------------------------------------------------------------------


def _cell_type_pairs(cts: list) -> list[tuple[str, str]]:
    if len(cts) == 1:
        return [(cts[0], cts[0])]
    return list(combinations(cts, 2))


def _get_ncorr_kernel_info(flat_kernels: dict, sigma: float, cts: list) -> dict:
    """Cache the cross-kernel and whitened-Frobenius normalizer for a sigma."""
    if len(cts) > 2:
        raise ValueError(
            "Quick permutation scoring supports one predeclared cell-type pair."
        )
    ct1 = cts[0]
    ct2 = cts[0] if len(cts) == 1 else cts[1]
    return {
        "K": _get_kernel_for_pair(flat_kernels, sigma, ct1, ct2),
        "normalizer": _kernel_normalizer(flat_kernels, sigma, ct1, ct2),
    }


def _compute_ncorr_quick(
    pc_mats: dict,
    w_dict: dict,
    flat_kernels: dict,
    sigma: float,
    cts: list,
    kernel_info: dict | None = None,
    y_resi: dict | None = None,
) -> float:
    """Score one axis for one predeclared cell-type pair."""
    if len(cts) > 2:
        raise ValueError(
            "Quick permutation scoring supports one predeclared cell-type pair."
        )
    ct1 = cts[0]
    ct2 = cts[0] if len(cts) == 1 else cts[1]
    A = pc_mats[ct1]
    B = pc_mats[ct2]
    w1 = np.asarray(w_dict[ct1])[:, :1]
    w2 = np.asarray(w_dict[ct2])[:, :1]
    score1 = A @ w1
    score2 = B @ w2
    if kernel_info is None:
        kernel_info = _get_ncorr_kernel_info(flat_kernels, sigma, cts)

    Y12 = None
    if y_resi is not None:
        Y12 = y_resi.get(ct1, {}).get(ct2)
    if Y12 is None:
        numerator = float((score1.T @ kernel_info["K"] @ score2).item())
    else:
        numerator = float((w1.T @ Y12 @ w2).item())

    denominator = (
        float(np.linalg.norm(score1))
        * float(np.linalg.norm(score2))
        * float(kernel_info["normalizer"])
    )
    return 0.0 if abs(denominator) < 1e-12 else numerator / denominator


def _copy_y_resi(y_resi: dict) -> dict:
    return {
        ct_i: {ct_j: np.asarray(value).copy() for ct_j, value in values.items()}
        for ct_i, values in y_resi.items()
    }


def _fit_conditional_axis(
    pc_mats: dict,
    flat_kernels: dict,
    sigma: float,
    cts: list,
    w_lower: dict | None = None,
    k_minus_1: int = 0,
    y_resi: dict | None = None,
    kernel_info: dict | None = None,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> dict:
    """Fit the leading residual axis after fixed observed-axis deflation."""
    if y_resi is None:
        y_resi = _compute_Y_resi(pc_mats, flat_kernels, sigma, cts)

    if k_minus_1 <= 0:
        if len(cts) == 2:
            weights = _solve_two_type_svd(y_resi, cts, n_cc=1)
        else:
            initial = _initialize_next_component(_copy_y_resi(y_resi), cts)
            weights = _bilinear_from_Y_resi(
                initial,
                _copy_y_resi(y_resi),
                pc_mats[cts[0]].shape[1],
                max_iter,
                tol,
                verbose=False,
            )
    else:
        if w_lower is None:
            raise ValueError("w_lower is required when k_minus_1 is positive.")
        y_deflated = _copy_y_resi(y_resi)
        for lower_axis in range(k_minus_1):
            y_deflated = _apply_deflation(
                y_deflated, w_lower, lower_axis, cts, method="rank1"
            )
        if len(cts) == 2:
            weights = _solve_two_type_svd(y_deflated, cts, n_cc=1)
        else:
            initial = _initialize_next_component(y_deflated, cts)
            weights = _bilinear_from_Y_resi(
                initial,
                y_deflated,
                pc_mats[cts[0]].shape[1],
                max_iter,
                tol,
                verbose=False,
            )

    weights = {ct: np.asarray(weights[ct])[:, :1].copy() for ct in cts}
    ncorr = _compute_ncorr_quick(
        pc_mats,
        weights,
        flat_kernels,
        sigma,
        cts,
        kernel_info=kernel_info,
        y_resi=y_resi,
    )
    return {"weights": weights, "ncorr": ncorr}


def _validate_permutation_request(
    obj,
    n_permu: int,
    permu_method: str,
    permu_which: str,
    require_ncorr: bool = False,
) -> list:
    if int(n_permu) < 1:
        raise ValueError("n_permu must be at least 1.")
    if permu_method not in {"bin", "global", "pc", "toroidal"}:
        raise ValueError(
            "permu_method must be 'bin', 'global', 'pc', or 'toroidal'."
        )
    if permu_which not in {"second_only", "both", "first_only"}:
        raise ValueError(
            "permu_which must be 'second_only', 'both', or 'first_only'."
        )
    if not getattr(obj, "skr_cca_out", None):
        raise ValueError("Run run_skr_cca() before permutation testing.")
    if require_ncorr and not getattr(obj, "normalized_correlation", None):
        raise ValueError("Run compute_normalized_correlation() first.")
    cts = list(getattr(obj, "cell_types_of_interest", []))
    if not cts:
        raise ValueError("cell_types_of_interest is empty.")
    return cts


def _valid_sigma_values(obj, sigma_values=None, require_weights: bool = False) -> list:
    requested = list(obj.sigma_values if sigma_values is None else sigma_values)
    available = list(getattr(obj, "sigma_values", []))
    valid = [sigma for sigma in requested if sigma in available]
    if require_weights:
        valid = [
            sigma
            for sigma in valid
            if obj.skr_cca_out.get(f"sigma_{sigma}") is not None
        ]
    if not valid:
        raise ValueError("No valid sigma values are available for permutation testing.")
    return valid


def _rng(seed: int | None):
    return np.random if seed is None else np.random.RandomState(seed)


# ---------------------------------------------------------------------------
# Legacy fixed-sigma workflow
# ---------------------------------------------------------------------------


def run_skr_cca_permu(
    obj: CoProSingle,
    n_permu: int = 20,
    permu_method: str = "bin",
    permu_which: str = "second_only",
    num_bins_x: int | None = None,
    num_bins_y: int | None = None,
    match_quantile: bool = False,
    conservative: bool = False,
    max_iter: int = 200,
    tol: float = 1e-5,
    seed: int | None = None,
    verbose: bool = True,
):
    """Run the compatible fixed-observed-sigma permutation workflow."""
    if conservative:
        permu_which = "second_only"
        num_bins_x = num_bins_y = 15
        match_quantile = True
    cts = _validate_permutation_request(
        obj, n_permu, permu_method, permu_which, require_ncorr=False
    )
    sigma = getattr(obj, "sigma_value_choice", None)
    if sigma is None:
        raise ValueError(
            "Run compute_normalized_correlation() first to set sigma_value_choice."
        )
    if sigma not in obj.sigma_values:
        raise ValueError("sigma_value_choice is not present in sigma_values.")

    if n_permu < 10:
        warnings.warn("n_permu < 10 may give unreliable p-values. Consider >= 100.")

    num_bins_x, num_bins_y = _resolve_bins(
        obj, sigma, permu_method, num_bins_x, num_bins_y, verbose
    )
    random = _rng(seed)
    cell_permu = _get_cell_permu(
        obj,
        permu_method,
        int(n_permu),
        cts,
        permu_which,
        num_bins_x,
        num_bins_y,
        match_quantile,
        rng=random,
    )
    obj.cell_permu = cell_permu
    obj.n_permu = int(n_permu)

    pc_mats = _prepare_pc_matrices(obj, getattr(obj, "scale_pcs", True), cts)
    results = {}
    for draw in range(int(n_permu)):
        pc_local = _permuted_pc_matrices(pc_mats, cell_permu, draw)
        weights = optimize_bilinear(
            pc_local,
            obj.kernel_matrices,
            sigma,
            max_iter=max_iter,
            tol=tol,
            verbose=False,
        )
        if obj.n_cc > 1:
            weights = optimize_bilinear_n(
                pc_local,
                obj.kernel_matrices,
                sigma,
                weights,
                cts,
                n_cc=obj.n_cc,
                max_iter=max_iter,
                tol=tol,
                verbose=False,
            )
        results[f"permu_{draw + 1}"] = weights
        if verbose and ((draw + 1) % 10 == 0 or draw + 1 == n_permu):
            print(f"  Completed {draw + 1} of {n_permu} permutations")
    obj.skr_cca_permu_out = results
    return obj


def compute_normalized_correlation_permu(
    obj: CoProSingle,
    tol: float = 1e-4,
    verbose: bool = True,
):
    """Score fixed-sigma permutation fits with whitened-Frobenius scaling."""
    del tol  # retained for API compatibility; the new normalizer has no SVD tolerance
    if not getattr(obj, "skr_cca_permu_out", None):
        raise ValueError("Run run_skr_cca_permu() first.")

    cts = obj.cell_types_of_interest
    sigma = obj.sigma_value_choice
    pairs = _cell_type_pairs(cts)
    pc_mats = _prepare_pc_matrices(obj, getattr(obj, "scale_pcs", True), cts)
    kernel_info = {}
    for ct1, ct2 in pairs:
        kernel_info[(ct1, ct2)] = {
            "K": _get_kernel_for_pair(obj.kernel_matrices, sigma, ct1, ct2),
            "normalizer": _kernel_normalizer(
                obj.kernel_matrices, sigma, ct1, ct2
            ),
        }

    results = {}
    items = list(obj.skr_cca_permu_out.items())
    for draw, (name, weights) in enumerate(items):
        pc_local = _permuted_pc_matrices(pc_mats, obj.cell_permu, draw)
        scores = {
            ct: pc_local[ct] @ np.asarray(weights[ct])[:, : obj.n_cc]
            for ct in cts
        }
        score_norms = {
            ct: np.linalg.norm(score, axis=0) for ct, score in scores.items()
        }
        rows = []
        for ct1, ct2 in pairs:
            info = kernel_info[(ct1, ct2)]
            numerators = np.sum(scores[ct1] * (info["K"] @ scores[ct2]), axis=0)
            denominators = (
                score_norms[ct1] * score_norms[ct2] * info["normalizer"]
            )
            values = np.divide(
                numerators,
                denominators,
                out=np.zeros_like(numerators, dtype=float),
                where=np.abs(denominators) >= 1e-12,
            )
            for cc, value in enumerate(values, start=1):
                rows.append(
                    {
                        "sigma": sigma,
                        "cell_type_1": ct1,
                        "cell_type_2": ct2,
                        "CC_index": cc,
                        "normalized_correlation": float(value),
                    }
                )
        results[name] = pd.DataFrame(rows)
        if verbose and ((draw + 1) % 20 == 0 or draw + 1 == len(items)):
            print(f"  Completed {draw + 1} of {len(items)} permutations")
    obj.normalized_correlation_permu = results
    return obj


# ---------------------------------------------------------------------------
# P-values
# ---------------------------------------------------------------------------


def calculate_pvalue(
    obj: CoProSingle,
    cc_index: int = 1,
    cell_type_1: str | None = None,
    cell_type_2: str | None = None,
    alternative: str = "greater",
) -> dict:
    """Calculate a Phipson-Smyth permutation p-value.

    The observed statistic and every null draw use the same maximum over all
    cell-type pairs. To keep the observed/null comparison fair, the observed
    maximum is taken only over the sigma bandwidths the null draws were actually
    scored at — the distinct ``sigma`` values present in
    ``normalized_correlation_permu``. For the legacy fixed-sigma null
    (:func:`compute_normalized_correlation_permu`) that is the single
    ``sigma_value_choice``; maxing the observed statistic over sigmas the null
    never used would bias the p-value downward (anti-conservative). For the
    fair-sigma / conditional nulls each draw already re-maximizes over every
    sigma, so their frames span the full sigma set and the restriction is a
    no-op, leaving those workflows unchanged. Passing both cell-type arguments
    retains the former pair-specific Python behavior.

    The Phipson-Smyth estimator ``(1 + #{null >= obs}) / (n + 1)`` itself is
    unchanged; only the sigma set over which ``obs`` is maximized is matched to
    the null.
    """
    if not getattr(obj, "normalized_correlation", None):
        raise ValueError("Run compute_normalized_correlation() first.")
    if not getattr(obj, "normalized_correlation_permu", None):
        raise ValueError("Run compute_normalized_correlation_permu() first.")
    positional_alternatives = {"greater", "less", "two.sided", "two_sided"}
    if (
        cell_type_1 in positional_alternatives
        and cell_type_2 is None
        and alternative == "greater"
    ):
        alternative = cell_type_1
        cell_type_1 = None
    if (cell_type_1 is None) != (cell_type_2 is None):
        raise ValueError("Specify both cell_type_1 and cell_type_2, or neither.")

    alternative = alternative.lower().replace("_", ".").replace("-", ".")
    if alternative not in {"greater", "less", "two.sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two.sided'.")

    null_frames = [
        frame
        for frame in obj.normalized_correlation_permu.values()
        if len(frame) > 0
    ]
    if not null_frames:
        raise ValueError("Permutation normalized correlation is empty.")

    # DEFECT-2 fix: restrict the observed maximum to exactly the sigma values
    # the null draws were scored at, so observed and null share the same sigma
    # set (see the docstring). ``None`` disables the restriction if the null
    # frames carry no sigma column (defensive; the pipeline always sets it).
    null_sigmas = None
    if all("sigma" in frame.columns for frame in null_frames):
        collected = set()
        for frame in null_frames:
            collected.update(frame["sigma"].to_numpy(dtype=float).tolist())
        if collected:
            null_sigmas = collected

    def statistic(frame: pd.DataFrame, restrict_sigmas=None) -> float:
        selected = frame[frame["CC_index"] == cc_index]
        if restrict_sigmas is not None and "sigma" in selected.columns:
            selected = selected[
                selected["sigma"].astype(float).isin(restrict_sigmas)
            ]
        if cell_type_1 is not None:
            selected = selected[
                (selected["cell_type_1"] == cell_type_1)
                & (selected["cell_type_2"] == cell_type_2)
            ]
        values = selected["normalized_correlation"].to_numpy(dtype=float)
        if len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError(
                "Normalized correlation is missing or non-finite for cc_index "
                "after matching the observed statistic to the null sigma set."
            )
        return float(np.max(values))

    observed_frames = [
        frame for frame in obj.normalized_correlation.values() if len(frame) > 0
    ]
    if not observed_frames:
        raise ValueError("Observed normalized correlation is empty.")
    observed = statistic(pd.concat(observed_frames, ignore_index=True), null_sigmas)
    permu_values = np.asarray(
        [statistic(frame, null_sigmas) for frame in null_frames],
        dtype=float,
    )
    n_permu = len(permu_values)

    p_greater = (1 + int(np.sum(permu_values >= observed))) / (n_permu + 1)
    p_less = (1 + int(np.sum(permu_values <= observed))) / (n_permu + 1)
    if alternative == "greater":
        p_value = p_greater
    elif alternative == "less":
        p_value = p_less
    else:
        p_value = min(2 * min(p_greater, p_less), 1.0)

    result = {
        "p_value": float(p_value),
        "mc_floor": 1.0 / (n_permu + 1),
        "observed": observed,
        "permu_mean": float(np.mean(permu_values)),
        "permu_sd": (
            float(np.std(permu_values, ddof=1)) if n_permu > 1 else np.nan
        ),
        "permu_values": permu_values,
        "n_permu": n_permu,
        "alternative": alternative,
        "pair_aggregation": "max" if cell_type_1 is None else "selected",
        "cell_type_1": cell_type_1,
        "cell_type_2": cell_type_2,
    }
    result["CC_index"] = cc_index
    return result


# ---------------------------------------------------------------------------
# Fair-sigma permutation inference
# ---------------------------------------------------------------------------


def run_skr_cca_permu_fair_sigma(
    obj: CoProSingle,
    n_permu: int = 100,
    sigma_values: list | None = None,
    permu_method: str = "bin",
    permu_which: str = "second_only",
    num_bins_x: int | None = None,
    num_bins_y: int | None = None,
    match_quantile: bool = False,
    max_iter: int = 200,
    tol: float = 1e-5,
    seed: int | None = None,
    verbose: bool = True,
):
    """Give each null draw the same max-over-sigma selection as observed data."""
    cts = _validate_permutation_request(
        obj, n_permu, permu_method, permu_which, require_ncorr=True
    )
    if len(cts) > 2:
        raise ValueError(
            "Fair-sigma permutation supports one predeclared cell-type pair."
        )
    sigma_values = _valid_sigma_values(obj, sigma_values)
    observed_sigma = obj.sigma_value_choice
    if observed_sigma is None:
        raise ValueError("sigma_value_choice is not set.")
    if obj.n_cc > 1:
        warnings.warn(
            "Fair-sigma permutation tests CC1 only; use the conditional test "
            "for higher canonical axes."
        )

    num_bins_x, num_bins_y = _resolve_bins(
        obj,
        observed_sigma,
        permu_method,
        num_bins_x,
        num_bins_y,
        verbose,
    )
    random = _rng(seed)
    cell_permu = _get_cell_permu(
        obj,
        permu_method,
        int(n_permu),
        cts,
        permu_which,
        num_bins_x,
        num_bins_y,
        match_quantile,
        rng=random,
    )
    pc_mats = _prepare_pc_matrices(obj, getattr(obj, "scale_pcs", True), cts)
    kernel_info = {
        sigma: _get_ncorr_kernel_info(obj.kernel_matrices, sigma, cts)
        for sigma in sigma_values
    }

    results = {}
    normalized = {}
    selected_sigmas = np.empty(int(n_permu), dtype=float)
    selected_stats = np.empty(int(n_permu), dtype=float)
    for draw in range(int(n_permu)):
        pc_local = _permuted_pc_matrices(pc_mats, cell_permu, draw)
        best_stat = -np.inf
        best_sigma = sigma_values[0]
        best_weights = None
        for sigma in sigma_values:
            y_resi = _compute_Y_resi(pc_local, obj.kernel_matrices, sigma, cts)
            fit = _fit_conditional_axis(
                pc_local,
                obj.kernel_matrices,
                sigma,
                cts,
                k_minus_1=0,
                y_resi=y_resi,
                kernel_info=kernel_info[sigma],
                max_iter=max_iter,
                tol=tol,
            )
            if np.isfinite(fit["ncorr"]) and fit["ncorr"] > best_stat:
                best_stat = fit["ncorr"]
                best_sigma = sigma
                best_weights = fit["weights"]

        if best_weights is None or not np.isfinite(best_stat):
            raise RuntimeError(
                f"All sigma fits failed for permutation {draw + 1}."
            )

        name = f"permu_{draw + 1}"
        results[name] = best_weights
        selected_sigmas[draw] = best_sigma
        selected_stats[draw] = best_stat
        ct1 = cts[0]
        ct2 = cts[0] if len(cts) == 1 else cts[1]
        normalized[name] = pd.DataFrame(
            [
                {
                    "sigma": best_sigma,
                    "cell_type_1": ct1,
                    "cell_type_2": ct2,
                    "CC_index": 1,
                    "normalized_correlation": best_stat,
                }
            ]
        )
        if verbose and ((draw + 1) % 10 == 0 or draw + 1 == n_permu):
            print(f"  Completed {draw + 1} of {n_permu} permutations")

    sigma_differs = selected_sigmas != float(observed_sigma)
    obj.cell_permu = cell_permu
    obj.skr_cca_permu_out = results
    obj.normalized_correlation_permu = normalized
    obj.n_permu = int(n_permu)
    obj.fair_sigma_permu = {
        "sigma_selected": selected_sigmas,
        "sigma_values_tested": np.asarray(sigma_values, dtype=float),
        "observed_best_sigma": observed_sigma,
        "sigma_differs": sigma_differs,
        "prop_sigma_differs": float(np.mean(sigma_differs)),
        "n_sigma_differs": int(np.sum(sigma_differs)),
        "permu_stats": selected_stats,
        "num_bins": {"x": num_bins_x, "y": num_bins_y},
    }
    return obj


# ---------------------------------------------------------------------------
# Conditional sequential step-down inference
# ---------------------------------------------------------------------------


def run_skr_cca_permu_conditional(
    obj: CoProSingle,
    n_permu: int = 100,
    sigma_values: list | None = None,
    permu_method: str = "bin",
    permu_which: str = "second_only",
    num_bins_x: int | None = None,
    num_bins_y: int | None = None,
    match_quantile: bool = False,
    alpha: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-5,
    seed: int | None = None,
    verbose: bool = True,
):
    """Run fair-sigma conditional tests with closed step-down axis control."""
    cts = _validate_permutation_request(
        obj, n_permu, permu_method, permu_which, require_ncorr=True
    )
    if len(cts) > 2:
        raise ValueError(
            "Conditional permutation supports one predeclared cell-type pair."
        )
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if obj.n_cc < 1:
        raise ValueError("n_cc must be at least 1.")
    if len(cts) == 1 and permu_which == "second_only":
        warnings.warn(
            "For one cell type, second_only is the identity permutation; "
            "use 'both' or 'first_only'."
        )

    sigma_values = _valid_sigma_values(
        obj, sigma_values=sigma_values, require_weights=True
    )
    observed_sigma_choice = obj.sigma_value_choice
    if observed_sigma_choice is None:
        raise ValueError("sigma_value_choice is not set.")
    num_bins_x, num_bins_y = _resolve_bins(
        obj,
        observed_sigma_choice,
        permu_method,
        num_bins_x,
        num_bins_y,
        verbose,
    )

    observed_stats = np.full(obj.n_cc, -np.inf, dtype=float)
    observed_sigmas = np.full(obj.n_cc, float(sigma_values[0]), dtype=float)
    observed_weights = {}
    for sigma in sigma_values:
        sigma_name = f"sigma_{sigma}"
        observed_weights[sigma] = obj.skr_cca_out[sigma_name]
        frame = obj.normalized_correlation.get(sigma_name)
        if frame is None or len(frame) == 0:
            continue
        for axis in range(obj.n_cc):
            values = frame.loc[
                frame["CC_index"] == axis + 1, "normalized_correlation"
            ].to_numpy(dtype=float)
            if len(values) and np.any(np.isfinite(values)):
                value = float(np.nanmax(values))
                if value > observed_stats[axis]:
                    observed_stats[axis] = value
                    observed_sigmas[axis] = sigma
    if not np.all(np.isfinite(observed_stats)):
        raise ValueError("Observed normalized correlations are missing for an axis.")

    random = _rng(seed)
    cell_permu = _get_cell_permu(
        obj,
        permu_method,
        int(n_permu),
        cts,
        permu_which,
        num_bins_x,
        num_bins_y,
        match_quantile,
        rng=random,
    )
    pc_mats = _prepare_pc_matrices(obj, getattr(obj, "scale_pcs", True), cts)
    kernel_info = {
        sigma: _get_ncorr_kernel_info(obj.kernel_matrices, sigma, cts)
        for sigma in sigma_values
    }

    permutation_stats = np.full((int(n_permu), obj.n_cc), np.nan, dtype=float)
    permutation_sigmas = np.full(
        (int(n_permu), obj.n_cc), float(sigma_values[0]), dtype=float
    )
    n_failed = 0
    for draw in range(int(n_permu)):
        pc_local = _permuted_pc_matrices(pc_mats, cell_permu, draw)
        try:
            draw_stats = np.full(obj.n_cc, -np.inf, dtype=float)
            draw_sigmas = np.full(obj.n_cc, float(sigma_values[0]), dtype=float)
            for sigma in sigma_values:
                y_resi = _compute_Y_resi(
                    pc_local, obj.kernel_matrices, sigma, cts
                )
                for axis in range(obj.n_cc):
                    fit = _fit_conditional_axis(
                        pc_local,
                        obj.kernel_matrices,
                        sigma,
                        cts,
                        w_lower=observed_weights[sigma],
                        k_minus_1=axis,
                        y_resi=y_resi,
                        kernel_info=kernel_info[sigma],
                        max_iter=max_iter,
                        tol=tol,
                    )
                    if np.isfinite(fit["ncorr"]) and fit["ncorr"] > draw_stats[axis]:
                        draw_stats[axis] = fit["ncorr"]
                        draw_sigmas[axis] = sigma
            if not np.all(np.isfinite(draw_stats)):
                raise RuntimeError("At least one canonical axis failed for all sigmas.")
            permutation_stats[draw] = draw_stats
            permutation_sigmas[draw] = draw_sigmas
        except Exception:
            n_failed += 1
        if verbose and ((draw + 1) % 10 == 0 or draw + 1 == n_permu):
            print(f"  Completed {draw + 1} of {n_permu} permutations")

    if n_failed:
        warnings.warn(
            f"{n_failed} of {n_permu} permutations failed and were dropped."
        )

    p_raw = np.full(obj.n_cc, np.nan, dtype=float)
    mc_floor = np.full(obj.n_cc, np.nan, dtype=float)
    for axis in range(obj.n_cc):
        null = permutation_stats[np.isfinite(permutation_stats[:, axis]), axis]
        if len(null):
            p_raw[axis] = (1 + int(np.sum(null >= observed_stats[axis]))) / (
                len(null) + 1
            )
            mc_floor[axis] = 1.0 / (len(null) + 1)

    p_stepdown = np.maximum.accumulate(np.where(np.isnan(p_raw), 1.0, p_raw))
    nonsignificant = np.flatnonzero(p_stepdown > alpha)
    n_significant = int(nonsignificant[0]) if len(nonsignificant) else obj.n_cc
    significant = np.arange(obj.n_cc) < n_significant
    per_axis = pd.DataFrame(
        {
            "CC_index": np.arange(1, obj.n_cc + 1),
            "observed_stat": observed_stats,
            "observed_sigma": observed_sigmas,
            "p_raw": p_raw,
            "p_stepdown": p_stepdown,
            "mc_floor": mc_floor,
            "significant": significant,
        }
    )

    obj.cell_permu = cell_permu
    obj.n_permu = int(n_permu)
    obj.conditional_permu = {
        "per_axis": per_axis,
        "n_significant_axes": n_significant,
        "alpha": float(alpha),
        "mc_floor": mc_floor,
        "obs_stats": observed_stats,
        "obs_sigma": observed_sigmas,
        "perm_stats": permutation_stats,
        "perm_sigma": permutation_sigmas,
        "sigma_values": np.asarray(sigma_values, dtype=float),
        "n_permu": int(n_permu),
        "n_failed": n_failed,
        "permu_method": permu_method,
        "permu_which": permu_which,
        "num_bins": {"x": num_bins_x, "y": num_bins_y},
    }
    return obj


def calculate_pvalue_stepdown(obj: CoProSingle) -> pd.DataFrame:
    """Return the per-axis table from conditional step-down inference."""
    conditional = getattr(obj, "conditional_permu", None)
    if not conditional:
        raise ValueError("Run run_skr_cca_permu_conditional() first.")
    result = conditional["per_axis"].copy()
    result.attrs.update(
        {
            "n_significant_axes": conditional["n_significant_axes"],
            "alpha": conditional["alpha"],
            "n_permu": conditional["n_permu"],
        }
    )
    return result
