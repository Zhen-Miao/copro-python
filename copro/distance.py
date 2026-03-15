"""compute_distance() — pairwise Euclidean distances between cell types."""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cdist

from .core import CoProSingle


def _dist_flat_name(ct_i: str, ct_j: str) -> str:
    return f"dist|{ct_i}|{ct_j}"


def _process_distance_matrix(
    dist_mat: np.ndarray,
    truncate: bool,
    percentile_choice: float | None = None,
    set_diag_inf: bool = False,
) -> tuple[np.ndarray, float]:
    """Process distance matrix: handle zeros, compute percentile, optionally truncate.

    Returns (processed_matrix, dist_percentile).
    """
    dist_mat = dist_mat.copy()

    if set_diag_inf:
        np.fill_diagonal(dist_mat, np.inf)

    # Replace zeros (overlapping cells) with smallest non-zero
    if np.any(dist_mat == 0):
        min_nz = np.min(dist_mat[dist_mat > 0]) if np.any(dist_mat > 0) else 1.0
        dist_mat[dist_mat == 0] = min_nz
        warnings.warn(
            "Zero distances detected; replaced with smallest non-zero distance."
        )

    # Choose percentile threshold
    finite_vals = dist_mat[np.isfinite(dist_mat) & (dist_mat > 0)]
    if len(finite_vals) == 0:
        raise ValueError("No finite non-zero distances found.")

    if percentile_choice is None:
        percentile_choice = min(1e-3, 2.0 / max(dist_mat.shape))

    dist_percentile = float(np.quantile(finite_vals, percentile_choice))

    if truncate:
        mask = (dist_mat < dist_percentile) & np.isfinite(dist_mat)
        dist_mat[mask] = dist_percentile

    return dist_mat, dist_percentile


def compute_distance(
    obj,
    dist_type: str = "Euclidean2D",
    normalize: bool = True,
    truncate: bool = True,
):
    """Compute pairwise Euclidean distance matrices between all cell-type pairs.

    Dispatches to multi-slide version for CoProMulti objects.

    For single-slide, 2+ types: pairs (ct_i, ct_j) stored under 'dist|ct_i|ct_j'.
    For single-slide, 1 type: within-type (ct, ct) stored under 'dist|ct|ct'.
    For multi-slide: keys include slide: 'dist|{slide}|ct_i|ct_j'.

    Normalization: scales so 0.001th-percentile distance (across all pairs) equals 0.01.
    """
    from .core import CoProMulti
    if isinstance(obj, CoProMulti):
        return _compute_distance_multi(obj, dist_type, normalize, truncate)

    # --- Single-slide path ---
    cts = obj.cell_types_of_interest
    if not cts:
        raise ValueError("No cell types of interest. Run subset_data() first.")

    if dist_type != "Euclidean2D":
        raise NotImplementedError(f"dist_type '{dist_type}' not implemented. Use 'Euclidean2D'.")

    loc = obj.location_data_sub
    if not {"x", "y"}.issubset(loc.columns):
        raise ValueError("location_data_sub must have columns 'x' and 'y'.")

    distances = {}

    if len(cts) == 1:
        # Within-type only
        ct = cts[0]
        mask = obj.cell_types_sub == ct
        coords = loc.loc[mask, ["x", "y"]].values.astype(float)
        dist_mat = cdist(coords, coords)
        dist_mat, dist_percentile = _process_distance_matrix(
            dist_mat, truncate, percentile_choice=1e-4, set_diag_inf=True
        )
        flat_name = _dist_flat_name(ct, ct)
        distances[flat_name] = dist_mat

        if normalize:
            scaling_factor = 0.01 / dist_percentile
            distances[flat_name] = dist_mat * scaling_factor

    else:
        # Between-type pairs
        pairs = list(combinations(cts, 2))
        dist_percentiles = []
        raw_mats = {}

        for ct_i, ct_j in pairs:
            mask_i = obj.cell_types_sub == ct_i
            mask_j = obj.cell_types_sub == ct_j
            coords_i = loc.loc[mask_i, ["x", "y"]].values.astype(float)
            coords_j = loc.loc[mask_j, ["x", "y"]].values.astype(float)

            dist_mat = cdist(coords_i, coords_j)
            dist_mat, dist_pct = _process_distance_matrix(dist_mat, truncate)
            dist_percentiles.append(dist_pct)

            flat_name = _dist_flat_name(ct_i, ct_j)
            raw_mats[flat_name] = dist_mat

        if normalize:
            min_percentile = min(dist_percentiles)
            scaling_factor = 0.01 / min_percentile
            for flat_name, dist_mat in raw_mats.items():
                distances[flat_name] = dist_mat * scaling_factor
        else:
            distances = raw_mats

    obj.distances = distances
    return obj


def _compute_distance_multi(obj, dist_type="Euclidean2D", normalize=True, truncate=True):
    """Multi-slide distance computation. Keys: 'dist|{slide}|ct_i|ct_j'."""
    cts = obj.cell_types_of_interest
    slides = obj.slide_list
    slide_ids = obj.meta_data_sub["slideID"].values
    loc = obj.location_data_sub

    if not {"x", "y"}.issubset(loc.columns):
        raise ValueError("location_data_sub must have columns 'x' and 'y'.")

    distances = {}
    all_percentiles = []
    raw_mats = {}

    if len(cts) == 1:
        ct = cts[0]
        for slide in slides:
            slide_ct_mask = (obj.cell_types_sub == ct) & (slide_ids == slide)
            if np.sum(slide_ct_mask) <= 5:
                continue
            coords = loc.loc[slide_ct_mask, ["x", "y"]].values.astype(float)
            dist_mat = cdist(coords, coords)
            dist_mat, pct = _process_distance_matrix(dist_mat, truncate, percentile_choice=1e-4, set_diag_inf=True)
            flat_name = f"dist|{slide}|{ct}|{ct}"
            raw_mats[flat_name] = dist_mat
            all_percentiles.append(pct)
    else:
        pairs = list(combinations(cts, 2))
        for slide in slides:
            for ct_i, ct_j in pairs:
                mask_i = (obj.cell_types_sub == ct_i) & (slide_ids == slide)
                mask_j = (obj.cell_types_sub == ct_j) & (slide_ids == slide)
                if np.sum(mask_i) <= 5 or np.sum(mask_j) <= 5:
                    continue
                coords_i = loc.loc[mask_i, ["x", "y"]].values.astype(float)
                coords_j = loc.loc[mask_j, ["x", "y"]].values.astype(float)
                dist_mat = cdist(coords_i, coords_j)
                dist_mat, pct = _process_distance_matrix(dist_mat, truncate)
                flat_name = f"dist|{slide}|{ct_i}|{ct_j}"
                raw_mats[flat_name] = dist_mat
                all_percentiles.append(pct)

    if normalize and all_percentiles:
        global_min = min(all_percentiles)
        scaling_factor = 0.01 / global_min
        for k, v in raw_mats.items():
            distances[k] = v * scaling_factor
    else:
        distances = raw_mats

    obj.distances = distances
    return obj
