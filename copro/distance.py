"""compute_distance() — pairwise Euclidean distances between cell types."""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cdist

from .core import CoProSingle


def _dist_flat_name(ct_i: str, ct_j: str) -> str:
    return f"dist|{ct_i}|{ct_j}"


def _store_self_distance_scale(obj, scaling: float) -> None:
    """Record the within-type (self) distance normalization factor.

    R parity: ``computeSelfDistance()`` (compute_self_distance_kernel.R) never
    writes ``@distanceScaleFactor`` — it applies the self scaling to the
    within-type matrices and leaves the CROSS-type factor set by
    ``computeDistance()`` untouched. That cross factor is what the sigma-aware
    permutation binning (``.recoverDistanceScaleFactor`` / ``.sigmaAwareBins``)
    relies on, because the cross-type permutation correlation is defined in
    CROSS-normalized units.

    We therefore store the self factor on a dedicated
    ``self_distance_scale_factor`` attribute and only populate
    ``distance_scale_factor`` when no cross-type factor has been recorded yet
    (i.e. ``compute_distance`` was never called). This preserves the cross
    value whenever it exists while keeping the standalone self path consistent
    with the sparse self-kernel path in ``kernel.py``.
    """
    obj.self_distance_scale_factor = scaling
    if getattr(obj, "distance_scale_factor", None) is None:
        obj.distance_scale_factor = scaling


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
        positive = dist_mat[np.isfinite(dist_mat) & (dist_mat > 0)]
        if len(positive) == 0:
            raise ValueError(
                "All spatial coordinates in a distance block are coincident; "
                "no positive distance is available."
            )
        min_nz = np.min(positive)
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
    normalize_target: float = 0.01,
    x_dist_scale: float = 1.0,
    y_dist_scale: float = 1.0,
    z_dist_scale: float = 1.0,
):
    """Compute pairwise Euclidean distance matrices between all cell-type pairs.

    Dispatches to multi-slide version for CoProMulti objects.

    For single-slide, 2+ types: pairs (ct_i, ct_j) stored under 'dist|ct_i|ct_j'.
    For single-slide, 1 type: within-type (ct, ct) stored under 'dist|ct|ct'.
    For multi-slide: keys include slide: 'dist|{slide}|ct_i|ct_j'.

    Normalization scales the low-distance percentile across all blocks to
    ``normalize_target`` (0.01 by default).
    """
    _validate_normalize_target(normalize_target)
    from .core import CoProMulti
    if isinstance(obj, CoProMulti):
        return _compute_distance_multi(
            obj, dist_type, normalize, truncate, normalize_target,
            x_dist_scale, y_dist_scale, z_dist_scale,
        )

    # --- Single-slide path ---
    cts = obj.cell_types_of_interest
    if not cts:
        raise ValueError("No cell types of interest. Run subset_data() first.")

    coord_cols, coord_scales = _distance_coordinate_spec(
        dist_type, x_dist_scale, y_dist_scale, z_dist_scale
    )

    loc = obj.location_data_sub
    if not set(coord_cols).issubset(loc.columns):
        raise ValueError(
            f"location_data_sub must have columns {coord_cols!r} for {dist_type}."
        )

    distances = {}

    if len(cts) == 1:
        # Within-type only
        ct = cts[0]
        mask = obj.cell_types_sub == ct
        coords = loc.loc[mask, coord_cols].values.astype(float) * coord_scales
        dist_mat = cdist(coords, coords)
        dist_mat, dist_percentile = _process_distance_matrix(
            dist_mat, truncate, percentile_choice=1e-4, set_diag_inf=True
        )
        flat_name = _dist_flat_name(ct, ct)
        distances[flat_name] = dist_mat

        if normalize:
            scaling_factor = normalize_target / dist_percentile
            distances[flat_name] = dist_mat * scaling_factor
            obj.distance_scale_factor = scaling_factor
        else:
            obj.distance_scale_factor = 1.0

    else:
        # Between-type pairs
        pairs = list(combinations(cts, 2))
        dist_percentiles = []
        raw_mats = {}

        for ct_i, ct_j in pairs:
            mask_i = obj.cell_types_sub == ct_i
            mask_j = obj.cell_types_sub == ct_j
            coords_i = loc.loc[mask_i, coord_cols].values.astype(float) * coord_scales
            coords_j = loc.loc[mask_j, coord_cols].values.astype(float) * coord_scales

            dist_mat = cdist(coords_i, coords_j)
            dist_mat, dist_pct = _process_distance_matrix(dist_mat, truncate)
            dist_percentiles.append(dist_pct)

            flat_name = _dist_flat_name(ct_i, ct_j)
            raw_mats[flat_name] = dist_mat

        if normalize:
            min_percentile = min(dist_percentiles)
            scaling_factor = normalize_target / min_percentile
            for flat_name, dist_mat in raw_mats.items():
                distances[flat_name] = dist_mat * scaling_factor
            obj.distance_scale_factor = scaling_factor
        else:
            distances = raw_mats
            obj.distance_scale_factor = 1.0

    obj.distances = distances
    return obj


def compute_self_distance(
    obj,
    dist_type: str = "Euclidean2D",
    normalize: bool = True,
    truncate: bool = True,
    verbose: bool = True,
    normalize_target: float = 0.01,
    x_dist_scale: float = 1.0,
    y_dist_scale: float = 1.0,
    z_dist_scale: float = 1.0,
):
    """Compute within-cell-type distance matrices when multiple cell types are present.

    Adds self-distance entries (``dist|A|A``, ``dist|B|B``, …) to
    ``obj.distances`` without overwriting existing cross-type distances.
    This is only needed when there are 2+ cell types — with a single
    cell type, ``compute_distance`` already computes the self-distance.

    Dispatches to multi-slide version for CoProMulti objects.
    """
    _validate_normalize_target(normalize_target)
    from .core import CoProMulti
    if isinstance(obj, CoProMulti):
        return _compute_self_distance_multi(
            obj, dist_type, normalize, truncate, verbose, normalize_target,
            x_dist_scale, y_dist_scale, z_dist_scale,
        )

    cts = obj.cell_types_of_interest
    if not cts:
        raise ValueError("No cell types of interest. Run subset_data() first.")
    if len(cts) == 1:
        warnings.warn("Only one cell type — use compute_distance() instead.")
        return obj

    coord_cols, coord_scales = _distance_coordinate_spec(
        dist_type, x_dist_scale, y_dist_scale, z_dist_scale
    )

    loc = obj.location_data_sub
    all_percentiles = []
    raw_mats = {}

    for ct in cts:
        mask = obj.cell_types_sub == ct
        coords = loc.loc[mask, coord_cols].values.astype(float) * coord_scales
        if len(coords) <= 5:
            if verbose:
                print(f"Skipping self-distance for {ct}: only {len(coords)} cells.")
            continue
        dist_mat = cdist(coords, coords)
        dist_mat, pct = _process_distance_matrix(
            dist_mat, truncate, percentile_choice=1e-4, set_diag_inf=True,
        )
        raw_mats[_dist_flat_name(ct, ct)] = dist_mat
        all_percentiles.append(pct)
        if verbose:
            print(f"Self-distance for {ct}: {len(coords)} cells")

    if normalize and all_percentiles:
        min_pct = min(all_percentiles)
        scaling = normalize_target / min_pct
        _store_self_distance_scale(obj, scaling)
        if verbose:
            print(f"Self-distance scaling factor: {scaling:.4f}")
        for k, v in raw_mats.items():
            obj.distances[k] = v * scaling
    else:
        _store_self_distance_scale(obj, 1.0)
        for k, v in raw_mats.items():
            obj.distances[k] = v

    return obj


def _compute_self_distance_multi(
    obj, dist_type, normalize, truncate, verbose, normalize_target,
    x_dist_scale, y_dist_scale, z_dist_scale,
):
    """Multi-slide self-distance. Adds 'dist|{slide}|ct|ct' entries."""
    cts = obj.cell_types_of_interest
    slides = obj.slide_list
    slide_ids = obj.meta_data_sub["slideID"].values
    loc = obj.location_data_sub
    coord_cols, coord_scales = _distance_coordinate_spec(
        dist_type, x_dist_scale, y_dist_scale, z_dist_scale
    )

    if len(cts) == 1:
        warnings.warn("Only one cell type — use compute_distance() instead.")
        return obj

    all_percentiles = []
    raw_mats = {}

    for ct in cts:
        for slide in slides:
            mask = (obj.cell_types_sub == ct) & (slide_ids == slide)
            n = int(mask.sum())
            if n <= 5:
                if verbose:
                    print(f"Skipping self-distance for {ct} in {slide}: {n} cells.")
                continue
            coords = loc.loc[mask, coord_cols].values.astype(float) * coord_scales
            dist_mat = cdist(coords, coords)
            dist_mat, pct = _process_distance_matrix(
                dist_mat, truncate, percentile_choice=1e-4, set_diag_inf=True,
            )
            flat_name = f"dist|{slide}|{ct}|{ct}"
            raw_mats[flat_name] = dist_mat
            all_percentiles.append(pct)

    if normalize and all_percentiles:
        min_pct = min(all_percentiles)
        scaling = normalize_target / min_pct
        _store_self_distance_scale(obj, scaling)
        if verbose:
            print(f"Global self-distance scaling factor: {scaling:.4f}")
        for k, v in raw_mats.items():
            obj.distances[k] = v * scaling
    else:
        _store_self_distance_scale(obj, 1.0)
        for k, v in raw_mats.items():
            obj.distances[k] = v

    return obj


def _compute_distance_multi(
    obj, dist_type="Euclidean2D", normalize=True, truncate=True,
    normalize_target=0.01, x_dist_scale=1.0, y_dist_scale=1.0,
    z_dist_scale=1.0,
):
    """Multi-slide distance computation. Keys: 'dist|{slide}|ct_i|ct_j'."""
    cts = obj.cell_types_of_interest
    slides = obj.slide_list
    slide_ids = obj.meta_data_sub["slideID"].values
    loc = obj.location_data_sub
    coord_cols, coord_scales = _distance_coordinate_spec(
        dist_type, x_dist_scale, y_dist_scale, z_dist_scale
    )

    if not set(coord_cols).issubset(loc.columns):
        raise ValueError(
            f"location_data_sub must have columns {coord_cols!r} for {dist_type}."
        )

    distances = {}
    all_percentiles = []
    raw_mats = {}

    if len(cts) == 1:
        ct = cts[0]
        for slide in slides:
            slide_ct_mask = (obj.cell_types_sub == ct) & (slide_ids == slide)
            if np.sum(slide_ct_mask) <= 5:
                continue
            coords = loc.loc[slide_ct_mask, coord_cols].values.astype(float) * coord_scales
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
                coords_i = loc.loc[mask_i, coord_cols].values.astype(float) * coord_scales
                coords_j = loc.loc[mask_j, coord_cols].values.astype(float) * coord_scales
                dist_mat = cdist(coords_i, coords_j)
                dist_mat, pct = _process_distance_matrix(dist_mat, truncate)
                flat_name = f"dist|{slide}|{ct_i}|{ct_j}"
                raw_mats[flat_name] = dist_mat
                all_percentiles.append(pct)

    if normalize and all_percentiles:
        global_min = min(all_percentiles)
        scaling_factor = normalize_target / global_min
        obj.distance_scale_factor = scaling_factor
        for k, v in raw_mats.items():
            distances[k] = v * scaling_factor
    else:
        distances = raw_mats
        obj.distance_scale_factor = 1.0

    obj.distances = distances
    return obj


def _distance_coordinate_spec(
    dist_type: str,
    x_dist_scale: float,
    y_dist_scale: float,
    z_dist_scale: float,
) -> tuple[list[str], np.ndarray]:
    """Validate a Euclidean distance mode and return columns/axis scales."""
    if dist_type not in {"Euclidean2D", "Euclidean3D"}:
        raise NotImplementedError(
            f"dist_type '{dist_type}' is not implemented; use Euclidean2D or Euclidean3D."
        )
    scales = [x_dist_scale, y_dist_scale]
    cols = ["x", "y"]
    if dist_type == "Euclidean3D":
        cols.append("z")
        scales.append(z_dist_scale)
    scales_arr = np.asarray(scales, dtype=float)
    if not np.all(np.isfinite(scales_arr)) or np.any(scales_arr <= 0):
        raise ValueError("Distance scale factors must be finite and positive.")
    return cols, scales_arr


def _validate_normalize_target(value: float) -> None:
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError("normalize_target must be a finite positive scalar.")
