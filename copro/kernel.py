"""Gaussian spatial kernels, including an exact sparse fixed-radius path."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def _kernel_flat_name(sigma: float, ct_i: str, ct_j: str) -> str:
    return f"kernel|sigma{sigma}|{ct_i}|{ct_j}"


def _dist_flat_name(ct_i: str, ct_j: str) -> str:
    return f"dist|{ct_i}|{ct_j}"


def _kernel_from_distance(
    dist_mat: np.ndarray, sigma: float, lower_limit: float
) -> np.ndarray:
    """Evaluate the Gaussian kernel and set values below its support floor to 0."""
    K = np.exp(-0.5 * (np.asarray(dist_mat, dtype=float) / sigma) ** 2)
    K[~np.isfinite(K) | (K < lower_limit)] = 0.0
    return K


def _process_kernel(
    K: np.ndarray,
    lower_limit: float,
    upper_quantile: float,
    row_normalize: bool = False,
    col_normalize: bool = False,
    normalize_kernel: bool = False,
) -> np.ndarray:
    """Clip and normalize a dense kernel without changing its storage type."""
    valid = K[np.isfinite(K) & (K >= lower_limit)]
    if len(valid) == 0:
        out = np.zeros_like(K, dtype=float)
        return out

    upper_clip = float(np.quantile(valid, upper_quantile))
    K = np.asarray(K, dtype=float).copy()
    K[np.isfinite(K) & (K >= upper_clip)] = upper_clip

    if normalize_kernel and not row_normalize and not col_normalize:
        row_sums = np.nansum(K, axis=1)
        usable = row_sums[row_sums > 1e-5]
        if usable.size:
            median_sum = float(np.median(usable))
            if np.isfinite(median_sum) and median_sum > 0:
                K /= median_sum
    elif row_normalize:
        row_sums = np.nansum(K, axis=1)
        nz = row_sums > 1e-4
        K[nz] /= row_sums[nz, np.newaxis]
    elif col_normalize:
        col_sums = np.nansum(K, axis=0)
        nz = col_sums > 1e-4
        K[:, nz] /= col_sums[np.newaxis, nz]

    K[~np.isfinite(K) | (K < lower_limit)] = 0.0
    return K


def _process_sparse_kernel(
    K: sparse.spmatrix,
    lower_limit: float,
    upper_quantile: float,
    row_normalize: bool = False,
    col_normalize: bool = False,
    normalize_kernel: bool = False,
) -> sparse.csr_matrix:
    """Sparse equivalent of :func:`_process_kernel`, without densification."""
    K = K.tocsr(copy=True)
    K.sum_duplicates()
    K.eliminate_zeros()
    if K.nnz == 0:
        return K

    upper_clip = float(np.quantile(K.data, upper_quantile))
    K.data[K.data >= upper_clip] = upper_clip

    if normalize_kernel and not row_normalize and not col_normalize:
        row_sums = np.asarray(K.sum(axis=1)).ravel()
        usable = row_sums[row_sums > 1e-5]
        if usable.size:
            median_sum = float(np.median(usable))
            if np.isfinite(median_sum) and median_sum > 0:
                K.data /= median_sum
    elif row_normalize:
        row_sums = np.asarray(K.sum(axis=1)).ravel()
        scale = np.ones_like(row_sums, dtype=float)
        nz = row_sums > 1e-4
        scale[nz] = 1.0 / row_sums[nz]
        K = (sparse.diags(scale) @ K).tocsr()
    elif col_normalize:
        col_sums = np.asarray(K.sum(axis=0)).ravel()
        scale = np.ones_like(col_sums, dtype=float)
        nz = col_sums > 1e-4
        scale[nz] = 1.0 / col_sums[nz]
        K = (K @ sparse.diags(scale)).tocsr()

    if K.nnz:
        K.data[(~np.isfinite(K.data)) | (K.data < lower_limit)] = 0.0
        K.eliminate_zeros()
    return K


def _nonzero_above(K, lower_limit: float) -> int:
    if sparse.issparse(K):
        return int(np.count_nonzero(K.data > lower_limit))
    return int(np.count_nonzero(np.asarray(K) > lower_limit))


def _has_nan(K) -> bool:
    if sparse.issparse(K):
        return bool(np.isnan(K.data).any())
    return bool(np.isnan(np.asarray(K)).all())


def _should_remove_sigma(
    K,
    lower_limit: float,
    sigma: float,
    ct_i: str,
    ct_j: str,
    min_ave_cell_neighbor: float,
    all_sigmas: list,
) -> bool:
    n1, n2 = K.shape
    required = float(min_ave_cell_neighbor) * min(n1, n2)
    available = _nonzero_above(K, lower_limit)

    if available < required:
        prop_nonzero = available / float(n1 * n2) if n1 and n2 else 0.0
        min_prop = required / float(n1 * n2) if n1 and n2 else np.inf
        warnings.warn(
            f"Kernel for {ct_i}-{ct_j} sigma={sigma}: too sparse "
            f"({prop_nonzero:.4f} < {min_prop:.4f}). "
            + ("Dropping this sigma." if len(all_sigmas) > 1 else "Only sigma available!"),
            stacklevel=3,
        )
        if len(all_sigmas) == 1:
            raise ValueError(
                f"Only sigma={sigma} provided but kernel is too sparse. Use a larger sigma."
            )
        return True

    if _has_nan(K):
        warnings.warn(
            f"Kernel for {ct_i}-{ct_j} sigma={sigma}: contains only NaN. Dropping sigma.",
            stacklevel=3,
        )
        if len(all_sigmas) == 1:
            raise ValueError(f"Only sigma={sigma} provided and kernel is all NaN.")
        return True
    return False


def _valid_multi_block(
    K,
    lower_limit: float,
    sigma: float,
    ct_i: str,
    ct_j: str,
    min_ave_cell_neighbor: float,
    slide,
) -> bool:
    """Multi-slide validity check; a bad block does not itself drop a sigma."""
    required = float(min_ave_cell_neighbor) * min(K.shape)
    if _nonzero_above(K, lower_limit) >= required:
        return True
    warnings.warn(
        f"Kernel for {ct_i}-{ct_j} on slide {slide} with sigma={sigma} is too sparse.",
        stacklevel=3,
    )
    return False


def _validate_kernel_args(
    sigma_values,
    lower_limit: float,
    upper_quantile: float,
    min_ave_cell_neighbor: float,
    row_normalize_kernel: bool,
    col_normalize_kernel: bool,
) -> list[float]:
    sigmas = list(sigma_values)
    if not sigmas or any(not np.isfinite(s) or s <= 0 for s in sigmas):
        raise ValueError("sigma_values must contain positive finite values.")
    if not 0 < lower_limit < 1:
        raise ValueError("lower_limit must be between 0 and 1.")
    if not 0 < upper_quantile < 1:
        raise ValueError("upper_quantile must be between 0 and 1.")
    if min_ave_cell_neighbor < 1:
        raise ValueError("min_ave_cell_neighbor must be at least 1.")
    if row_normalize_kernel and col_normalize_kernel:
        raise ValueError("Cannot use row and column normalization together.")
    return sigmas


def _kernel_key(sigma: float, slide, ct_i: str, ct_j: str) -> str:
    if slide is None:
        return _kernel_flat_name(sigma, ct_i, ct_j)
    return f"kernel|sigma{sigma}|{slide}|{ct_i}|{ct_j}"


def _distance_key(slide, ct_i: str, ct_j: str) -> str:
    if slide is None:
        return _dist_flat_name(ct_i, ct_j)
    return f"dist|{slide}|{ct_i}|{ct_j}"


def _pairs(cell_types: list[str]) -> list[tuple[str, str]]:
    if len(cell_types) == 1:
        return [(cell_types[0], cell_types[0])]
    return list(combinations(cell_types, 2))


def _dense_kernel_core(
    obj,
    sigma_values: list[float],
    lower_limit: float,
    upper_quantile: float,
    min_ave_cell_neighbor: float,
    row_normalize_kernel: bool,
    col_normalize_kernel: bool,
    normalize_kernel: bool,
):
    """Classic distance-to-kernel path for single- and multi-slide objects."""
    from .core import CoProMulti

    if not obj.distances:
        raise ValueError(
            "Distance matrices missing. Run compute_distance() first or use method='sparse'."
        )
    cts = list(obj.cell_types_of_interest)
    is_multi = isinstance(obj, CoProMulti)
    slides = list(obj.slide_list) if is_multi else [None]
    pairs = _pairs(cts)
    kernels = {}
    sigmas_to_remove = set()

    for sigma in sigma_values:
        sigma_bad = False
        sigma_has_valid_kernel = False
        for slide in slides:
            for ct_i, ct_j in pairs:
                key = _distance_key(slide, ct_i, ct_j)
                reverse = _distance_key(slide, ct_j, ct_i)
                transpose = False
                if key not in obj.distances and reverse in obj.distances:
                    key, transpose = reverse, True
                if key not in obj.distances:
                    warnings.warn(f"No distance matrix for {ct_i}-{ct_j}. Skipping.")
                    continue

                dist = np.asarray(obj.distances[key], dtype=float)
                if transpose:
                    dist = dist.T
                # Keep Inf diagonals as zero kernel entries. This matches the R
                # path and the sparse within-type construction exactly.
                K = _kernel_from_distance(dist, sigma, lower_limit)
                if is_multi:
                    # A weak slide/type block is omitted. It must not discard
                    # valid blocks for the same bandwidth on other slides.
                    if not _valid_multi_block(
                        K, lower_limit, sigma, ct_i, ct_j,
                        min_ave_cell_neighbor, slide,
                    ):
                        continue
                elif _should_remove_sigma(
                    K, lower_limit, sigma, ct_i, ct_j,
                    min_ave_cell_neighbor, sigma_values,
                ):
                    sigma_bad = True
                    continue
                kernels[_kernel_key(sigma, slide, ct_i, ct_j)] = _process_kernel(
                    K, lower_limit, upper_quantile,
                    row_normalize_kernel, col_normalize_kernel, normalize_kernel,
                )
                sigma_has_valid_kernel = True
        if is_multi and not sigma_has_valid_kernel:
            if len(sigma_values) == 1:
                raise ValueError(
                    f"Only sigma={sigma} was provided, but no valid kernel "
                    "blocks were generated across slides. Use a larger sigma."
                )
            warnings.warn(
                f"Removing sigma={sigma}: no valid kernel blocks were generated "
                "across slides.",
                stacklevel=2,
            )
            sigmas_to_remove.add(sigma)
        elif not is_multi and sigma_bad:
            sigmas_to_remove.add(sigma)

    for sigma in sigmas_to_remove:
        prefix = f"kernel|sigma{sigma}|"
        for key in [k for k in kernels if k.startswith(prefix)]:
            del kernels[key]

    obj.sigma_values = [s for s in sigma_values if s not in sigmas_to_remove]
    obj.kernel_matrices = kernels
    return obj


@dataclass
class _SparseBlock:
    slide: object
    ct_i: str
    ct_j: str
    A: np.ndarray
    B: np.ndarray
    within: bool
    tree_a: cKDTree
    tree_b: cKDTree
    percentile: float | None = None
    min_positive: float | None = None


def _coordinate_spec(location_data, dist_type, x_scale, y_scale, z_scale):
    lower_to_actual = {str(c).lower(): c for c in location_data.columns}
    if dist_type is None:
        dist_type = "Euclidean3D" if "z" in lower_to_actual else "Euclidean2D"
    normalized = str(dist_type).lower().replace("_", "").replace("-", "")
    if normalized in {"euclidean2d", "2d"}:
        axes, scales = ["x", "y"], [x_scale, y_scale]
        canonical = "Euclidean2D"
    elif normalized in {"euclidean3d", "3d"}:
        axes, scales = ["x", "y", "z"], [x_scale, y_scale, z_scale]
        canonical = "Euclidean3D"
    elif normalized in {"morphologyaware", "morphology"}:
        raise ValueError(
            "method='sparse' supports Euclidean2D/Euclidean3D only; "
            "use method='dense' for morphology-aware distances."
        )
    else:
        raise ValueError("dist_type must be 'Euclidean2D' or 'Euclidean3D'.")

    missing = [axis for axis in axes if axis not in lower_to_actual]
    if missing:
        raise ValueError(f"location_data_sub is missing coordinate column(s): {missing}.")
    scales = np.asarray(scales, dtype=float)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Coordinate scale factors must be positive and finite.")
    columns = [lower_to_actual[axis] for axis in axes]
    return canonical, columns, scales


def _make_sparse_blocks(
    obj,
    self_kernel: bool,
    dist_type,
    x_dist_scale: float,
    y_dist_scale: float,
    z_dist_scale: float,
) -> list[_SparseBlock]:
    from .core import CoProMulti

    cts = list(obj.cell_types_of_interest)
    if not cts:
        raise ValueError("No cell types of interest. Run subset_data() first.")
    if obj.location_data_sub is None or obj.cell_types_sub is None:
        raise ValueError("Subset data are missing. Run subset_data() first.")

    _, columns, scales = _coordinate_spec(
        obj.location_data_sub, dist_type,
        x_dist_scale, y_dist_scale, z_dist_scale,
    )
    coords_all = obj.location_data_sub.loc[:, columns].to_numpy(dtype=float) * scales
    if not np.isfinite(coords_all).all():
        raise ValueError("Spatial coordinates must be finite.")

    is_multi = isinstance(obj, CoProMulti)
    slides = list(obj.slide_list) if is_multi else [None]
    if is_multi:
        if obj.meta_data_sub is None or "slideID" not in obj.meta_data_sub.columns:
            raise ValueError("meta_data_sub must contain 'slideID' for multi-slide kernels.")
        slide_ids = obj.meta_data_sub["slideID"].to_numpy()
    else:
        slide_ids = None

    blocks = []
    pairs = [(ct, ct) for ct in cts] if self_kernel else _pairs(cts)
    cell_types = np.asarray(obj.cell_types_sub)
    for slide in slides:
        slide_mask = np.ones(len(cell_types), dtype=bool) if slide is None else slide_ids == slide
        for ct_i, ct_j in pairs:
            idx_i = np.flatnonzero(slide_mask & (cell_types == ct_i))
            idx_j = idx_i if ct_i == ct_j else np.flatnonzero(
                slide_mask & (cell_types == ct_j)
            )
            # Match the dense/R multi-slide path, which ignores tiny slide/type
            # blocks rather than letting them invalidate an otherwise usable sigma.
            if is_multi and (idx_i.size <= 5 or idx_j.size <= 5):
                continue
            if idx_i.size == 0 or idx_j.size == 0:
                warnings.warn(
                    f"Skipping empty kernel block {ct_i}-{ct_j}"
                    + ("" if slide is None else f" on slide {slide}"),
                    stacklevel=3,
                )
                continue
            A = np.ascontiguousarray(coords_all[idx_i])
            B = A if ct_i == ct_j else np.ascontiguousarray(coords_all[idx_j])
            tree_a = cKDTree(A)
            tree_b = tree_a if ct_i == ct_j else cKDTree(B)
            blocks.append(_SparseBlock(
                slide, str(ct_i), str(ct_j), A, B, ct_i == ct_j,
                tree_a, tree_b,
            ))
    if not blocks:
        raise ValueError("No non-empty cell-type blocks are available for kernel construction.")
    return blocks


def _query_block(block: _SparseBlock, radius: float):
    result = block.tree_a.sparse_distance_matrix(
        block.tree_b, radius, output_type="coo_matrix"
    )
    rows = np.asarray(result.row, dtype=np.int64)
    cols = np.asarray(result.col, dtype=np.int64)
    values = np.asarray(result.data, dtype=float)
    if block.within:
        keep = rows != cols
        rows, cols, values = rows[keep], cols[keep], values[keep]
    return rows, cols, values


def _low_percentile_block(block: _SparseBlock, probability: float) -> tuple[float, float]:
    """Recover the exact dense type-7 low quantile using radius expansion."""
    n_a, n_b = len(block.A), len(block.B)
    total = n_a * (n_a - 1) if block.within else n_a * n_b
    if total <= 0:
        raise ValueError(
            f"Kernel block {block.ct_i}-{block.ct_j} has too few cells for distances."
        )

    h = (total - 1) * probability
    lo, hi = int(np.floor(h)), int(np.ceil(h))
    need = hi + 1

    combined = block.A if block.within else np.vstack([block.A, block.B])
    span = np.ptp(combined, axis=0)
    max_radius = float(np.linalg.norm(span))
    if not np.isfinite(max_radius) or max_radius <= 0:
        raise ValueError(
            f"Cannot normalize kernel block {block.ct_i}-{block.ct_j}: "
            "all coordinates are coincident."
        )
    final_radius = max_radius * (1.0 + 1e-12) + np.finfo(float).eps
    radius = max(max_radius / 1024.0, np.finfo(float).eps)

    while True:
        _, _, values = _query_block(block, min(radius, final_radius))
        positive = values[values > 0]
        if values.size >= need and positive.size:
            min_positive = float(positive.min())
            ordered = np.sort(np.where(values == 0, min_positive, values))
            if ordered.size >= need:
                frac = h - lo
                percentile = float(ordered[lo] + frac * (ordered[hi] - ordered[lo]))
                return percentile, min_positive
        if radius >= final_radius:
            break
        radius = min(radius * 2.0, final_radius)

    raise ValueError(
        f"Cannot compute a finite positive low-distance percentile for "
        f"{block.ct_i}-{block.ct_j}; check for degenerate coordinates."
    )


def _build_sparse_raw_kernel(
    block: _SparseBlock,
    sigma: float,
    max_sigma: float,
    lower_limit: float,
    scaling_factor: float,
    truncate: bool,
) -> sparse.csr_matrix:
    support = max_sigma * np.sqrt(-2.0 * np.log(lower_limit))
    raw_radius = support / scaling_factor
    rows, cols, values = _query_block(block, raw_radius * (1.0 + 1e-12))
    if values.size == 0:
        return sparse.csr_matrix((len(block.A), len(block.B)), dtype=float)

    values = np.where(values == 0, block.min_positive, values)
    if truncate:
        values = np.maximum(values, block.percentile)
    values = values * scaling_factor
    kvals = np.exp(-0.5 * (values / sigma) ** 2)
    keep = np.isfinite(kvals) & (kvals >= lower_limit)
    K = sparse.coo_matrix(
        (kvals[keep], (rows[keep], cols[keep])),
        shape=(len(block.A), len(block.B)),
    ).tocsr()
    K.sum_duplicates()
    K.eliminate_zeros()
    return K


def _sparse_kernel_core(
    obj,
    sigma_values: list[float],
    lower_limit: float,
    upper_quantile: float,
    min_ave_cell_neighbor: float,
    row_normalize_kernel: bool,
    col_normalize_kernel: bool,
    normalize_kernel: bool,
    dist_type,
    x_dist_scale: float,
    y_dist_scale: float,
    z_dist_scale: float,
    normalize_distance: bool,
    normalize_target: float,
    truncate: bool,
    self_kernel: bool,
    verbose: bool,
    overwrite: bool = False,
):
    if not np.isfinite(normalize_target) or normalize_target <= 0:
        raise ValueError("normalize_target must be a positive finite scalar.")
    blocks = _make_sparse_blocks(
        obj, self_kernel, dist_type,
        x_dist_scale, y_dist_scale, z_dist_scale,
    )

    for block in blocks:
        probability = 1e-4 if block.within else min(
            1e-3, 2.0 / max(len(block.A), len(block.B))
        )
        block.percentile, block.min_positive = _low_percentile_block(
            block, probability
        )

    scaling_factor = (
        normalize_target / min(b.percentile for b in blocks)
        if normalize_distance else 1.0
    )
    if not np.isfinite(scaling_factor) or scaling_factor <= 0:
        raise ValueError(
            "Cannot compute distance normalization: scaling factor is not positive and finite."
        )
    obj.distance_scale_factor = float(scaling_factor)
    if verbose and normalize_distance:
        print(f"Distance normalization scaling factor: {scaling_factor:g}")

    existing_sigmas = list(obj.sigma_values)
    kernels = (
        {}
        if not self_kernel or overwrite
        else dict(obj.kernel_matrices)
    )
    sigmas_to_remove = set()
    max_sigma = max(sigma_values)
    from .core import CoProMulti
    is_multi = isinstance(obj, CoProMulti)
    for sigma in sigma_values:
        sigma_bad = False
        sigma_has_valid_kernel = False
        for block in blocks:
            K = _build_sparse_raw_kernel(
                block, sigma, max_sigma, lower_limit,
                scaling_factor, truncate,
            )
            if is_multi:
                valid = _valid_multi_block(
                    K, lower_limit, sigma, block.ct_i, block.ct_j,
                    min_ave_cell_neighbor, block.slide,
                )
                if self_kernel:
                    # R treats a sigma as invalid for self-kernels when any
                    # required self block is invalid, but preserves cross kernels.
                    sigma_bad = sigma_bad or not valid
                elif not valid:
                    # Cross-type multi-slide kernels keep valid slide blocks and
                    # drop the sigma only when no valid block exists anywhere.
                    continue
            else:
                valid = not _should_remove_sigma(
                    K, lower_limit, sigma, block.ct_i, block.ct_j,
                    min_ave_cell_neighbor, sigma_values,
                )
                sigma_bad = sigma_bad or not valid
                if not valid:
                    continue
            K = _process_sparse_kernel(
                K, lower_limit, upper_quantile,
                row_normalize_kernel, col_normalize_kernel, normalize_kernel,
            )
            kernels[_kernel_key(
                sigma, block.slide, block.ct_i, block.ct_j
            )] = K
            sigma_has_valid_kernel = sigma_has_valid_kernel or valid
        no_valid_multi_cross = (
            not self_kernel and is_multi and not sigma_has_valid_kernel
        )
        if no_valid_multi_cross and len(sigma_values) == 1:
            raise ValueError(
                f"Only sigma={sigma} was provided, but no valid kernel "
                "blocks were generated across slides. Use a larger sigma."
            )
        if (self_kernel and sigma_bad) or no_valid_multi_cross or (
            not self_kernel and not is_multi and sigma_bad
        ):
            if no_valid_multi_cross:
                warnings.warn(
                    f"Removing sigma={sigma}: no valid kernel blocks were generated "
                    "across slides.",
                    stacklevel=2,
                )
            sigmas_to_remove.add(sigma)

    for sigma in sigmas_to_remove:
        prefix = f"kernel|sigma{sigma}|"
        keys = [k for k in kernels if k.startswith(prefix)]
        if self_kernel:
            keys = [k for k in keys if k.split("|")[-2] == k.split("|")[-1]]
        for key in keys:
            del kernels[key]

    surviving = [s for s in sigma_values if s not in sigmas_to_remove]
    if not self_kernel or overwrite or not existing_sigmas:
        obj.sigma_values = surviving
    obj.kernel_matrices = kernels
    return obj


def _workload_counts(obj, self_kernel: bool) -> tuple[int, float]:
    """Return largest block dimension and dense entries (Python ints avoid overflow)."""
    from .core import CoProMulti

    cts = list(obj.cell_types_of_interest)
    cell_types = np.asarray(obj.cell_types_sub)
    if isinstance(obj, CoProMulti):
        slides = list(obj.slide_list)
        slide_ids = obj.meta_data_sub["slideID"].to_numpy()
    else:
        slides, slide_ids = [None], None

    max_block = 0
    entries = 0
    for slide in slides:
        slide_mask = np.ones(len(cell_types), dtype=bool) if slide is None else slide_ids == slide
        counts = [int(np.count_nonzero(slide_mask & (cell_types == ct))) for ct in cts]
        max_block = max(max_block, max(counts, default=0))
        if self_kernel:
            entries += sum(n * n for n in counts)
        elif len(counts) == 1:
            entries += counts[0] * counts[0]
        else:
            entries += sum(counts[i] * counts[j] for i, j in combinations(range(len(counts)), 2))
    return max_block, float(entries)


def _has_all_self_distances(obj) -> bool:
    from .core import CoProMulti

    cts = list(obj.cell_types_of_interest)
    slides = list(obj.slide_list) if isinstance(obj, CoProMulti) else [None]
    return all(
        _distance_key(slide, ct, ct) in obj.distances
        for slide in slides for ct in cts
    )


def compute_sparse_kernel(
    obj,
    sigma_values: list,
    lower_limit: float = 1e-7,
    upper_quantile: float = 0.85,
    min_ave_cell_neighbor: float = 2.0,
    row_normalize_kernel: bool = False,
    col_normalize_kernel: bool = False,
    normalize_kernel: bool = False,
    dist_type: str | None = None,
    x_dist_scale: float = 1.0,
    y_dist_scale: float = 1.0,
    z_dist_scale: float = 1.0,
    normalize_distance: bool = True,
    normalize_target: float = 0.01,
    truncate: bool = True,
    verbose: bool = False,
):
    """Build exact sparse Gaussian kernels directly from spatial coordinates.

    Only pairs within the Gaussian support radius implied by ``lower_limit``
    are enumerated with :class:`scipy.spatial.cKDTree`; all omitted values are
    exactly those the dense path would set to zero. Dense distance matrices are
    neither required nor created.
    """
    sigmas = _validate_kernel_args(
        sigma_values, lower_limit, upper_quantile, min_ave_cell_neighbor,
        row_normalize_kernel, col_normalize_kernel,
    )
    return _sparse_kernel_core(
        obj, sigmas, lower_limit, upper_quantile, min_ave_cell_neighbor,
        row_normalize_kernel, col_normalize_kernel, normalize_kernel,
        dist_type, x_dist_scale, y_dist_scale, z_dist_scale,
        normalize_distance, normalize_target, truncate,
        self_kernel=False, verbose=verbose,
    )


def compute_kernel_matrix(
    obj,
    sigma_values: list,
    lower_limit: float = 1e-7,
    upper_quantile: float = 0.85,
    min_ave_cell_neighbor: float = 2.0,
    row_normalize_kernel: bool = False,
    col_normalize_kernel: bool = False,
    normalize_kernel: bool = False,
    method: str = "auto",
    drop_distances: bool = True,
    auto_threshold: int = 5000,
    dist_type: str | None = None,
    x_dist_scale: float = 1.0,
    y_dist_scale: float = 1.0,
    z_dist_scale: float = 1.0,
    normalize_distance: bool = True,
    normalize_target: float = 0.01,
    truncate: bool = True,
    verbose: bool = False,
):
    """Compute kernels with a dense, sparse, or workload-aware automatic path.

    ``method='dense'`` preserves the original distance-to-kernel behavior.
    ``method='sparse'`` fuses coordinates, fixed-radius neighbors, and Gaussian
    evaluation. ``method='auto'`` selects sparse storage when the largest block
    reaches ``auto_threshold`` cells or total dense entries reach its square.
    ``drop_distances`` defaults to ``True`` because downstream CoPro steps only
    need kernels; pass ``False`` when distances must remain inspectable.
    """
    cts = list(obj.cell_types_of_interest)
    if not cts:
        raise ValueError("No cell types of interest. Run subset_data() first.")
    sigmas = _validate_kernel_args(
        sigma_values, lower_limit, upper_quantile, min_ave_cell_neighbor,
        row_normalize_kernel, col_normalize_kernel,
    )
    method = str(method).lower()
    if method not in {"auto", "dense", "sparse"}:
        raise ValueError("method must be 'auto', 'dense', or 'sparse'.")
    if not isinstance(auto_threshold, (int, np.integer)) or auto_threshold < 1:
        raise ValueError("auto_threshold must be a positive integer.")

    selected = method
    if method == "auto":
        max_block, entries = _workload_counts(obj, self_kernel=False)
        selected = (
            "sparse"
            if max_block >= auto_threshold or entries >= float(auto_threshold) ** 2
            else "dense"
        )
        if verbose:
            print(
                f"compute_kernel_matrix: method='auto' -> '{selected}' "
                f"(largest block={max_block}, dense entries={entries:g})"
            )

    if selected == "dense":
        result = _dense_kernel_core(
            obj, sigmas, lower_limit, upper_quantile,
            min_ave_cell_neighbor, row_normalize_kernel,
            col_normalize_kernel, normalize_kernel,
        )
    else:
        result = _sparse_kernel_core(
            obj, sigmas, lower_limit, upper_quantile,
            min_ave_cell_neighbor, row_normalize_kernel,
            col_normalize_kernel, normalize_kernel,
            dist_type, x_dist_scale, y_dist_scale, z_dist_scale,
            normalize_distance, normalize_target, truncate,
            self_kernel=False, verbose=verbose,
        )

    if drop_distances:
        result.distances = {}
    return result


def _dense_self_kernel_core(
    obj,
    sigma_values,
    lower_limit,
    upper_quantile,
    min_ave_cell_neighbor,
    row_normalize_kernel,
    col_normalize_kernel,
    normalize_kernel,
    verbose,
    overwrite,
):
    from .core import CoProMulti

    cts = list(obj.cell_types_of_interest)
    slides = list(obj.slide_list) if isinstance(obj, CoProMulti) else [None]
    if overwrite:
        obj.kernel_matrices = {}
    for slide in slides:
        for ct in cts:
            key = _distance_key(slide, ct, ct)
            if key not in obj.distances:
                raise ValueError(
                    f"Self-distance for {ct}"
                    + ("" if slide is None else f" on {slide}")
                    + " not found. Run compute_self_distance() first or use method='sparse'."
                )

    for sigma in sigma_values:
        for slide in slides:
            for ct in cts:
                dist = np.asarray(obj.distances[_distance_key(slide, ct, ct)], dtype=float)
                K = _kernel_from_distance(dist, sigma, lower_limit)
                if _should_remove_sigma(
                    K, lower_limit, sigma, ct, ct,
                    min_ave_cell_neighbor, sigma_values,
                ):
                    continue
                obj.kernel_matrices[_kernel_key(sigma, slide, ct, ct)] = _process_kernel(
                    K, lower_limit, upper_quantile,
                    row_normalize_kernel, col_normalize_kernel, normalize_kernel,
                )
                if verbose:
                    label = ct if slide is None else f"{slide}, {ct}"
                    print(f"  sigma={sigma}, {label}: {K.shape[0]}x{K.shape[1]}")
    if not obj.sigma_values:
        obj.sigma_values = list(sigma_values)
    return obj


def compute_self_kernel(
    obj,
    sigma_values: list | None = None,
    lower_limit: float = 1e-7,
    upper_quantile: float = 0.85,
    min_ave_cell_neighbor: float = 2.0,
    row_normalize_kernel: bool = False,
    col_normalize_kernel: bool = False,
    verbose: bool = True,
    normalize_kernel: bool = False,
    method: str = "auto",
    auto_threshold: int = 5000,
    dist_type: str | None = None,
    x_dist_scale: float = 1.0,
    y_dist_scale: float = 1.0,
    z_dist_scale: float = 1.0,
    normalize_distance: bool = True,
    normalize_target: float = 0.01,
    truncate: bool = True,
    overwrite: bool = False,
):
    """Add within-cell-type kernels using dense, sparse, or automatic dispatch."""
    cts = list(obj.cell_types_of_interest)
    if len(cts) == 1:
        warnings.warn("Only one cell type - use compute_kernel_matrix() instead.")
        return obj
    if sigma_values is None:
        sigma_values = obj.sigma_values
    sigmas = _validate_kernel_args(
        sigma_values, lower_limit, upper_quantile, min_ave_cell_neighbor,
        row_normalize_kernel, col_normalize_kernel,
    )
    method = str(method).lower()
    if method not in {"auto", "dense", "sparse"}:
        raise ValueError("method must be 'auto', 'dense', or 'sparse'.")
    if not isinstance(auto_threshold, (int, np.integer)) or auto_threshold < 1:
        raise ValueError("auto_threshold must be a positive integer.")

    selected = method
    if method == "auto":
        max_block, entries = _workload_counts(obj, self_kernel=True)
        selected = (
            "sparse"
            if not _has_all_self_distances(obj)
            or max_block >= auto_threshold
            or entries >= float(auto_threshold) ** 2
            else "dense"
        )
        if verbose:
            print(
                f"compute_self_kernel: method='auto' -> '{selected}' "
                f"(largest block={max_block}, dense entries={entries:g})"
            )

    if selected == "dense":
        return _dense_self_kernel_core(
            obj, sigmas, lower_limit, upper_quantile,
            min_ave_cell_neighbor, row_normalize_kernel,
            col_normalize_kernel, normalize_kernel, verbose, overwrite,
        )
    return _sparse_kernel_core(
        obj, sigmas, lower_limit, upper_quantile,
        min_ave_cell_neighbor, row_normalize_kernel,
        col_normalize_kernel, normalize_kernel,
        dist_type, x_dist_scale, y_dist_scale, z_dist_scale,
        normalize_distance, normalize_target, truncate,
        self_kernel=True, verbose=verbose, overwrite=overwrite,
    )
