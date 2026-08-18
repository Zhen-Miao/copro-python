"""compute_pca() — truncated SVD-based PCA matching R prcomp_irlba."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.utils.extmath import svd_flip

from .core import CoProSingle

ZERO_SD_THRESHOLD = 1e-3
NZ_PROPORTION_THRESHOLD = 0.01


def _sign_correct_components(U: np.ndarray, Vt: np.ndarray):
    """Apply sklearn sign convention: largest-magnitude element of each loading is positive.

    This matches prcomp_irlba's sign correction so loadings are comparable.
    Vt rows are loadings (right singular vectors); U columns are left singular vectors.
    """
    # svd_flip from sklearn aligns signs based on max-abs element in U columns
    U, Vt = svd_flip(U, Vt)
    return U, Vt


def _column_moments(X, center: bool, scale: bool):
    """Column means, *unguarded* scales, and non-zero proportions.

    The scale is the centered standard deviation when ``center`` is True and the
    uncentered root mean square otherwise — both with the ``n - 1`` denominator,
    which is what R's ``sd()`` and ``scale(center = FALSE, scale = TRUE)`` use.

    Sparse input recovers the variance from column sums of squares (R's
    ``.sparse_pca_parameters()``); dense input uses the exact two-pass form
    (R's ``center_scale_matrix_opt()``). The two agree to ~1e-12 on
    log-normalized expression, and the one-pass form is what keeps the sparse
    path from having to materialize a centered copy.

    Applying the guard is left to the caller, because the within-slide path has
    to combine the raw scales across slides before deciding which genes to
    guard.
    """
    n = X.shape[0]
    denom = max(1, n - 1)

    if sparse.issparse(X):
        means = np.asarray(X.mean(axis=0), dtype=float).ravel()
        nz_proportion = np.asarray((X != 0).sum(axis=0), dtype=float).ravel() / n
        if not scale:
            return means, np.ones(X.shape[1]), nz_proportion
        sumsq = np.asarray(X.multiply(X).sum(axis=0), dtype=float).ravel()
        variance = (
            np.maximum((sumsq - n * means**2) / denom, 0.0) if center
            else sumsq / denom
        )
        return means, np.sqrt(variance), nz_proportion

    means = X.mean(axis=0)
    nz_proportion = np.sum(X != 0, axis=0) / n
    if not scale:
        return means, np.ones(X.shape[1]), nz_proportion
    if center:
        scales = (X - means).std(axis=0, ddof=1)
    else:
        scales = np.sqrt(np.einsum("ij,ij->j", X, X) / denom)
    return means, scales, nz_proportion


def _degenerate_columns(
    scales,
    nz_proportion,
    zero_sd_threshold: float = ZERO_SD_THRESHOLD,
    nz_proportion_threshold: float = NZ_PROPORTION_THRESHOLD,
):
    """Genes that must be left unscaled: tiny spread, or too few non-zeros."""
    return (
        ~np.isfinite(scales)
        | (scales < zero_sd_threshold)
        | (nz_proportion < nz_proportion_threshold)
    )


def _sparse_center_scale_params(
    X,
    center: bool,
    scale: bool,
    zero_sd_threshold: float = ZERO_SD_THRESHOLD,
    nz_proportion_threshold: float = NZ_PROPORTION_THRESHOLD,
):
    """Column means/scales for a sparse matrix without densifying it.

    Mirrors R ``.sparse_pca_parameters()`` (CoPro 1.3.0,
    ``R/13_compute_PCA.R``), which feeds the same two vectors to ``irlba``'s
    ``center``/``scale.`` arguments.
    """
    means, scales, nz_proportion = _column_moments(X, center, scale)
    col_means = means if center else np.zeros(X.shape[1])

    if not scale:
        return col_means, np.ones(X.shape[1])

    col_stds = scales.copy()
    col_stds[
        _degenerate_columns(scales, nz_proportion, zero_sd_threshold, nz_proportion_threshold)
    ] = 1.0
    return col_means, col_stds


def _implicit_center_scale_operator(X, col_means, col_stds):
    """``(X - 1 m') diag(s)^-1`` as a matrix-free operator.

    This is what R's IRLBA does with its ``center``/``scale.`` arguments: the
    Krylov solver only needs products with the standardized matrix, so the
    sparse input never has to be densified. Products are exact, so the
    resulting SVD matches the dense path to machine precision.
    """
    n, p = X.shape
    inv = np.asarray(1.0 / col_stds, dtype=float)
    mu = np.asarray(col_means, dtype=float) * inv   # column means after scaling

    def matmat(V):
        V = np.asarray(V, dtype=float)
        out = np.asarray(X @ (V * inv[:, None]), dtype=float)
        out -= mu @ V
        return out

    def rmatmat(U):
        U = np.asarray(U, dtype=float)
        out = np.asarray(X.T @ U, dtype=float)
        out -= np.outer(col_means, U.sum(axis=0))
        out *= inv[:, None]
        return out

    return LinearOperator(
        (n, p),
        matvec=lambda v: matmat(np.asarray(v).reshape(p, 1)).ravel(),
        rmatvec=lambda u: rmatmat(np.asarray(u).reshape(n, 1)).ravel(),
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=float,
    )


def _project_scaled(X, rotation, col_means, col_stds):
    """Project rows of ``X`` onto ``rotation`` after implicit centering/scaling."""
    if sparse.issparse(X):
        # ((X - 1 m') D^-1) V == X (D^-1 V) - 1 (m' D^-1 V), so the scaling is
        # folded into the rotation once and the offset is a single row vector.
        scaled_rotation = rotation / col_stds[:, None]
        return np.asarray(X @ scaled_rotation) - col_means @ scaled_rotation
    return ((X - col_means) / col_stds) @ rotation


def _as_svd_ready_sparse(X):
    """Return a float64 sparse matrix in a format with fast mat-vec products.

    Returns ``X`` itself when it already qualifies — callers must own their
    input and must not mutate the result in place.
    """
    if X.format not in ("csr", "csc"):
        X = X.tocsr()
    if X.dtype != np.float64:
        X = X.astype(np.float64)
    return X


def _truncated_svd(A, k: int):
    """Top-``k`` SVD in descending order with the prcomp_irlba sign convention."""
    U, s, Vt = svds(A, k=k)
    U, s, Vt = U[:, ::-1], s[::-1], Vt[::-1, :]
    return _sign_correct_components(U, Vt) + (s,)


def _center_scale(X: np.ndarray, center: bool, scale: bool):
    """Center and/or scale a dense matrix; return (X_scaled, col_means, col_stds).

    The dense counterpart of _sparse_center_scale_params(). The tiny-variance /
    very-sparse safeguard matches R's center_scale_matrix_opt(); the uncentered
    root-mean-square scale when center=False matches R's
    scale(center = FALSE, scale = TRUE).

    Note that R applies the safeguard in its *sparse* path
    (.sparse_pca_parameters) for every scale=TRUE case, but its dense
    scale-only branch is a bare scale() call with no safeguard, which divides
    an all-zero gene by zero. We follow the sparse convention on both paths so
    the two agree and no gene is divided by zero.

    Returns ``X`` itself when neither flag is set — callers must own their
    input and must not mutate the result in place.
    """
    n_cols = X.shape[1]
    col_means = X.mean(axis=0) if center else np.zeros(n_cols)

    if not scale:
        # No division needed; only centering can require a new array.
        return (X - col_means if center else X), col_means, np.ones(n_cols)

    _, scales, nz_proportion = _column_moments(X, center, scale)
    X_scaled = (X - col_means) if center else X.copy()

    col_stds_safe = scales.copy()
    col_stds_safe[_degenerate_columns(scales, nz_proportion)] = 1.0

    X_scaled /= col_stds_safe  # in place on our own copy: no second full-size temporary
    return X_scaled, col_means, col_stds_safe


def _slide_row_groups(slide_ids, slides):
    """Row indices of each slide, in the order given by ``slides``."""
    return [np.flatnonzero(slide_ids == slide) for slide in slides]


def _within_slide_params(
    X,
    slide_ids,
    slides,
    center: bool,
    scale: bool,
    zero_sd_threshold: float = ZERO_SD_THRESHOLD,
    nz_proportion_threshold: float = NZ_PROPORTION_THRESHOLD,
):
    """Per-slide gene means and scales, mirroring R ``.withinSlidePCAParameters()``.

    Returns ``(rows, centers, scales, guarded)`` where ``centers`` and
    ``scales`` are (n_slides, n_genes).

    A gene is guarded on *every* slide as soon as it is degenerate on any one of
    them. Deciding per block instead would let a gene be standardized on one
    slide and left raw on another, which puts a per-slide scale difference back
    into exactly the low-detection genes whose detection rate is itself often
    the batch effect.
    """
    p = X.shape[1]
    rows = _slide_row_groups(slide_ids, slides)
    centers = np.zeros((len(slides), p))
    scales = np.ones((len(slides), p))
    guarded = np.zeros(p, dtype=bool)

    for k, idx in enumerate(rows):
        if idx.size == 0:
            continue
        means, block_scale, nz_proportion = _column_moments(X[idx], center, scale)
        if center:
            centers[k] = means
        if scale:
            guarded |= _degenerate_columns(
                block_scale, nz_proportion, zero_sd_threshold, nz_proportion_threshold
            )
            scales[k] = block_scale

    if scale and guarded.any():
        scales[:, guarded] = 1.0
    # An empty block keeps scale 1 regardless; also stop a stray non-finite or
    # non-positive scale from reaching the divisor.
    scales[~np.isfinite(scales) | (scales <= 0)] = 1.0

    return rows, centers, scales, guarded


def _within_slide_operator(X, rows, centers, scales):
    """Within-slide standardized matrix as a matrix-free operator.

    Row ``i`` of the result is ``(X[i] - centers[k]) / scales[k]`` for the slide
    ``k`` that cell ``i`` belongs to, stacked in the original row order. Mirrors
    R's ``CoProWithinSlideMatrix``.
    """
    n, p = X.shape
    blocks = [X[idx] if idx.size else None for idx in rows]
    inv = 1.0 / scales                       # (n_slides, n_genes)

    def matmat(V):
        V = np.asarray(V, dtype=float)
        out = np.zeros((n, V.shape[1]), dtype=float)
        for k, idx in enumerate(rows):
            if idx.size == 0:
                continue
            scaled_V = V * inv[k][:, None]
            out[idx] = np.asarray(blocks[k] @ scaled_V, dtype=float) - centers[k] @ scaled_V
        return out

    def rmatmat(U):
        U = np.asarray(U, dtype=float)
        out = np.zeros((p, U.shape[1]), dtype=float)
        for k, idx in enumerate(rows):
            if idx.size == 0:
                continue
            U_k = U[idx]
            contribution = np.asarray(blocks[k].T @ U_k, dtype=float)
            contribution -= np.outer(centers[k], U_k.sum(axis=0))
            out += contribution * inv[k][:, None]
        return out

    return LinearOperator(
        (n, p),
        matvec=lambda v: matmat(np.asarray(v).reshape(p, 1)).ravel(),
        rmatvec=lambda u: rmatmat(np.asarray(u).reshape(n, 1)).ravel(),
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=float,
    )


def _materialize_within_slide(X, rows, centers, scales):
    """Dense counterpart of _within_slide_operator()."""
    out = np.zeros(X.shape, dtype=float)
    for k, idx in enumerate(rows):
        if idx.size == 0:
            continue
        out[idx] = (X[idx] - centers[k]) / scales[k]
    return out


def _max_pca_rank(n_cells: int, n_genes: int) -> int:
    return max(0, min(n_cells - 1, n_genes - 1))


def _max_within_slide_pca_rank(n_cells: int, n_genes: int, n_blocks: int, center: bool) -> int:
    """Feasible rank once each slide block has been centered separately.

    Centering every non-empty slide block removes one independent row-space
    direction per block, which is stricter than pooled centering's n - 1 bound.
    Mirrors R ``.max_within_slide_pca_rank()``.
    """
    rank = _max_pca_rank(n_cells, n_genes)
    if not center:
        return rank
    return max(0, min(rank, n_cells - n_blocks))


def compute_pca(
    obj,
    n_pca: int = 30,
    center: bool = True,
    scale: bool = True,
    center_per_slide: bool = True,
):
    """Compute truncated PCA for each cell type of interest.

    Dispatches to multi-slide version for CoProMulti objects.

    Stores in obj.pca_global[ct]:
      'components': V  (n_pca × n_genes) — loadings (rotation in R)
      'scores':     X_rot  (n_cells × n_pca) — PC scores (x in R)
      'sdev':       singular_values / sqrt(n-1)  — matches R $sdev

    ``center_per_slide`` applies only to CoProMulti and is ignored otherwise.
    """
    from .core import CoProMulti
    if isinstance(obj, CoProMulti):
        return _compute_pca_multi(obj, n_pca, center, scale, center_per_slide)

    # --- Single-slide path ---
    cts = obj.cell_types_of_interest
    if not cts:
        raise ValueError("No cell types of interest set. Run subset_data() first.")

    n_pca_use = min(n_pca, obj.normalized_data_sub.shape[1] - 1)

    for ct in cts:
        mask = obj.cell_types_sub == ct
        sub = obj.normalized_data_sub[mask]
        n_cells, n_genes = sub.shape
        k = min(n_pca_use, _max_pca_rank(n_cells, n_genes))
        if k < 1:
            raise ValueError(
                f"Too few cells or genes for PCA for cell type '{ct}' "
                f"({n_cells} cells, {n_genes} genes); at least two of each are required."
            )

        if sparse.issparse(sub):
            # Matrix-free path (same idea as R's sparse prcomp_irlba call): the
            # Krylov solver only needs products with the standardized matrix,
            # so the cell-type block is never densified.
            sub = _as_svd_ready_sparse(sub)
            col_means, col_stds = _sparse_center_scale_params(sub, center, scale)
            operand = _implicit_center_scale_operator(sub, col_means, col_stds)
        else:
            sub = np.asarray(sub, dtype=float)
            operand, _, _ = _center_scale(sub, center, scale)

        # Truncated SVD via ARPACK (same Krylov family as R's IRLBA), with the
        # prcomp_irlba sign convention applied to the descending-order result.
        U, Vt, s = _truncated_svd(operand, k)

        # Scores = U * s  (equivalent to X_scaled @ V)
        scores = U * s  # (n_cells, k)

        # sdev = singular_values / sqrt(n - 1)  — matches R prcomp_irlba$sdev
        sdev = s / np.sqrt(n_cells - 1)

        obj.pca_global[ct] = {
            "components": Vt,          # (k, n_genes) — matches R $rotation^T
            "rotation": Vt.T,          # (n_genes, k) — matches R $rotation
            "scores": scores,          # (n_cells, k) — matches R $x
            "sdev": sdev,              # (k,)
            "n_cells": n_cells,
        }

    obj.n_pca = n_pca
    return obj


def _compute_pca_multi(obj, n_pca=30, center=True, scale=True, center_per_slide=True):
    """Compute global PCA for CoProMulti.

    With ``center_per_slide`` (the default, matching R CoPro 1.3.0), each
    (slide, cell type) block is standardized against its own gene means and
    scales before the single shared SVD, so a per-slide shift cannot drive the
    shared loadings. Per-slide scores are then rows of the global score matrix.

    With ``center_per_slide=False`` the older pooled behaviour is used: fit on
    all cells of the type at once, then project each slide.
    """
    cts = obj.cell_types_of_interest
    if not cts:
        raise ValueError("No cell types of interest. Run subset_data() first.")
    if "slideID" not in obj.meta_data_sub.columns:
        raise ValueError("meta_data_sub must have 'slideID' for CoProMulti.")

    slide_ids = obj.meta_data_sub["slideID"].values
    slides = obj.slide_list

    pca_global = {}
    pca_results = {slide: {} for slide in slides}

    for ct in cts:
        mask_ct = obj.cell_types_sub == ct
        X_ct = obj.normalized_data_sub[mask_ct]
        slide_ct = slide_ids[mask_ct]
        n_cells, n_genes = X_ct.shape

        if center_per_slide:
            n_blocks = len(np.unique(slide_ct))
            max_rank = _max_within_slide_pca_rank(n_cells, n_genes, n_blocks, center)
        else:
            max_rank = _max_pca_rank(n_cells, n_genes)
        k = min(n_pca, max_rank)
        if k < 1:
            raise ValueError(
                f"Too few cells or genes for PCA for cell type '{ct}' "
                f"({n_cells} cells, {n_genes} genes"
                + (f", {n_blocks} slides" if center_per_slide else "")
                + ")."
            )

        is_sparse = sparse.issparse(X_ct)
        if is_sparse:
            X_ct = _as_svd_ready_sparse(X_ct)
        else:
            X_ct = np.asarray(X_ct, dtype=float)

        if center_per_slide:
            rows, slide_centers, slide_scales, _ = _within_slide_params(
                X_ct, slide_ct, slides, center, scale
            )
            operand = (
                _within_slide_operator(X_ct, rows, slide_centers, slide_scales)
                if is_sparse
                else _materialize_within_slide(X_ct, rows, slide_centers, slide_scales)
            )
        else:
            # Center and scale on all cells of this type — implicitly when the
            # block is sparse, so it is never densified.
            if is_sparse:
                col_means, col_stds = _sparse_center_scale_params(X_ct, center, scale)
                operand = _implicit_center_scale_operator(X_ct, col_means, col_stds)
            else:
                operand, col_means, col_stds = _center_scale(X_ct, center, scale)

        # Global SVD via ARPACK (same Krylov family as R's IRLBA), descending
        # order with the prcomp_irlba sign convention.
        U, Vt, s = _truncated_svd(operand, k)
        sdev = s / np.sqrt(max(n_cells - 1, 1))
        scores_global = U * s  # (n_cells_ct, k)
        rotation = Vt.T        # (n_genes, k)

        pca_global[ct] = {
            "components": Vt,      # (k, n_genes)
            "rotation": rotation,  # (n_genes, k)
            "scores": scores_global,
            "sdev": sdev,
            "n_cells": n_cells,
        }

        if center_per_slide:
            # The shared loading lives in within-slide standardized gene
            # coordinates; a single raw-unit back-projection does not exist
            # when the per-slide scales differ.
            pca_global[ct]["preprocessing"] = "within_slide"
            pca_global[ct]["slide_centers"] = slide_centers
            pca_global[ct]["slide_scales"] = slide_scales
            # Per-slide scores are rows of the one global score matrix; under
            # within-slide preprocessing they already have zero mean by
            # construction, so no separate projection is needed.
            for slide, idx in zip(slides, rows):
                if idx.size == 0:
                    continue
                pca_results[slide][ct] = scores_global[idx]
        else:
            pca_global[ct]["preprocessing"] = "pooled"
            pca_global[ct]["col_means"] = col_means
            pca_global[ct]["col_stds"] = col_stds
            for slide in slides:
                slide_mask = slide_ct == slide
                if not np.any(slide_mask):
                    continue
                # Project: scores = ((X_slide - col_means) / col_stds) @ rotation,
                # done without materializing the standardized block. col_means is
                # zero when center=False and col_stds is one when scale=False, so
                # this covers all four center/scale combinations.
                pca_results[slide][ct] = _project_scaled(
                    X_ct[slide_mask], rotation, col_means, col_stds
                )

    obj.pca_global = pca_global
    obj.pca_results = pca_results
    obj.n_pca = n_pca
    return obj
