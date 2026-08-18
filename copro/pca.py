"""compute_pca() — truncated SVD-based PCA matching R prcomp_irlba."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.utils.extmath import svd_flip

from .core import CoProSingle


def _sign_correct_components(U: np.ndarray, Vt: np.ndarray):
    """Apply sklearn sign convention: largest-magnitude element of each loading is positive.

    This matches prcomp_irlba's sign correction so loadings are comparable.
    Vt rows are loadings (right singular vectors); U columns are left singular vectors.
    """
    # svd_flip from sklearn aligns signs based on max-abs element in U columns
    U, Vt = svd_flip(U, Vt)
    return U, Vt


def _sparse_center_scale_params(
    X,
    center: bool,
    scale: bool,
    zero_sd_threshold: float = 1e-3,
    nz_proportion_threshold: float = 0.01,
):
    """Column means/scales for a sparse matrix without densifying it.

    Mirrors R ``.sparse_pca_parameters()``: the variance is recovered from the
    column sums of squares, and columns with a tiny standard deviation or too
    few non-zeros are left unscaled.
    """
    n, p = X.shape
    means = np.asarray(X.mean(axis=0), dtype=float).ravel()
    sumsq = np.asarray(X.multiply(X).sum(axis=0), dtype=float).ravel()

    if center:
        col_means = means
        variance = np.maximum((sumsq - n * means**2) / max(1, n - 1), 0.0)
    else:
        col_means = np.zeros(p)
        # R's scale(center = FALSE, scale = TRUE) divides by the *uncentered*
        # root mean square, so the sparse and dense paths agree.
        variance = sumsq / max(1, n - 1)

    if scale:
        col_stds = np.sqrt(variance)
        col_nz = np.asarray((X != 0).sum(axis=0), dtype=float).ravel() / n
        bad_cols = (
            ~np.isfinite(col_stds)
            | (col_stds < zero_sd_threshold)
            | (col_nz < nz_proportion_threshold)
        )
        col_stds[bad_cols] = 1.0
    else:
        col_stds = np.ones(p)

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
    """Return a float64 sparse matrix in a format with fast mat-vec products."""
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

    The dense counterpart of _sparse_center_scale_params(): same tiny-variance /
    very-sparse safeguard (R's center_scale_matrix_opt), and the same uncentered
    root-mean-square scale when center=False (R's scale(center = FALSE)).
    """
    n_rows, n_cols = X.shape
    col_means = X.mean(axis=0) if center else np.zeros(n_cols)

    if not scale:
        # No division needed; only centering can require a new array.
        return (X - col_means if center else X), col_means, np.ones(n_cols)

    if center:
        X_scaled = X - col_means
        col_stds = X_scaled.std(axis=0, ddof=1)
    else:
        X_scaled = X.copy()
        sumsq = np.einsum("ij,ij->j", X, X)
        col_stds = np.sqrt(sumsq / max(1, n_rows - 1))

    col_nz = np.sum(X != 0, axis=0) / n_rows
    col_stds_safe = col_stds.copy()
    bad_cols = ~np.isfinite(col_stds) | (col_stds < 1e-3) | (col_nz < 0.01)
    col_stds_safe[bad_cols] = 1.0

    X_scaled /= col_stds_safe  # in place on our own copy: no second full-size temporary
    return X_scaled, col_means, col_stds_safe


def compute_pca(obj, n_pca: int = 30, center: bool = True, scale: bool = True):
    """Compute truncated PCA for each cell type of interest.

    Dispatches to multi-slide version for CoProMulti objects.

    Stores in obj.pca_global[ct]:
      'components': V  (n_pca × n_genes) — loadings (rotation in R)
      'scores':     X_rot  (n_cells × n_pca) — PC scores (x in R)
      'sdev':       singular_values / sqrt(n-1)  — matches R $sdev
    """
    from .core import CoProMulti
    if isinstance(obj, CoProMulti):
        return _compute_pca_multi(obj, n_pca, center, scale)

    # --- Single-slide path ---
    cts = obj.cell_types_of_interest
    if not cts:
        raise ValueError("No cell types of interest set. Run subset_data() first.")

    n_pca_use = min(n_pca, obj.normalized_data_sub.shape[1] - 1)

    for ct in cts:
        mask = obj.cell_types_sub == ct
        sub = obj.normalized_data_sub[mask]
        n_cells, n_genes = sub.shape
        k = min(n_pca_use, n_cells - 1, n_genes - 1)

        if sparse.issparse(sub):
            # Matrix-free path (same idea as R's sparse prcomp_irlba call): the
            # Krylov solver only needs products with the standardized matrix,
            # so the cell-type block is never densified.
            sub = _as_svd_ready_sparse(sub)
            col_means, col_stds = _sparse_center_scale_params(sub, center, scale)
            operand = _implicit_center_scale_operator(sub, col_means, col_stds)
        else:
            # Same safeguard as the multi-slide path and utils.center_scale_matrix
            # (R's center_scale_matrix_opt): tiny-variance or very-sparse genes
            # are left unscaled.
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


def _compute_pca_multi(obj, n_pca=30, center=True, scale=True):
    """Compute global PCA for CoProMulti. Fits on all cells concatenated, projects per slide."""
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

        k = min(n_pca, X_ct.shape[0] - 1, X_ct.shape[1] - 1)
        if k < 1:
            raise ValueError(f"Too few cells or genes for PCA for cell type '{ct}'.")

        # Center and scale on all cells of this type — implicitly when the
        # block is sparse, so it is never densified.
        if sparse.issparse(X_ct):
            X_ct = _as_svd_ready_sparse(X_ct)
            col_means, col_stds = _sparse_center_scale_params(X_ct, center, scale)
            operand = _implicit_center_scale_operator(X_ct, col_means, col_stds)
        else:
            X_ct = np.asarray(X_ct, dtype=float)
            operand, col_means, col_stds = _center_scale(X_ct, center, scale)

        # Global SVD via ARPACK (same Krylov family as R's IRLBA), descending
        # order with the prcomp_irlba sign convention.
        U, Vt, s = _truncated_svd(operand, k)
        sdev = s / np.sqrt(max(X_ct.shape[0] - 1, 1))
        scores_global = U * s  # (n_cells_ct, k)
        rotation = Vt.T        # (n_genes, k)

        pca_global[ct] = {
            "components": Vt,      # (k, n_genes)
            "rotation": rotation,  # (n_genes, k)
            "scores": scores_global,
            "sdev": sdev,
            "col_means": col_means,
            "col_stds": col_stds,
        }

        # Project each slide's cells using global rotation
        for slide in slides:
            slide_mask = slide_ct == slide
            if not np.any(slide_mask):
                continue
            # Project: scores = ((X_slide - col_means) / col_stds) @ rotation,
            # done without materializing the standardized block. col_means is
            # zero when center=False and col_stds is one when scale=False, so
            # this covers all four center/scale combinations.
            scores_slide = _project_scaled(
                X_ct[slide_mask], rotation, col_means, col_stds
            )  # (n_cells_slide, k)
            pca_results[slide][ct] = scores_slide

    obj.pca_global = pca_global
    obj.pca_results = pca_results
    obj.n_pca = n_pca
    return obj
