"""compute_pca() — truncated SVD-based PCA matching R prcomp_irlba."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import svd_flip

from .core import CoProSingle
from .utils import center_scale_matrix


def _sign_correct_components(U: np.ndarray, Vt: np.ndarray):
    """Apply sklearn sign convention: largest-magnitude element of each loading is positive.

    This matches prcomp_irlba's sign correction so loadings are comparable.
    Vt rows are loadings (right singular vectors); U columns are left singular vectors.
    """
    # svd_flip from sklearn aligns signs based on max-abs element in U columns
    U, Vt = svd_flip(U, Vt)
    return U, Vt


def _center_scale(X: np.ndarray, center: bool, scale: bool):
    """Center and/or scale matrix, return (X_scaled, col_means, col_stds).

    Matches the logic in center_scale_matrix() for the center+scale case:
    zero-std and nearly-constant columns are not scaled (scale factor = 1).
    """
    col_means = X.mean(axis=0) if center else np.zeros(X.shape[1])
    X_centered = X - col_means if center else X.copy()
    if scale:
        col_stds = X_centered.std(axis=0, ddof=1)
        col_nz = np.sum(X != 0, axis=0) / X.shape[0]
        col_stds_safe = col_stds.copy()
        bad_cols = (col_stds < 1e-3) | (col_nz < 0.01)
        col_stds_safe[bad_cols] = 1.0
    else:
        col_stds_safe = np.ones(X.shape[1])
    X_scaled = X_centered / col_stds_safe
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
        sub = obj.normalized_data_sub[mask].astype(float)
        if sparse.issparse(sub):
            # Densify only the already-subsetted cell type, never the full
            # AnnData matrix. Centered PCA is inherently dense.
            sub = sub.toarray()

        # Center and scale
        if center and scale:
            sub_scaled = center_scale_matrix(sub)
        elif center:
            sub_scaled = sub - sub.mean(axis=0)
        elif scale:
            col_sds = sub.std(axis=0, ddof=1)
            col_sds[col_sds < 1e-10] = 1.0
            sub_scaled = sub / col_sds
        else:
            sub_scaled = sub

        n_cells, n_genes = sub_scaled.shape
        k = min(n_pca_use, n_cells - 1, n_genes - 1)

        # Truncated SVD via ARPACK (same Krylov family as R's IRLBA)
        # svds returns singular values in ascending order — reverse to descending
        U, s, Vt = svds(sub_scaled, k=k)
        U, s, Vt = U[:, ::-1], s[::-1], Vt[::-1, :]

        # Apply sign correction (matches prcomp_irlba convention)
        U, Vt = _sign_correct_components(U, Vt)

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
        X_ct = obj.normalized_data_sub[mask_ct].astype(float)
        if sparse.issparse(X_ct):
            X_ct = X_ct.toarray()
        slide_ct = slide_ids[mask_ct]

        k = min(n_pca, X_ct.shape[0] - 1, X_ct.shape[1] - 1)
        if k < 1:
            raise ValueError(f"Too few cells or genes for PCA for cell type '{ct}'.")

        # Center and scale on all cells of this type
        X_scaled, col_means, col_stds = _center_scale(X_ct, center, scale)

        # Global SVD via ARPACK (same Krylov family as R's IRLBA)
        # svds returns singular values in ascending order — reverse to descending
        U, s, Vt = svds(X_scaled, k=k)
        U, s, Vt = U[:, ::-1], s[::-1], Vt[::-1, :]
        U, Vt = svd_flip(U, Vt)
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
            X_slide = X_ct[slide_mask]
            if scale:
                X_slide_scaled = (X_slide - col_means) / col_stds
            elif center:
                X_slide_scaled = X_slide - col_means
            else:
                X_slide_scaled = X_slide
            # Project: scores = X_slide_scaled @ rotation
            scores_slide = X_slide_scaled @ rotation  # (n_cells_slide, k)
            pca_results[slide][ct] = scores_slide

    obj.pca_global = pca_global
    obj.pca_results = pca_results
    obj.n_pca = n_pca
    return obj
