"""run_skr_cca() — orchestrates the SkrCCA optimization over all sigma values."""

from __future__ import annotations

import numpy as np

from .core import CoProSingle
from .optimization import optimize_bilinear, optimize_bilinear_n


def _prepare_pc_matrices(
    obj: CoProSingle,
    scale_pcs: bool,
    cell_types: list,
) -> dict:
    """Build PC score matrices, optionally scaled by std dev of each PC.

    Matches R: scale(pca$x, center=FALSE, scale=sdev)
    """
    pc_mats = {}
    for ct in cell_types:
        pca = obj.pca_global[ct]
        scores = pca["scores"].astype(float)  # (n_cells, k)
        if scale_pcs:
            sdev = pca["sdev"]  # (k,)
            # Divide each column by its std dev (matching R scale(..., center=FALSE, scale=sdev))
            sdev_safe = sdev.copy()
            sdev_safe[sdev_safe < 1e-10] = 1.0
            scores = scores / sdev_safe[np.newaxis, :]
        pc_mats[ct] = scores
    return pc_mats


def _prepare_pc_matrices_multi(obj, scale_pcs: bool, cell_types: list) -> dict:
    """Return {slide: {ct: X_scaled}} for multi-slide."""
    slides = obj.slide_list
    result = {slide: {} for slide in slides}
    for ct in cell_types:
        pca = obj.pca_global[ct]
        sdev = pca["sdev"]
        sdev_safe = sdev.copy()
        sdev_safe[sdev_safe < 1e-10] = 1.0
        for slide in slides:
            if slide not in obj.pca_results or ct not in obj.pca_results[slide]:
                continue
            scores = obj.pca_results[slide][ct].astype(float)
            if scale_pcs:
                scores = scores / sdev_safe[np.newaxis, :]
            result[slide][ct] = scores
    return result


def run_skr_cca(
    obj,
    scale_pcs: bool = True,
    n_cc: int = 2,
    tol: float = 1e-5,
    max_iter: int = 500,
):
    """Run SkrCCA optimization for all sigma values.

    Dispatches to multi-slide version for CoProMulti objects.
    Stores results in obj.skr_cca_out['sigma_{sigma}'][ct] = w_matrix (n_pca, n_cc).
    """
    from .core import CoProMulti
    if isinstance(obj, CoProMulti):
        return _run_skr_cca_multi(obj, scale_pcs, n_cc, tol, max_iter)

    # --- Single-slide path ---
    cts = obj.cell_types_of_interest
    if not cts:
        raise ValueError("No cell types of interest. Run subset_data() first.")
    if not obj.sigma_values:
        raise ValueError("No sigma values. Run compute_kernel_matrix() first.")
    if not obj.pca_global:
        raise ValueError("PCA results missing. Run compute_pca() first.")

    X_dict = _prepare_pc_matrices(obj, scale_pcs, cts)

    cca_out = {}
    for sigma in obj.sigma_values:
        sigma_name = f"sigma_{sigma}"
        print(f"Running SkrCCA for sigma = {sigma}")

        try:
            # First component
            w_dict = optimize_bilinear(
                X_dict=X_dict,
                flat_kernels=obj.kernel_matrices,
                sigma=sigma,
                max_iter=max_iter,
                tol=tol,
            )

            # Additional components
            if n_cc > 1:
                w_dict = optimize_bilinear_n(
                    X_dict=X_dict,
                    flat_kernels=obj.kernel_matrices,
                    sigma=sigma,
                    w_dict=w_dict,
                    cell_types=cts,
                    n_cc=n_cc,
                    max_iter=max_iter,
                    tol=tol,
                )

            cca_out[sigma_name] = w_dict

        except Exception as e:
            import warnings
            warnings.warn(f"Optimization failed for sigma={sigma}: {e}")
            cca_out[sigma_name] = None

    obj.skr_cca_out = cca_out
    obj.n_cc = n_cc
    obj.scale_pcs = scale_pcs
    return obj


def _run_skr_cca_multi(obj, scale_pcs, n_cc, tol, max_iter):
    """Multi-slide SkrCCA optimization. Shared weight vectors across slides."""
    from .optimization import optimize_bilinear_multi_slides, optimize_bilinear_n_multi_slides
    cts = obj.cell_types_of_interest
    slides = obj.slide_list
    X_list_all = _prepare_pc_matrices_multi(obj, scale_pcs, cts)

    cca_out = {}
    for sigma in obj.sigma_values:
        sigma_name = f"sigma_{sigma}"
        print(f"Running SkrCCA (multi) for sigma = {sigma}")
        try:
            w_dict = optimize_bilinear_multi_slides(
                X_list_all, obj.kernel_matrices, sigma, slides, max_iter, tol
            )
            if n_cc > 1:
                w_dict = optimize_bilinear_n_multi_slides(
                    X_list_all, obj.kernel_matrices, sigma, slides,
                    w_dict, cts, n_cc, max_iter, tol
                )
            cca_out[sigma_name] = w_dict
        except Exception as e:
            import warnings
            warnings.warn(f"Multi-slide optimization failed for sigma={sigma}: {e}")
            cca_out[sigma_name] = None

    obj.skr_cca_out = cca_out
    obj.n_cc = n_cc
    obj.scale_pcs = scale_pcs
    return obj
