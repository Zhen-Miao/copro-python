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

    # Match the R implementation's structural validation: shared multi-slide
    # weights require every requested cell type to have a PCA score matrix on
    # every slide.  Silently skipping a missing entry is especially dangerous
    # when it occurs on the first slide because the optimizer infers its cell
    # types from that slide and can otherwise return an incomplete result.
    for slide in slides:
        slide_results = obj.pca_results.get(slide)
        for ct in cell_types:
            if slide_results is None or slide_results.get(ct) is None:
                raise ValueError(
                    f"Missing PCA data for slide '{slide}', cell type '{ct}'."
                )

    result = {slide: {} for slide in slides}
    for ct in cell_types:
        pca = obj.pca_global[ct]
        sdev = pca["sdev"]
        sdev_safe = sdev.copy()
        sdev_safe[sdev_safe < 1e-10] = 1.0
        for slide in slides:
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
    step_size: float = 1.0,
    sigma_choice: float | None = None,
    verbose: bool = True,
):
    """Run SkrCCA optimization for all sigma values.

    Dispatches to multi-slide version for CoProMulti objects.
    Stores one ``(n_pca, n_cc)`` weight matrix per sigma and cell type in
    ``obj.skr_cca_out``.
    """
    from .core import CoProMulti
    if isinstance(obj, CoProMulti):
        return _run_skr_cca_multi(
            obj, scale_pcs, n_cc, tol, max_iter,
            step_size=step_size, sigma_choice=sigma_choice, verbose=verbose,
        )

    # --- Single-slide path ---
    cts = obj.cell_types_of_interest
    if not cts:
        raise ValueError("No cell types of interest. Run subset_data() first.")
    if not obj.sigma_values:
        raise ValueError("No sigma values. Run compute_kernel_matrix() first.")
    if not obj.pca_global:
        raise ValueError("PCA results missing. Run compute_pca() first.")

    X_dict = _prepare_pc_matrices(obj, scale_pcs, cts)
    _validate_optimizer_args(X_dict, n_cc, tol, max_iter, step_size)
    sdev2_dict = None if scale_pcs else {
        ct: np.asarray(obj.pca_global[ct]["sdev"], dtype=float) ** 2
        for ct in cts
    }
    sigmas = obj.sigma_values
    if sigma_choice is not None:
        if sigma_choice not in sigmas:
            raise ValueError(
                f"sigma_choice={sigma_choice} is unavailable; choose from {sigmas}."
            )
        sigmas = [sigma_choice]

    cca_out = {}
    for sigma in sigmas:
        sigma_name = f"sigma_{sigma}"
        if verbose:
            print(f"Running SkrCCA for sigma = {sigma}")

        try:
            # First component
            w_dict = optimize_bilinear(
                X_dict=X_dict,
                flat_kernels=obj.kernel_matrices,
                sigma=sigma,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                step_size=step_size,
                sdev2_dict=sdev2_dict,
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
                    verbose=verbose,
                    step_size=step_size,
                    sdev2_dict=sdev2_dict,
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


def run_skr_cca_supervised(
    obj,
    supervised_weights: dict,
    scale_pcs: bool = True,
    n_cc: int = 4,
    tol: float = 1e-5,
    max_iter: int = 500,
    step_size: float = 1.0,
    verbose: bool = True,
):
    """Run SkrCCA with user-supplied first-component weights (supervised/guided mode).

    The first canonical component is fixed to the supplied weight vectors,
    and additional components (2 … *n_cc*) are optimised via the standard
    bilinear procedure while being constrained to be orthogonal to the
    supervised component.

    Parameters
    ----------
    obj : CoProSingle
        Must already have PCA results and kernel matrices.
    supervised_weights : dict
        ``{cell_type: w1_vector}`` — one 1-D array per cell type whose
        length matches the number of PCs.  Each vector will be
        L2-normalised internally.
    scale_pcs : bool
        Whether to scale PC scores by their standard deviation (matches
        R ``scale(pca$x, center=FALSE, scale=sdev)``).
    n_cc : int
        Total number of CCs (including the supervised first CC).
    tol : float
        Convergence tolerance for the optimiser.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    obj
        Updated in-place with ``skr_cca_out``, ``n_cc``, ``scale_pcs``.
    """
    cts = obj.cell_types_of_interest
    if not cts:
        raise ValueError("No cell types of interest. Run subset_data() first.")

    # Validate supervised_weights
    for ct in cts:
        if ct not in supervised_weights:
            raise ValueError(f"supervised_weights missing key '{ct}'.")

    X_dict = _prepare_pc_matrices(obj, scale_pcs, cts)
    _validate_optimizer_args(X_dict, n_cc, tol, max_iter, step_size)
    sdev2_dict = None if scale_pcs else {
        ct: np.asarray(obj.pca_global[ct]["sdev"], dtype=float) ** 2
        for ct in cts
    }
    obj.scale_pcs = scale_pcs
    obj.n_cc = n_cc

    cca_out = {}
    for sigma in obj.sigma_values:
        sigma_name = f"sigma_{sigma}"
        if verbose:
            print(f"Running supervised SkrCCA for sigma = {sigma}")

        # First component: use supervised weights (normalised)
        w_dict = {}
        for ct in cts:
            w = np.asarray(supervised_weights[ct], dtype=float).ravel()
            metric = None if sdev2_dict is None else sdev2_dict[ct]
            if metric is None:
                norm = np.linalg.norm(w)
            else:
                norm = np.sqrt(np.sum(w**2 * metric))
            if norm < 1e-12:
                raise ValueError(f"supervised_weights['{ct}'] has near-zero norm.")
            w_dict[ct] = (w / norm).reshape(-1, 1)

        # Additional components via standard optimization
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
                verbose=verbose,
                step_size=step_size,
                sdev2_dict=sdev2_dict,
            )

        cca_out[sigma_name] = w_dict

    obj.skr_cca_out = cca_out
    return obj


def _run_skr_cca_multi(
    obj, scale_pcs, n_cc, tol, max_iter,
    step_size=1.0, sigma_choice=None, verbose=True,
):
    """Multi-slide SkrCCA optimization. Shared weight vectors across slides."""
    from .optimization import optimize_bilinear_multi_slides, optimize_bilinear_n_multi_slides
    cts = obj.cell_types_of_interest
    slides = obj.slide_list
    X_list_all = _prepare_pc_matrices_multi(obj, scale_pcs, cts)
    first_slide = next((s for s in slides if X_list_all.get(s)), None)
    if first_slide is None:
        raise ValueError("No per-slide PCA matrices are available.")
    _validate_optimizer_args(X_list_all[first_slide], n_cc, tol, max_iter, step_size)
    sdev2_dict = None if scale_pcs else {
        ct: np.asarray(obj.pca_global[ct]["sdev"], dtype=float) ** 2
        for ct in cts
    }
    sigmas = obj.sigma_values
    if sigma_choice is not None:
        if sigma_choice not in sigmas:
            raise ValueError(
                f"sigma_choice={sigma_choice} is unavailable; choose from {sigmas}."
            )
        sigmas = [sigma_choice]

    cca_out = {}
    for sigma in sigmas:
        sigma_name = f"sigma_{sigma}"
        if verbose:
            print(f"Running SkrCCA (multi) for sigma = {sigma}")
        try:
            w_dict = optimize_bilinear_multi_slides(
                X_list_all, obj.kernel_matrices, sigma, slides, max_iter, tol,
                verbose=verbose, step_size=step_size, sdev2_dict=sdev2_dict,
            )
            if n_cc > 1:
                w_dict = optimize_bilinear_n_multi_slides(
                    X_list_all, obj.kernel_matrices, sigma, slides,
                    w_dict, cts, n_cc, max_iter, tol,
                    verbose=verbose, step_size=step_size,
                    sdev2_dict=sdev2_dict,
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


def _validate_optimizer_args(X_dict, n_cc, tol, max_iter, step_size):
    if not isinstance(n_cc, (int, np.integer)) or n_cc < 1:
        raise ValueError("n_cc must be a positive integer.")
    max_axes = min(np.asarray(X).shape[1] for X in X_dict.values())
    if n_cc > max_axes:
        raise ValueError(f"n_cc={n_cc} exceeds the available PC dimension ({max_axes}).")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not np.isfinite(step_size) or not 0 < step_size <= 1:
        raise ValueError("step_size must be in (0, 1].")
