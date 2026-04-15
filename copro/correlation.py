"""compute_normalized_correlation() — spectral-norm normalized CCA correlation."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from .core import CoProSingle
from .skrcca import _prepare_pc_matrices


def _spectral_norm(K: np.ndarray, tol: float = 1e-4) -> float:
    """Largest singular value of K (spectral norm)."""
    try:
        s = svds(K.astype(float), k=1, tol=tol, return_singular_vectors=False)
        return float(s[0])
    except Exception:
        return float(np.linalg.norm(K, ord=2))



def _get_kernel_for_pair(flat_kernels, sigma, ct_i, ct_j, slide=None):
    """Retrieve kernel, optionally slide-aware, trying both orderings."""
    if slide is None:
        name = f"kernel|sigma{sigma}|{ct_i}|{ct_j}"
        name_sym = f"kernel|sigma{sigma}|{ct_j}|{ct_i}"
    else:
        name = f"kernel|sigma{sigma}|{slide}|{ct_i}|{ct_j}"
        name_sym = f"kernel|sigma{sigma}|{slide}|{ct_j}|{ct_i}"
    if name in flat_kernels:
        return flat_kernels[name]
    if name_sym in flat_kernels:
        return flat_kernels[name_sym].T
    raise KeyError(f"Kernel not found for ({ct_i},{ct_j}) sigma={sigma} slide={slide}")


def compute_normalized_correlation(obj, tol: float = 1e-4):
    """Compute normalized CCA correlation for each sigma × pair × CC.

    Dispatches to multi-slide version for CoProMulti objects.

    Formula:
        numerator   = (A @ w1)^T K (B @ w2)
        denominator = ||A @ w1|| * ||B @ w2|| * ||K||_spec
        norm_corr   = numerator / denominator

    Stores in obj.normalized_correlation[sigma_name] = DataFrame.
    Chooses obj.sigma_value_choice as sigma maximizing mean CC1 correlation.
    """
    from .core import CoProMulti
    if isinstance(obj, CoProMulti):
        return _compute_normalized_correlation_multi(obj, tol)

    # --- Single-slide path ---
    cts = obj.cell_types_of_interest
    if not cts:
        raise ValueError("No cell types of interest.")
    if not obj.skr_cca_out:
        raise ValueError("CCA results missing. Run run_skr_cca() first.")

    scale_pcs = getattr(obj, "scale_pcs", True)
    n_cc = obj.n_cc

    # Scaled PC matrices
    X_dict = _prepare_pc_matrices(obj, scale_pcs, cts)

    # Pairs
    if len(cts) == 1:
        pairs = [(cts[0], cts[0])]
    else:
        pairs = list(combinations(cts, 2))

    print("Calculating spectral norms (may take a while)...")

    # Precompute spectral norms for each sigma × pair
    spec_norms = {}
    for sigma in obj.sigma_values:
        spec_norms[sigma] = {}
        for ct_i, ct_j in pairs:
            try:
                K = _get_kernel_for_pair(obj.kernel_matrices, sigma, ct_i, ct_j)
                spec_norms[sigma][(ct_i, ct_j)] = _spectral_norm(K, tol=tol)
                spec_norms[sigma][(ct_j, ct_i)] = spec_norms[sigma][(ct_i, ct_j)]
            except KeyError:
                spec_norms[sigma][(ct_i, ct_j)] = np.nan

    print("Finished calculating spectral norms.")

    correlation_value = {}

    for sigma in obj.sigma_values:
        sigma_name = f"sigma_{sigma}"
        w_sigma = obj.skr_cca_out.get(sigma_name)
        if w_sigma is None:
            continue

        rows = []
        for ct_i, ct_j in pairs:
            A = X_dict[ct_i]
            B = X_dict[ct_j]
            try:
                K = _get_kernel_for_pair(obj.kernel_matrices, sigma, ct_i, ct_j)
            except KeyError:
                continue
            norm_K = spec_norms[sigma].get((ct_i, ct_j), np.nan)

            for cc in range(n_cc):
                w1 = w_sigma[ct_i][:, cc : cc + 1]
                w2 = w_sigma[ct_j][:, cc : cc + 1]

                Aw1 = A @ w1
                Bw2 = B @ w2

                numerator = float((Aw1.T @ K @ Bw2).flat[0])
                denom = float(np.sqrt(np.sum(Aw1 ** 2))) * float(np.sqrt(np.sum(Bw2 ** 2))) * norm_K

                norm_corr = 0.0 if abs(denom) < 1e-9 else numerator / denom

                rows.append({
                    "sigma": sigma,
                    "cell_type_1": ct_i,
                    "cell_type_2": ct_j,
                    "CC_index": cc + 1,
                    "normalized_correlation": norm_corr,
                })

        correlation_value[sigma_name] = pd.DataFrame(rows)

    obj.normalized_correlation = correlation_value

    # Choose sigma maximizing mean CC1 correlation
    all_cc1 = []
    for sigma_name, df in correlation_value.items():
        if df is not None and len(df) > 0:
            cc1 = df[df["CC_index"] == 1]
            mean_corr = cc1["normalized_correlation"].mean()
            sigma_val = float(sigma_name.replace("sigma_", ""))
            all_cc1.append((sigma_val, mean_corr))

    if all_cc1:
        obj.sigma_value_choice = max(all_cc1, key=lambda x: x[1])[0]

    return obj


def _compute_normalized_correlation_multi(obj, tol=1e-4):
    """Multi-slide normalized correlation: per-slide values matching R format.

    R computes normalized correlation independently for each slide using the
    raw (unscaled) per-slide PCA scores from pcaResults (not scaled by sdev).
    We replicate this: for each (sigma, slide, pair, CC), compute norm_corr
    using only that slide's raw PCA scores and per-slide spectral norm.
    Sigma choice is based on the mean CC1 correlation across slides.
    """
    cts = obj.cell_types_of_interest
    slides = obj.slide_list
    n_cc = obj.n_cc

    # Use raw (unscaled) per-slide PCA scores — matching R's pcaResults usage
    X_list_all = {
        slide: {ct: obj.pca_results[slide][ct].astype(float)
                for ct in cts if ct in obj.pca_results.get(slide, {})}
        for slide in slides
    }

    if len(cts) == 1:
        pairs = [(cts[0], cts[0])]
    else:
        pairs = list(combinations(cts, 2))

    # Precompute per-slide spectral norms for each sigma × pair
    print("Calculating spectral norms (multi-slide)...")
    spec_norms = {}  # spec_norms[sigma][(ct_i, ct_j, slide)]
    for sigma in obj.sigma_values:
        spec_norms[sigma] = {}
        for ct_i, ct_j in pairs:
            for slide in slides:
                try:
                    K = _get_kernel_for_pair(obj.kernel_matrices, sigma, ct_i, ct_j, slide)
                    val = _spectral_norm(K, tol)
                except KeyError:
                    val = np.nan
                spec_norms[sigma][(ct_i, ct_j, slide)] = val
                spec_norms[sigma][(ct_j, ct_i, slide)] = val
    print("Finished spectral norms.")

    correlation_value = {}

    for sigma in obj.sigma_values:
        sigma_name = f"sigma_{sigma}"
        w_sigma = obj.skr_cca_out.get(sigma_name)
        if w_sigma is None:
            continue

        rows = []
        for ct_i, ct_j in pairs:
            for cc in range(n_cc):
                w1 = w_sigma[ct_i][:, cc:cc+1]
                w2 = w_sigma[ct_j][:, cc:cc+1]

                # Per-slide correlation (matches R format)
                for slide in slides:
                    A = X_list_all[slide].get(ct_i)
                    B = X_list_all[slide].get(ct_j)
                    if A is None or B is None:
                        continue
                    try:
                        K = _get_kernel_for_pair(obj.kernel_matrices, sigma, ct_i, ct_j, slide)
                    except KeyError:
                        continue

                    norm_K = spec_norms[sigma].get((ct_i, ct_j, slide), np.nan)
                    Aw1 = A @ w1
                    Bw2 = B @ w2
                    numerator = float((Aw1.T @ K @ Bw2).flat[0])
                    denom = (float(np.linalg.norm(Aw1)) *
                             float(np.linalg.norm(Bw2)) *
                             norm_K)
                    norm_corr = 0.0 if abs(denom) < 1e-9 else numerator / denom

                    rows.append({
                        "sigma": sigma,
                        "slideID": slide,
                        "cell_type_1": ct_i,
                        "cell_type_2": ct_j,
                        "CC_index": cc + 1,
                        "normalized_correlation": norm_corr,
                    })

        correlation_value[sigma_name] = pd.DataFrame(rows)

    obj.normalized_correlation = correlation_value

    # Choose sigma maximizing mean CC1 correlation across slides
    all_cc1 = []
    for sigma_name, df in correlation_value.items():
        if df is not None and len(df) > 0:
            cc1 = df[df["CC_index"] == 1]
            mean_corr = cc1["normalized_correlation"].mean()
            sigma_val = float(sigma_name.replace("sigma_", ""))
            all_cc1.append((sigma_val, mean_corr))
    if all_cc1:
        obj.sigma_value_choice = max(all_cc1, key=lambda x: x[1])[0]

    return obj


# ---------------------------------------------------------------------------
# Bidirectional correlation
# ---------------------------------------------------------------------------


def compute_bidir_correlation(
    obj,
    normalize_K: str = "row_or_col",
    filter_kernel: bool = True,
    K_row_sum_cutoff: float = 5e-3,
    K_col_sum_cutoff: float = 5e-3,
):
    """Compute bidirectional correlation for each sigma × pair × CC.

    bidir_corr = mean( cor(K^T @ A_w, B_w),  cor(A_w, K @ B_w) )

    Unlike normalized correlation (which uses spectral norm), this metric
    uses standard Pearson correlation of the kernel-smoothed scores.

    Stores results in ``obj.bidir_correlation[sigma_name]`` as DataFrames.

    Parameters
    ----------
    obj : CoProSingle or CoProMulti
        Must have CCA weights computed.
    normalize_K : str
        ``"row_or_col"``, ``"sinkhorn_knopp"``, or ``"none"``.
    filter_kernel : bool
        Remove rows/columns with low kernel sums.
    K_row_sum_cutoff : float
        Minimum row sum threshold when filtering.
    K_col_sum_cutoff : float
        Minimum column sum threshold when filtering.

    Returns
    -------
    obj
        With ``bidir_correlation`` dict populated.
    """
    from .core import CoProMulti
    if isinstance(obj, CoProMulti):
        return _compute_bidir_correlation_multi(
            obj, normalize_K, filter_kernel,
            K_row_sum_cutoff, K_col_sum_cutoff,
        )

    cts = obj.cell_types_of_interest
    if not obj.skr_cca_out:
        raise ValueError("CCA results missing. Run run_skr_cca() first.")

    scale_pcs = getattr(obj, "scale_pcs", True)
    n_cc = obj.n_cc
    X_dict = _prepare_pc_matrices(obj, scale_pcs, cts)

    if len(cts) == 1:
        pairs = [(cts[0], cts[0])]
    else:
        pairs = list(combinations(cts, 2))

    bidir_corr = {}
    for sigma in obj.sigma_values:
        sigma_name = f"sigma_{sigma}"
        w_sigma = obj.skr_cca_out.get(sigma_name)
        if w_sigma is None:
            continue

        rows = []
        for ct_i, ct_j in pairs:
            A = X_dict[ct_i]
            B = X_dict[ct_j]
            try:
                K = _get_kernel_for_pair(obj.kernel_matrices, sigma, ct_i, ct_j)
            except KeyError:
                continue

            W_i = w_sigma[ct_i]  # (n_pca, n_cc)
            W_j = w_sigma[ct_j]

            A_all = A @ W_i  # (n_A, n_cc)
            B_all = B @ W_j  # (n_B, n_cc)

            # Filter kernel
            K_use = K.copy()
            A_use = A_all.copy()
            B_use = B_all.copy()
            if filter_kernel:
                row_keep = K_use.sum(axis=1) > K_row_sum_cutoff
                K_use = K_use[row_keep]
                A_use = A_use[row_keep]
                col_keep = K_use.sum(axis=0) > K_col_sum_cutoff
                K_use = K_use[:, col_keep]
                B_use = B_use[col_keep]

            corrs = _compute_bidir_corrs_all_cc(
                A_use, B_use, K_use, normalize_K
            )

            for cc in range(n_cc):
                rows.append({
                    "sigma": sigma,
                    "cell_type_1": ct_i,
                    "cell_type_2": ct_j,
                    "CC_index": cc + 1,
                    "bidir_correlation": corrs[cc],
                })

        bidir_corr[sigma_name] = pd.DataFrame(rows)

    obj.bidir_correlation = bidir_corr
    return obj


def _compute_bidir_correlation_multi(
    obj, normalize_K, filter_kernel, K_row_sum_cutoff, K_col_sum_cutoff,
):
    """Multi-slide bidirectional correlation."""
    cts = obj.cell_types_of_interest
    slides = obj.slide_list
    n_cc = obj.n_cc

    # Raw per-slide PCA scores
    X_list_all = {
        slide: {ct: obj.pca_results[slide][ct].astype(float)
                for ct in cts if ct in obj.pca_results.get(slide, {})}
        for slide in slides
    }

    if len(cts) == 1:
        pairs = [(cts[0], cts[0])]
    else:
        pairs = list(combinations(cts, 2))

    bidir_corr = {}
    for sigma in obj.sigma_values:
        sigma_name = f"sigma_{sigma}"
        w_sigma = obj.skr_cca_out.get(sigma_name)
        if w_sigma is None:
            continue

        rows = []
        for ct_i, ct_j in pairs:
            W_i = w_sigma[ct_i]
            W_j = w_sigma[ct_j]

            for slide in slides:
                A_raw = X_list_all[slide].get(ct_i)
                B_raw = X_list_all[slide].get(ct_j)
                if A_raw is None or B_raw is None:
                    continue
                try:
                    K = _get_kernel_for_pair(
                        obj.kernel_matrices, sigma, ct_i, ct_j, slide
                    )
                except KeyError:
                    continue

                A_all = A_raw @ W_i
                B_all = B_raw @ W_j

                K_use = K.copy()
                A_use = A_all.copy()
                B_use = B_all.copy()
                if filter_kernel:
                    row_keep = K_use.sum(axis=1) > K_row_sum_cutoff
                    K_use = K_use[row_keep]
                    A_use = A_use[row_keep]
                    col_keep = K_use.sum(axis=0) > K_col_sum_cutoff
                    K_use = K_use[:, col_keep]
                    B_use = B_use[col_keep]

                corrs = _compute_bidir_corrs_all_cc(
                    A_use, B_use, K_use, normalize_K
                )

                for cc in range(n_cc):
                    rows.append({
                        "sigma": sigma,
                        "slideID": slide,
                        "cell_type_1": ct_i,
                        "cell_type_2": ct_j,
                        "CC_index": cc + 1,
                        "bidir_correlation": corrs[cc],
                    })

        bidir_corr[sigma_name] = pd.DataFrame(rows)

    obj.bidir_correlation = bidir_corr
    return obj


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, 0.0 if either vector has zero variance."""
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def _sinkhorn_knopp(
    K: np.ndarray, max_iter: int = 100, tol: float = 1e-6,
) -> np.ndarray:
    """Sinkhorn-Knopp doubly-stochastic normalization."""
    K = K.copy().astype(float)
    K[K < 0] = 0.0
    for _ in range(max_iter):
        rs = K.sum(axis=1, keepdims=True)
        rs[rs < 1e-12] = 1.0
        K = K / rs
        cs = K.sum(axis=0, keepdims=True)
        cs[cs < 1e-12] = 1.0
        K = K / cs
        if (np.abs(K.sum(axis=1) - 1).max() < tol and
                np.abs(K.sum(axis=0) - 1).max() < tol):
            break
    return K


def _compute_bidir_corrs_all_cc(
    A_all: np.ndarray,
    B_all: np.ndarray,
    K: np.ndarray,
    normalize_K: str,
) -> np.ndarray:
    """Compute bidirectional correlation for all CCs."""
    n_cc = A_all.shape[1]

    if normalize_K == "row_or_col":
        rs = K.sum(axis=1, keepdims=True)
        rs[rs < 1e-12] = 1.0
        K_row = K / rs
        cs = K.sum(axis=0, keepdims=True)
        cs[cs < 1e-12] = 1.0
        K_col = K / cs
        KA = K_row.T @ A_all
        KB = K_col @ B_all
    elif normalize_K == "sinkhorn_knopp":
        K_norm = _sinkhorn_knopp(K)
        KA = K_norm.T @ A_all
        KB = K_norm @ B_all
    else:
        KA = K.T @ A_all
        KB = K @ B_all

    corrs = np.zeros(n_cc)
    for cc in range(n_cc):
        cor1 = _safe_pearsonr(KA[:, cc], B_all[:, cc])
        cor2 = _safe_pearsonr(A_all[:, cc], KB[:, cc])
        corrs[cc] = (cor1 + cor2) / 2.0
    return corrs
