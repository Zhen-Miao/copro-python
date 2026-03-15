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
