"""Score transfer — apply gene weights from a reference CoPro to a target dataset."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .core import CoProSingle, CoProMulti
from .correlation import _spectral_norm, _get_kernel_for_pair


# ---------------------------------------------------------------------------
# Quantile normalization
# ---------------------------------------------------------------------------

def quantile_normalize(
    A: np.ndarray,
    B: np.ndarray,
    ties_method: str = "min",
    verbose: bool = True,
) -> np.ndarray:
    """Quantile-normalize B so that each column matches the distribution of A.

    Parameters
    ----------
    A : np.ndarray
        Reference matrix (n_ref × n_features). Defines target distributions.
    B : np.ndarray
        Matrix to normalize (n_tar × n_features). Must have same number of
        columns as A.
    ties_method : str
        How to handle ties when ranking. Passed to ``scipy.stats.rankdata``.
    verbose : bool
        Print progress messages.

    Returns
    -------
    np.ndarray
        Normalized B with shape (n_tar, n_features).
    """
    from scipy.stats import rankdata

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2-D arrays.")
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have the same number of columns.")
    if A.shape[0] < 2 or B.shape[0] < 2:
        raise ValueError("A and B must each have at least 2 rows.")

    n_ref = A.shape[0]
    n_tar = B.shape[0]
    n_features = A.shape[1]

    # Quantile positions for the reference (evenly spaced 0..1)
    prob_A = np.linspace(0, 1, n_ref)

    # Sort each column of A to define the reference distribution
    sorted_A = np.sort(A, axis=0)

    B_out = np.empty_like(B, dtype=float)

    for i in range(n_features):
        # Rank B column, convert to probabilities in [0, 1]
        ranks = rankdata(B[:, i], method=ties_method).astype(float)
        prob_B = (ranks - 1) / (n_tar - 1)

        # Interpolate B probabilities into A's sorted values
        B_out[:, i] = np.interp(prob_B, prob_A, sorted_A[:, i])

    if verbose:
        print(f"Quantile normalization done: {n_tar} target cells, {n_features} features.")

    return B_out


# ---------------------------------------------------------------------------
# Internal: transfer scores for one cell type
# ---------------------------------------------------------------------------

def _transfer_scores(
    mat_A: np.ndarray,
    mat_B: np.ndarray,
    gs_ct: np.ndarray,
    use_quantile_normalization: bool = True,
    gs_weight_threshold: float = 0.0,
    verbose: bool = True,
) -> np.ndarray:
    """Transfer gene weights from reference to target for one cell type.

    Parameters
    ----------
    mat_A : np.ndarray
        Reference expression matrix (n_ref × n_genes).
    mat_B : np.ndarray
        Target expression matrix (n_tar × n_genes). Same genes, same order.
    gs_ct : np.ndarray
        Gene score matrix (n_genes × n_cc) from the reference object.
    use_quantile_normalization : bool
        Quantile-normalize mat_B to mat_A before scoring.
    gs_weight_threshold : float
        Zero out gene weights with |weight| < threshold.
    verbose : bool
        Print progress messages.

    Returns
    -------
    np.ndarray
        Transferred cell scores (n_tar × n_cc).
    """
    # Optional quantile normalization
    if use_quantile_normalization:
        B_qn = quantile_normalize(mat_A, mat_B, verbose=verbose)
    else:
        B_qn = mat_B.astype(float)

    # Standardize B using reference mean and sd
    A_mean = mat_A.mean(axis=0)
    A_sd = mat_A.std(axis=0, ddof=1)       # sample std to match R's sd()
    A_sd_safe = A_sd.copy()
    A_sd_safe[np.isnan(A_sd_safe) | (A_sd_safe < 1e-8)] = 1.0

    B_cs = (B_qn - A_mean) / A_sd_safe

    # Threshold gene weights
    gs_filt = gs_ct.copy()
    for cc in range(gs_filt.shape[1]):
        small = np.abs(gs_filt[:, cc]) < gs_weight_threshold
        if verbose:
            n_retain = int(np.sum(~small))
            print(f"  retaining {n_retain} genes for CC_{cc+1} "
                  f"with threshold {gs_weight_threshold}")
        gs_filt[small, cc] = 0.0

    return B_cs @ gs_filt


# ---------------------------------------------------------------------------
# Public: get_transfer_cell_scores
# ---------------------------------------------------------------------------

def get_transfer_cell_scores(
    ref_obj,
    tar_obj,
    sigma_choice: float,
    use_quantile_normalization: bool = True,
    agg_cell_type: bool = False,
    gs_weight_threshold: float = 0.0,
    gene_score_type: str = "PCA",
    verbose: bool = True,
) -> dict | np.ndarray:
    """Transfer gene weights from a reference CoPro object to score target cells.

    Uses the gene scores learned on the reference to compute cell scores for
    every cell in the target object (e.g. another sample or single-cell data).

    Parameters
    ----------
    ref_obj : CoProSingle or CoProMulti
        Reference object (must have gene scores computed).
    tar_obj : CoProSingle or CoProMulti
        Target object (must have been through ``subset_data``).
    sigma_choice : float
        Sigma value to use (must exist in the reference gene scores).
    use_quantile_normalization : bool
        Quantile-normalize target to reference per-gene distribution.
    agg_cell_type : bool
        If True, return a single (n_cells × n_cc) array with rows ordered
        as in ``tar_obj.normalized_data_sub``. If False, return a dict
        keyed by cell type.
    gs_weight_threshold : float
        Zero out gene weights with absolute value below this threshold.
    gene_score_type : str
        ``"PCA"`` uses ``obj.gene_scores``; ``"regression"`` uses
        ``obj.gene_scores_regression`` (must call
        ``compute_regression_gene_scores`` first).
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict[str, np.ndarray] or np.ndarray
        If ``agg_cell_type=False``: ``{cell_type: (n_cells_ct, n_cc)}``.
        If ``agg_cell_type=True``: ``(n_cells_total, n_cc)`` ordered as in
        ``tar_obj.normalized_data_sub``.
    """
    # Validate objects
    if not isinstance(ref_obj, (CoProSingle, CoProMulti)):
        raise TypeError("ref_obj must be a CoProSingle or CoProMulti.")
    if not isinstance(tar_obj, (CoProSingle, CoProMulti)):
        raise TypeError("tar_obj must be a CoProSingle or CoProMulti.")

    # Select gene score source
    if gene_score_type == "regression":
        gs_all = getattr(ref_obj, "gene_scores_regression", {})
        if not gs_all:
            raise ValueError(
                "gene_scores_regression not found. "
                "Run compute_regression_gene_scores() on the reference first."
            )
    else:
        gs_all = ref_obj.gene_scores
        if not gs_all:
            raise ValueError(
                "gene_scores not found. "
                "Run compute_gene_and_cell_scores() on the reference first."
            )

    cts = ref_obj.cell_types_of_interest
    cts_tar = tar_obj.cell_types_of_interest
    if sorted(cts) != sorted(cts_tar):
        raise ValueError(
            f"Cell types mismatch: ref={cts}, tar={cts_tar}."
        )

    # Gene name intersection
    ref_genes = _get_gene_names(ref_obj)
    tar_genes = _get_gene_names(tar_obj)
    if ref_genes is not None and tar_genes is not None:
        if list(ref_genes) != list(tar_genes):
            genes_sel = sorted(set(ref_genes) & set(tar_genes))
            if len(genes_sel) == 0:
                raise ValueError("No overlapping genes between reference and target.")
            if verbose:
                print(f"Gene name mismatch — using {len(genes_sel)} shared genes "
                      f"(ref={len(ref_genes)}, tar={len(tar_genes)}).")
            ref_idx = np.array([list(ref_genes).index(g) for g in genes_sel])
            tar_idx = np.array([list(tar_genes).index(g) for g in genes_sel])
        else:
            ref_idx = tar_idx = None
    else:
        # No gene names — assume same ordering
        ref_idx = tar_idx = None

    B_cs = {}
    for ct in cts:
        # Gene scores for this cell type
        gs_key = f"geneScores|sigma{sigma_choice}|{ct}"
        if gs_key not in gs_all:
            raise KeyError(
                f"Gene score key '{gs_key}' not found. "
                f"Available keys: {list(gs_all.keys())}"
            )
        gs_ct = gs_all[gs_key]  # (n_genes, n_cc)

        # Expression matrices for this cell type
        ref_mask = ref_obj.cell_types_sub == ct
        tar_mask = tar_obj.cell_types_sub == ct
        mat_A = ref_obj.normalized_data_sub[ref_mask].astype(float)
        mat_B = tar_obj.normalized_data_sub[tar_mask].astype(float)

        # Subset to shared genes
        if ref_idx is not None:
            mat_A = mat_A[:, ref_idx]
            mat_B = mat_B[:, tar_idx]
            gs_ct = gs_ct[ref_idx]

        if verbose:
            print(f"Transferring scores for cell type '{ct}' "
                  f"({mat_B.shape[0]} target cells, {mat_A.shape[1]} genes)")

        B_cs[ct] = _transfer_scores(
            mat_A, mat_B, gs_ct,
            use_quantile_normalization=use_quantile_normalization,
            gs_weight_threshold=gs_weight_threshold,
            verbose=verbose,
        )

    if agg_cell_type:
        # Stack in the order of tar_obj.cell_types_sub
        n_cc = B_cs[cts[0]].shape[1]
        n_total = tar_obj.normalized_data_sub.shape[0]
        out = np.zeros((n_total, n_cc))
        for ct in cts:
            tar_mask = tar_obj.cell_types_sub == ct
            out[tar_mask] = B_cs[ct]
        return out

    return B_cs


# ---------------------------------------------------------------------------
# Public: get_transfer_norm_corr
# ---------------------------------------------------------------------------

def get_transfer_norm_corr(
    tar_obj,
    transfer_cell_scores: dict,
    sigma_choice: float,
    tol: float = 1e-4,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute normalized correlation from transferred cell scores.

    Parameters
    ----------
    tar_obj : CoProSingle or CoProMulti
        Target object (must have kernel matrices computed).
    transfer_cell_scores : dict
        ``{cell_type: np.ndarray (n_cells, n_cc)}`` from
        ``get_transfer_cell_scores(agg_cell_type=False)``.
    sigma_choice : float
        Sigma value for the target kernel.
    tol : float
        SVD tolerance for spectral norm.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        Columns: sigma, cell_type_1, cell_type_2, CC_index,
        normalized_correlation (and slideID for multi-slide).
    """
    if not isinstance(tar_obj, (CoProSingle, CoProMulti)):
        raise TypeError("tar_obj must be a CoProSingle or CoProMulti.")
    if not transfer_cell_scores:
        raise ValueError("transfer_cell_scores must be a non-empty dict.")

    cts = list(transfer_cell_scores.keys())
    n_cc = transfer_cell_scores[cts[0]].shape[1]

    # Pairs
    if len(cts) == 1:
        pairs = [(cts[0], cts[0])]
    else:
        pairs = list(combinations(cts, 2))

    is_multi = isinstance(tar_obj, CoProMulti)

    if not is_multi:
        return _transfer_norm_corr_single(
            tar_obj, transfer_cell_scores, sigma_choice,
            cts, pairs, n_cc, tol, verbose,
        )
    else:
        return _transfer_norm_corr_multi(
            tar_obj, transfer_cell_scores, sigma_choice,
            cts, pairs, n_cc, tol, verbose,
        )


def _transfer_norm_corr_single(
    tar_obj, scores, sigma, cts, pairs, n_cc, tol, verbose,
):
    """Single-slide transfer normalized correlation."""
    # Precompute spectral norms
    spec_norms = {}
    for ct_i, ct_j in pairs:
        try:
            K = _get_kernel_for_pair(tar_obj.kernel_matrices, sigma, ct_i, ct_j)
            spec_norms[(ct_i, ct_j)] = _spectral_norm(K, tol)
        except KeyError:
            spec_norms[(ct_i, ct_j)] = np.nan

    rows = []
    for ct_i, ct_j in pairs:
        try:
            K = _get_kernel_for_pair(tar_obj.kernel_matrices, sigma, ct_i, ct_j)
        except KeyError:
            continue
        norm_K = spec_norms.get((ct_i, ct_j), np.nan)
        if np.isnan(norm_K) or norm_K < 1e-9:
            continue

        A_scores = scores[ct_i]  # (n_A, n_cc)
        B_scores = scores[ct_j]  # (n_B, n_cc)

        for cc in range(n_cc):
            A_w = A_scores[:, cc]
            B_w = B_scores[:, cc]
            numerator = float(A_w @ K @ B_w)
            denom = float(np.linalg.norm(A_w) * np.linalg.norm(B_w) * norm_K)
            nc = 0.0 if abs(denom) < 1e-9 else numerator / denom
            rows.append({
                "sigma": sigma,
                "cell_type_1": ct_i,
                "cell_type_2": ct_j,
                "CC_index": cc + 1,
                "normalized_correlation": nc,
            })

    return pd.DataFrame(rows)


def _transfer_norm_corr_multi(
    tar_obj, scores, sigma, cts, pairs, n_cc, tol, verbose,
):
    """Multi-slide transfer normalized correlation."""
    slides = tar_obj.slide_list

    # Precompute per-slide spectral norms
    spec_norms = {}
    for ct_i, ct_j in pairs:
        for slide in slides:
            try:
                K = _get_kernel_for_pair(
                    tar_obj.kernel_matrices, sigma, ct_i, ct_j, slide
                )
                spec_norms[(ct_i, ct_j, slide)] = _spectral_norm(K, tol)
            except KeyError:
                spec_norms[(ct_i, ct_j, slide)] = np.nan

    # Build cell-to-slide mapping from tar_obj
    slide_ids = tar_obj.meta_data_sub["slideID"].values

    rows = []
    for ct_i, ct_j in pairs:
        for slide in slides:
            norm_K = spec_norms.get((ct_i, ct_j, slide), np.nan)
            if np.isnan(norm_K) or norm_K < 1e-9:
                continue
            try:
                K = _get_kernel_for_pair(
                    tar_obj.kernel_matrices, sigma, ct_i, ct_j, slide
                )
            except KeyError:
                continue

            # Get cells for this slide and type
            mask_i = (tar_obj.cell_types_sub == ct_i) & (slide_ids == slide)
            mask_j = (tar_obj.cell_types_sub == ct_j) & (slide_ids == slide)

            # Indices within the cell-type subset
            ct_i_mask = tar_obj.cell_types_sub == ct_i
            ct_j_mask = tar_obj.cell_types_sub == ct_j
            idx_i = np.where(mask_i[ct_i_mask])[0]
            idx_j = np.where(mask_j[ct_j_mask])[0]

            if len(idx_i) == 0 or len(idx_j) == 0:
                continue

            A_scores = scores[ct_i][idx_i]
            B_scores = scores[ct_j][idx_j]

            for cc in range(n_cc):
                A_w = A_scores[:, cc]
                B_w = B_scores[:, cc]
                numerator = float(A_w @ K @ B_w)
                denom = float(np.linalg.norm(A_w) * np.linalg.norm(B_w) * norm_K)
                nc = 0.0 if abs(denom) < 1e-9 else numerator / denom
                rows.append({
                    "sigma": sigma,
                    "slideID": slide,
                    "cell_type_1": ct_i,
                    "cell_type_2": ct_j,
                    "CC_index": cc + 1,
                    "normalized_correlation": nc,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public: get_transfer_bidir_corr
# ---------------------------------------------------------------------------

def get_transfer_bidir_corr(
    tar_obj,
    transfer_cell_scores: dict,
    sigma_choice: float,
    normalize_K: str = "row_or_col",
    filter_kernel: bool = True,
    K_row_sum_cutoff: float = 5e-3,
    K_col_sum_cutoff: float = 5e-3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute bidirectional correlation from transferred cell scores.

    bidir_corr = mean( cor(K^T @ A, B),  cor(A, K @ B) )

    Parameters
    ----------
    tar_obj : CoProSingle or CoProMulti
        Target object with kernel matrices.
    transfer_cell_scores : dict
        ``{cell_type: np.ndarray (n_cells, n_cc)}``.
    sigma_choice : float
        Sigma value for the kernel.
    normalize_K : str
        Kernel normalization: ``"row_or_col"``, ``"sinkhorn_knopp"``, or
        ``"none"``.
    filter_kernel : bool
        Remove rows/columns with very low kernel sums.
    K_row_sum_cutoff : float
        Minimum row sum threshold.
    K_col_sum_cutoff : float
        Minimum column sum threshold.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns: sigma, cell_type_1, cell_type_2, CC_index, bidir_correlation.
    """
    cts = list(transfer_cell_scores.keys())
    n_cc = transfer_cell_scores[cts[0]].shape[1]

    if len(cts) == 1:
        pairs = [(cts[0], cts[0])]
    else:
        pairs = list(combinations(cts, 2))

    rows = []
    for ct_i, ct_j in pairs:
        try:
            K = _get_kernel_for_pair(
                tar_obj.kernel_matrices, sigma_choice, ct_i, ct_j
            )
        except KeyError:
            continue

        A_all = transfer_cell_scores[ct_i]  # (n_A, n_cc)
        B_all = transfer_cell_scores[ct_j]  # (n_B, n_cc)

        # Filter kernel
        if filter_kernel:
            row_keep = K.sum(axis=1) > K_row_sum_cutoff
            K = K[row_keep]
            A_all = A_all[row_keep]
            col_keep = K.sum(axis=0) > K_col_sum_cutoff
            K = K[:, col_keep]
            B_all = B_all[col_keep]

        corrs = _compute_bidir_correlations(A_all, B_all, K, normalize_K)

        for cc in range(n_cc):
            rows.append({
                "sigma": sigma_choice,
                "cell_type_1": ct_i,
                "cell_type_2": ct_j,
                "CC_index": cc + 1,
                "bidir_correlation": corrs[cc],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_bidir_correlations(
    A_all: np.ndarray,
    B_all: np.ndarray,
    K: np.ndarray,
    normalize_K: str,
) -> np.ndarray:
    """Compute bidirectional correlations for all CCs.

    Returns array of length n_cc.
    """
    n_cc = A_all.shape[1]

    if normalize_K == "row_or_col":
        rs = K.sum(axis=1, keepdims=True)
        rs[rs < 1e-12] = 1.0
        K_row = K / rs
        cs = K.sum(axis=0, keepdims=True)
        cs[cs < 1e-12] = 1.0
        K_col = K / cs
        KA = K_row.T @ A_all     # (n_B, n_cc)
        KB = K_col @ B_all       # (n_A, n_cc)
    elif normalize_K == "sinkhorn_knopp":
        K_norm = _sinkhorn_knopp(K)
        KA = K_norm.T @ A_all
        KB = K_norm @ B_all
    else:  # "none"
        KA = K.T @ A_all
        KB = K @ B_all

    corrs = np.zeros(n_cc)
    for cc in range(n_cc):
        cor1 = _safe_pearsonr(KA[:, cc], B_all[:, cc])
        cor2 = _safe_pearsonr(A_all[:, cc], KB[:, cc])
        corrs[cc] = (cor1 + cor2) / 2.0

    return corrs


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, returning 0.0 if either vector has zero variance."""
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
        # Check convergence
        if (np.abs(K.sum(axis=1) - 1).max() < tol and
                np.abs(K.sum(axis=0) - 1).max() < tol):
            break
    return K


def _get_gene_names(obj) -> list | None:
    """Try to get gene names from the object's metadata or PCA."""
    if hasattr(obj, "gene_names") and obj.gene_names is not None:
        return list(obj.gene_names)
    # If normalized_data was from a DataFrame, column names may be available
    if hasattr(obj, "gene_list") and obj.gene_list is not None:
        return list(obj.gene_list)
    return None
