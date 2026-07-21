"""Score transfer — apply gene weights from a reference CoPro to a target dataset."""

from __future__ import annotations

from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from scipy import sparse

from .core import CoProSingle, CoProMulti
from .correlation import (
    _compute_bidir_corrs_all_cc,
    _filter_cross_kernel,
    _get_kernel_for_pair,
    _kernel_normalizer,
)


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
        ``"PCA"`` uses PCA-backprojected scores, ``"gene_space"`` uses
        weights stored directly by :func:`run_gene_space_cca`, and
        ``"regression"`` uses ``obj.gene_scores_regression``.
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
    score_type = str(gene_score_type).lower().replace("-", "_")
    if score_type == "regression":
        gs_all = getattr(ref_obj, "gene_scores_regression", {})
        if not gs_all:
            raise ValueError(
                "gene_scores_regression not found. "
                "Run compute_regression_gene_scores() on the reference first."
            )
    elif score_type in {"pca", "gene_space", "genespace", "gscca"}:
        gs_all = ref_obj.gene_scores
        if not gs_all:
            raise ValueError(
                "gene_scores not found. "
                "Run compute_gene_and_cell_scores() on the reference first."
            )
    else:
        raise ValueError("gene_score_type must be 'PCA', 'gene_space', or 'regression'.")

    cts = ref_obj.cell_types_of_interest
    cts_tar = tar_obj.cell_types_of_interest
    if sorted(cts) != sorted(cts_tar):
        raise ValueError(
            f"Cell types mismatch: ref={cts}, tar={cts_tar}."
        )

    # Full expression-space gene names. Gene-space CCA may have filtered its
    # weights to a smaller explicitly named set, handled per cell type below.
    ref_genes = _get_gene_names(ref_obj)
    tar_genes = _get_gene_names(tar_obj)
    ref_genes = None if ref_genes is None else list(ref_genes)
    tar_genes = None if tar_genes is None else list(tar_genes)

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
        if sparse.issparse(mat_A):
            mat_A = mat_A.toarray()
        if sparse.issparse(mat_B):
            mat_B = mat_B.toarray()

        if ref_genes is not None and tar_genes is not None:
            if gs_ct.shape[0] == len(ref_genes):
                score_genes = ref_genes
            else:
                score_genes = list(getattr(ref_obj, "gene_space_genes", []))
                if len(score_genes) != gs_ct.shape[0]:
                    raise ValueError(
                        f"Gene-score rows ({gs_ct.shape[0]}) cannot be aligned to "
                        "the reference expression genes."
                    )
            tar_lookup = {gene: idx for idx, gene in enumerate(tar_genes)}
            ref_lookup = {gene: idx for idx, gene in enumerate(ref_genes)}
            shared = [
                gene for gene in score_genes
                if gene in ref_lookup and gene in tar_lookup
            ]
            if not shared:
                raise ValueError("No overlapping genes between reference and target.")
            score_lookup = {gene: idx for idx, gene in enumerate(score_genes)}
            ref_idx = np.asarray([ref_lookup[gene] for gene in shared], dtype=int)
            tar_idx = np.asarray([tar_lookup[gene] for gene in shared], dtype=int)
            score_idx = np.asarray([score_lookup[gene] for gene in shared], dtype=int)
            mat_A = mat_A[:, ref_idx]
            mat_B = mat_B[:, tar_idx]
            gs_ct = gs_ct[score_idx]
            if verbose and (
                len(shared) != len(ref_genes) or ref_genes != tar_genes
            ):
                print(
                    f"Gene alignment — using {len(shared)} shared genes "
                    f"(ref={len(ref_genes)}, tar={len(tar_genes)})."
                )
        elif gs_ct.shape[0] != mat_A.shape[1]:
            indices = getattr(ref_obj, "gene_space_gene_indices", None)
            if indices is None or len(indices) != gs_ct.shape[0]:
                raise ValueError(
                    "Filtered gene-space weights require gene names or valid "
                    "gene_space_gene_indices for score transfer."
                )
            indices = np.asarray(indices, dtype=int)
            if indices.size and indices.max() >= mat_B.shape[1]:
                raise ValueError("Target expression does not contain all filtered genes.")
            mat_A = mat_A[:, indices]
            mat_B = mat_B[:, indices]

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
    calculation_mode: str | None = None,
    sigma_choice_tar: float | None = None,
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
        Reference sigma used to label the transferred result.
    tol : float
        Retained for API compatibility; the current normalizer is exact.
    verbose : bool
        Print progress messages.
    calculation_mode : str or None
        For ``CoProMulti``, ``"per_slide"`` (default) or ``"aggregate"``.
        The R spelling ``"perSlide"`` is also accepted. Ignored for single
        slide targets.
    sigma_choice_tar : float or None
        Sigma used to select target kernels. Defaults to ``sigma_choice``.

    Returns
    -------
    pd.DataFrame
        Columns: sigma, cell_type_1, cell_type_2, CC_index,
        Per-slide output contains ``normalized_correlation`` and ``slideID``.
        Aggregate output contains ``aggregate_correlation``.
    """
    cts, pairs, n_cc = _validate_transfer_correlation_inputs(
        tar_obj, transfer_cell_scores, sigma_choice
    )
    target_sigma = _resolve_target_sigma(sigma_choice, sigma_choice_tar)
    mode = _resolve_transfer_calculation_mode(tar_obj, calculation_mode)

    if isinstance(tar_obj, CoProSingle):
        return _transfer_norm_corr_single(
            tar_obj, transfer_cell_scores, sigma_choice, target_sigma,
            cts, pairs, n_cc, tol, verbose,
        )
    return _transfer_norm_corr_multi(
        tar_obj, transfer_cell_scores, sigma_choice, target_sigma,
        cts, pairs, n_cc, tol, verbose, mode,
    )


def _transfer_norm_corr_single(
    tar_obj, scores, sigma, target_sigma, cts, pairs, n_cc, tol, verbose,
):
    """Single-slide transfer normalized correlation."""
    del tol, verbose
    # Precompute matched-sigma whitened-Frobenius normalizers.
    kernel_norms = {}
    for ct_i, ct_j in pairs:
        try:
            kernel_norms[(ct_i, ct_j)] = _kernel_normalizer(
                tar_obj.kernel_matrices, target_sigma, ct_i, ct_j
            )
        except KeyError:
            kernel_norms[(ct_i, ct_j)] = np.nan

    rows = []
    for ct_i, ct_j in pairs:
        try:
            K = _get_kernel_for_pair(
                tar_obj.kernel_matrices, target_sigma, ct_i, ct_j
            )
        except KeyError:
            continue
        norm_K = kernel_norms.get((ct_i, ct_j), np.nan)
        if np.isnan(norm_K) or norm_K < 1e-9:
            continue

        A_scores = scores[ct_i]  # (n_A, n_cc)
        B_scores = scores[ct_j]  # (n_B, n_cc)
        _validate_kernel_score_shapes(K, A_scores, B_scores, ct_i, ct_j)

        for cc in range(n_cc):
            A_w = A_scores[:, cc]
            B_w = B_scores[:, cc]
            numerator = float(A_w @ (K @ B_w))
            denom = float(np.linalg.norm(A_w) * np.linalg.norm(B_w) * norm_K)
            nc = 0.0 if abs(denom) < 1e-9 else numerator / denom
            rows.append({
                "sigma": sigma,
                "cell_type_1": ct_i,
                "cell_type_2": ct_j,
                "CC_index": cc + 1,
                "normalized_correlation": nc,
            })

    return pd.DataFrame(rows, columns=[
        "sigma", "cell_type_1", "cell_type_2", "CC_index",
        "normalized_correlation",
    ])


def _transfer_norm_corr_multi(
    tar_obj, scores, sigma, target_sigma, cts, pairs, n_cc, tol, verbose,
    calculation_mode,
):
    """Multi-slide transfer normalized correlation."""
    del tol, verbose
    slides = _target_slides(tar_obj)

    # Precompute per-slide matched-sigma normalizers.
    kernel_norms = {}
    for ct_i, ct_j in pairs:
        for slide in slides:
            try:
                kernel_norms[(ct_i, ct_j, slide)] = _kernel_normalizer(
                    tar_obj.kernel_matrices, target_sigma, ct_i, ct_j, slide
                )
            except KeyError:
                kernel_norms[(ct_i, ct_j, slide)] = np.nan

    if calculation_mode == "per_slide":
        rows = []
        for ct_i, ct_j in pairs:
            for slide in slides:
                norm_K = kernel_norms.get((ct_i, ct_j, slide), np.nan)
                if not np.isfinite(norm_K) or norm_K < 1e-9:
                    continue
                inputs = _transfer_slide_inputs(
                    tar_obj, scores, target_sigma, slide, ct_i, ct_j
                )
                if inputs is None:
                    continue
                K, A_scores, B_scores = inputs
                for cc in range(n_cc):
                    A_w = A_scores[:, cc]
                    B_w = B_scores[:, cc]
                    numerator = float(A_w @ (K @ B_w))
                    denom = float(
                        np.linalg.norm(A_w) * np.linalg.norm(B_w) * norm_K
                    )
                    value = 0.0 if abs(denom) < 1e-9 else numerator / denom
                    rows.append({
                        "sigma": sigma,
                        "slideID": slide,
                        "cell_type_1": ct_i,
                        "cell_type_2": ct_j,
                        "CC_index": cc + 1,
                        "normalized_correlation": value,
                    })
        return pd.DataFrame(rows, columns=[
            "sigma", "slideID", "cell_type_1", "cell_type_2", "CC_index",
            "normalized_correlation",
        ])

    rows = []
    for ct_i, ct_j in pairs:
        for cc in range(n_cc):
            total_numerator = 0.0
            total_norm_i = 0.0
            total_norm_j = 0.0
            total_kernel_norm = 0.0
            valid_slides = 0
            for slide in slides:
                norm_K = kernel_norms.get((ct_i, ct_j, slide), np.nan)
                if not np.isfinite(norm_K) or norm_K < 1e-9:
                    continue
                inputs = _transfer_slide_inputs(
                    tar_obj, scores, target_sigma, slide, ct_i, ct_j
                )
                if inputs is None:
                    continue
                K, A_scores, B_scores = inputs
                A_w = A_scores[:, cc]
                B_w = B_scores[:, cc]
                total_numerator += float(A_w @ (K @ B_w))
                total_norm_i += float(np.sum(A_w**2))
                total_norm_j += float(np.sum(B_w**2))
                total_kernel_norm += float(norm_K)
                valid_slides += 1
            if valid_slides:
                denom = (
                    np.sqrt(total_norm_i)
                    * np.sqrt(total_norm_j)
                    * (total_kernel_norm / valid_slides)
                )
                value = 0.0 if abs(denom) < 1e-9 else total_numerator / denom
                rows.append({
                    "sigma": sigma,
                    "cell_type_1": ct_i,
                    "cell_type_2": ct_j,
                    "CC_index": cc + 1,
                    "aggregate_correlation": value,
                })
    return pd.DataFrame(rows, columns=[
        "sigma", "cell_type_1", "cell_type_2", "CC_index",
        "aggregate_correlation",
    ])


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
    calculation_mode: str | None = None,
    sigma_choice_tar: float | None = None,
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
        Reference sigma used to label the transferred result.
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
    calculation_mode : str or None
        For ``CoProMulti``, ``"per_slide"`` (default) or ``"aggregate"``.
        The R spelling ``"perSlide"`` is also accepted.
    sigma_choice_tar : float or None
        Sigma used to select target kernels. Defaults to ``sigma_choice``.

    Returns
    -------
    pd.DataFrame
        Per-slide output contains ``bidir_correlation`` and ``slideID``.
        Aggregate output contains ``aggregate_correlation``.
    """
    del verbose
    if normalize_K not in {"row_or_col", "sinkhorn_knopp", "none"}:
        raise ValueError(
            "normalize_K must be 'row_or_col', 'sinkhorn_knopp', or 'none'."
        )
    cts, pairs, n_cc = _validate_transfer_correlation_inputs(
        tar_obj, transfer_cell_scores, sigma_choice
    )
    target_sigma = _resolve_target_sigma(sigma_choice, sigma_choice_tar)
    mode = _resolve_transfer_calculation_mode(tar_obj, calculation_mode)

    if isinstance(tar_obj, CoProSingle):
        rows = []
        for ct_i, ct_j in pairs:
            try:
                K = _get_kernel_for_pair(
                    tar_obj.kernel_matrices, target_sigma, ct_i, ct_j
                )
            except KeyError:
                continue
            A_scores = transfer_cell_scores[ct_i]
            B_scores = transfer_cell_scores[ct_j]
            _validate_kernel_score_shapes(K, A_scores, B_scores, ct_i, ct_j)
            corrs = _transfer_bidir_values(
                K, A_scores, B_scores, normalize_K, filter_kernel,
                K_row_sum_cutoff, K_col_sum_cutoff, ct_i, ct_j,
            )
            for cc, value in enumerate(corrs, start=1):
                rows.append({
                    "sigma": sigma_choice,
                    "cell_type_1": ct_i,
                    "cell_type_2": ct_j,
                    "CC_index": cc,
                    "bidir_correlation": value,
                })
        return pd.DataFrame(rows, columns=[
            "sigma", "cell_type_1", "cell_type_2", "CC_index",
            "bidir_correlation",
        ])

    slides = _target_slides(tar_obj)
    if mode == "per_slide":
        rows = []
        for slide in slides:
            for ct_i, ct_j in pairs:
                inputs = _transfer_slide_inputs(
                    tar_obj, transfer_cell_scores, target_sigma,
                    slide, ct_i, ct_j,
                )
                if inputs is None:
                    continue
                K, A_scores, B_scores = inputs
                corrs = _transfer_bidir_values(
                    K, A_scores, B_scores, normalize_K, filter_kernel,
                    K_row_sum_cutoff, K_col_sum_cutoff, ct_i, ct_j,
                )
                for cc, value in enumerate(corrs, start=1):
                    rows.append({
                        "sigma": sigma_choice,
                        "slideID": slide,
                        "cell_type_1": ct_i,
                        "cell_type_2": ct_j,
                        "CC_index": cc,
                        "bidir_correlation": value,
                    })
        return pd.DataFrame(rows, columns=[
            "sigma", "slideID", "cell_type_1", "cell_type_2", "CC_index",
            "bidir_correlation",
        ])

    rows = []
    for ct_i, ct_j in pairs:
        correlation_sum = np.zeros(n_cc, dtype=float)
        valid_slides = np.zeros(n_cc, dtype=int)
        for slide in slides:
            inputs = _transfer_slide_inputs(
                tar_obj, transfer_cell_scores, target_sigma,
                slide, ct_i, ct_j,
            )
            if inputs is None:
                continue
            K, A_scores, B_scores = inputs
            corrs = _transfer_bidir_values(
                K, A_scores, B_scores, normalize_K, filter_kernel,
                K_row_sum_cutoff, K_col_sum_cutoff, ct_i, ct_j,
            )
            finite = np.isfinite(corrs)
            correlation_sum[finite] += corrs[finite]
            valid_slides[finite] += 1
        for cc in range(n_cc):
            if valid_slides[cc]:
                rows.append({
                    "sigma": sigma_choice,
                    "cell_type_1": ct_i,
                    "cell_type_2": ct_j,
                    "CC_index": cc + 1,
                    "aggregate_correlation": (
                        correlation_sum[cc] / valid_slides[cc]
                    ),
                })
    return pd.DataFrame(rows, columns=[
        "sigma", "cell_type_1", "cell_type_2", "CC_index",
        "aggregate_correlation",
    ])


def _validate_transfer_correlation_inputs(
    tar_obj, transfer_cell_scores, sigma_choice,
):
    """Validate shared transfer-correlation inputs and infer pairs/CC count."""
    if not isinstance(tar_obj, (CoProSingle, CoProMulti)):
        raise TypeError("tar_obj must be a CoProSingle or CoProMulti.")
    if not isinstance(transfer_cell_scores, dict) or not transfer_cell_scores:
        raise ValueError("transfer_cell_scores must be a non-empty dict.")
    if not _is_positive_numeric_scalar(sigma_choice):
        raise ValueError("sigma_choice must be a positive numeric scalar.")

    cts = list(transfer_cell_scores)
    n_cc = None
    for ct in cts:
        matrix = transfer_cell_scores[ct]
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            raise ValueError(f"transfer_cell_scores[{ct!r}] must be a 2-D ndarray.")
        if n_cc is None:
            n_cc = matrix.shape[1]
        elif matrix.shape[1] != n_cc:
            raise ValueError("All transferred score matrices need the same CC count.")
        expected_rows = int(np.sum(np.asarray(tar_obj.cell_types_sub) == ct))
        if matrix.shape[0] != expected_rows:
            raise ValueError(
                f"Transferred scores for {ct!r} have {matrix.shape[0]} rows; "
                f"expected {expected_rows}."
            )
    if n_cc is None or n_cc < 1:
        raise ValueError("Transferred score matrices must have at least one CC.")
    pairs = [(cts[0], cts[0])] if len(cts) == 1 else list(combinations(cts, 2))
    return cts, pairs, n_cc


def _resolve_target_sigma(sigma_choice, sigma_choice_tar):
    if sigma_choice_tar is None:
        return sigma_choice
    if not _is_positive_numeric_scalar(sigma_choice_tar):
        raise ValueError("sigma_choice_tar must be a positive numeric scalar.")
    warnings.warn(
        "Using different sigma values for reference and target objects is not "
        "recommended and is intended for development use only."
    )
    return sigma_choice_tar


def _is_positive_numeric_scalar(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return False
    numeric = float(value)
    return np.isfinite(numeric) and numeric > 0


def _resolve_transfer_calculation_mode(tar_obj, calculation_mode):
    if isinstance(tar_obj, CoProSingle):
        return "single"
    if calculation_mode is None:
        return "per_slide"
    normalized = str(calculation_mode).replace("-", "_").lower()
    if normalized in {"perslide", "per_slide"}:
        return "per_slide"
    if normalized == "aggregate":
        return "aggregate"
    raise ValueError(
        "calculation_mode must be 'per_slide' (or 'perSlide') or 'aggregate'."
    )


def _target_slides(tar_obj):
    slides = list(getattr(tar_obj, "slide_list", []))
    if not slides:
        slides = sorted(tar_obj.meta_data_sub["slideID"].unique().tolist())
    return slides


def _score_indices_for_slide(tar_obj, cell_type, slide):
    cell_types = np.asarray(tar_obj.cell_types_sub)
    slide_ids = tar_obj.meta_data_sub["slideID"].to_numpy()
    type_mask = cell_types == cell_type
    return np.flatnonzero(slide_ids[type_mask] == slide)


def _validate_kernel_score_shapes(K, A_scores, B_scores, ct_i, ct_j):
    if K.shape != (A_scores.shape[0], B_scores.shape[0]):
        raise ValueError(
            f"Kernel shape {K.shape} does not match transferred scores for "
            f"{ct_i}-{ct_j}: {(A_scores.shape[0], B_scores.shape[0])}."
        )


def _transfer_slide_inputs(
    tar_obj, scores, target_sigma, slide, ct_i, ct_j,
):
    try:
        K = _get_kernel_for_pair(
            tar_obj.kernel_matrices, target_sigma, ct_i, ct_j, slide
        )
    except KeyError:
        return None
    idx_i = _score_indices_for_slide(tar_obj, ct_i, slide)
    idx_j = _score_indices_for_slide(tar_obj, ct_j, slide)
    if len(idx_i) == 0 or len(idx_j) == 0:
        return None
    A_scores = scores[ct_i][idx_i]
    B_scores = scores[ct_j][idx_j]
    _validate_kernel_score_shapes(K, A_scores, B_scores, ct_i, ct_j)
    return K, A_scores, B_scores


def _transfer_bidir_values(
    K, A_scores, B_scores, normalize_K, filter_kernel,
    row_cutoff, col_cutoff, ct_i, ct_j,
):
    if filter_kernel:
        filtered = _filter_cross_kernel(
            K, A_scores, B_scores, row_cutoff, col_cutoff
        )
    else:
        filtered = (K, A_scores, B_scores)
    if filtered is None:
        warnings.warn(
            f"Kernel filtering removed all cells for {ct_i}-{ct_j}; "
            "returning zero bidirectional correlation."
        )
        return np.zeros(A_scores.shape[1], dtype=float)
    K_use, A_use, B_use = filtered
    return _compute_bidir_corrs_all_cc(A_use, B_use, K_use, normalize_K)


def _get_gene_names(obj) -> list | None:
    """Try to get gene names from the object's metadata or PCA."""
    if hasattr(obj, "gene_names") and obj.gene_names is not None:
        return list(obj.gene_names)
    # If normalized_data was from a DataFrame, column names may be available
    if hasattr(obj, "gene_list") and obj.gene_list is not None:
        return list(obj.gene_list)
    return None
