"""test_gene_scores() — identify genes associated with CoPro axes via GLM or LMM."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signed Z-score helpers  (mirrors R get.cor.zscores)
# ---------------------------------------------------------------------------

def _one_sided_pvalue(estimate: np.ndarray, pvalue: np.ndarray) -> np.ndarray:
    """Convert two-sided p-value to one-sided (upper tail) given sign of estimate."""
    p = pvalue.copy()
    p[p == 0] = np.nanmin(p[p > 0]) if np.any(p > 0) else 1e-300
    p_one = np.full_like(p, np.nan)
    pos = (estimate > 0) & ~np.isnan(estimate)
    neg = (estimate <= 0) & ~np.isnan(estimate)
    p_one[pos] = p[pos] / 2.0
    p_one[neg] = 1.0 - p[neg] / 2.0
    return p_one


def _signed_zscore(estimate: np.ndarray, pvalue: np.ndarray) -> np.ndarray:
    """Compute signed Z-score: positive when estimate > 0 is significant."""
    p_up = _one_sided_pvalue(estimate, pvalue)
    p_down = _one_sided_pvalue(-estimate, pvalue)
    v = np.column_stack([p_up, p_down])

    # Positive z when p_up < 0.5  (gene positively associated)
    # Negative z when p_down < 0.5  (gene negatively associated)
    z = -np.log10(v[:, 0])
    flip = v[:, 0] > 0.5
    z[flip] = np.log10(v[flip, 1])
    return z


# ---------------------------------------------------------------------------
# GLM (OLS) path
# ---------------------------------------------------------------------------

def _test_gene_glm(
    expression: np.ndarray,
    cell_score: np.ndarray,
    covariate_df: pd.DataFrame | None,
    gene_names: list | None,
) -> pd.DataFrame:
    """OLS regression: cell_score ~ gene_g + covariates, for each gene g.

    Returns DataFrame with columns: gene, estimate, pvalue, z_score.
    """
    from statsmodels.api import OLS, add_constant

    n_cells, n_genes = expression.shape

    # Build covariate matrix (without gene column — added per iteration)
    if covariate_df is not None and covariate_df.shape[1] > 0:
        cov_mat = covariate_df.values.astype(float)
    else:
        cov_mat = None

    estimates = np.full(n_genes, np.nan)
    pvalues = np.full(n_genes, np.nan)

    y = cell_score.astype(float)

    for g in range(n_genes):
        x_g = expression[:, g].astype(float)
        if cov_mat is not None:
            X = np.column_stack([x_g, cov_mat])
        else:
            X = x_g.reshape(-1, 1)
        X = add_constant(X)

        try:
            result = OLS(y, X).fit()
            # Coefficient for x_g is at index 1 (0 = const)
            estimates[g] = result.params[1]
            pvalues[g] = result.pvalues[1]
        except Exception:
            pass  # leave as NaN

    z = _signed_zscore(estimates, pvalues)

    gnames = gene_names if gene_names is not None else [f"gene_{i}" for i in range(n_genes)]
    return pd.DataFrame({
        "gene": gnames,
        "estimate": estimates,
        "pvalue": pvalues,
        "z_score": z,
    })


# ---------------------------------------------------------------------------
# LMM path
# ---------------------------------------------------------------------------

def _test_gene_lmm(
    expression: np.ndarray,
    cell_score: np.ndarray,
    covariate_df: pd.DataFrame | None,
    group_var: np.ndarray,
    gene_names: list | None,
) -> pd.DataFrame:
    """LMM: cell_score ~ gene_g + covariates + (1 | group), for each gene g.

    Uses statsmodels MixedLM.

    Returns DataFrame with columns: gene, estimate, pvalue, z_score.
    """
    import statsmodels.formula.api as smf

    n_cells, n_genes = expression.shape

    estimates = np.full(n_genes, np.nan)
    pvalues = np.full(n_genes, np.nan)

    # Build base dataframe once
    base_df = pd.DataFrame({"y": cell_score.astype(float), "group": group_var})
    if covariate_df is not None and covariate_df.shape[1] > 0:
        for col in covariate_df.columns:
            base_df[col] = covariate_df[col].values

    # Build formula string
    fixed_parts = ["x"]
    if covariate_df is not None and covariate_df.shape[1] > 0:
        for col in covariate_df.columns:
            fixed_parts.append(col)
    formula_str = "y ~ " + " + ".join(fixed_parts)

    for g in range(n_genes):
        df = base_df.copy()
        df["x"] = expression[:, g].astype(float)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(formula_str, df, groups=df["group"])
                result = model.fit(reml=True, method="lbfgs", maxiter=200)
            estimates[g] = result.fe_params["x"]
            pvalues[g] = result.pvalues["x"]
        except Exception:
            pass  # leave as NaN

    z = _signed_zscore(estimates, pvalues)

    gnames = gene_names if gene_names is not None else [f"gene_{i}" for i in range(n_genes)]
    return pd.DataFrame({
        "gene": gnames,
        "estimate": estimates,
        "pvalue": pvalues,
        "z_score": z,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def test_gene_scores(
    obj,
    sigma: float | None = None,
    test_type: str = "lmm",
    covariates: list | None = None,
    group_var: str = "slideID",
    gene_names: list | None = None,
    verbose: bool = True,
):
    """Test gene–axis associations via GLM or LMM for each cell type × CC.

    For each gene *g*, fits:

    - **GLM** (OLS): ``cell_score ~ gene_g [+ covariates]``
    - **LMM**: ``cell_score ~ gene_g [+ covariates] + (1 | group_var)``

    and extracts the coefficient estimate and p-value for gene_g, then
    converts to a signed Z-score (positive = gene positively associated
    with the axis).

    Parameters
    ----------
    obj : CoProSingle or CoProMulti
        Must have cell scores computed.
    sigma : float or None
        Sigma to use. Defaults to ``obj.sigma_value_choice``.
    test_type : str
        ``"lmm"`` (default) or ``"glm"``.
    covariates : list of str or None
        Column names from ``meta_data`` (or ``meta_data_sub``) to include
        as fixed-effect covariates.
    group_var : str
        Column name for the random-intercept grouping variable (LMM only).
        Default ``"slideID"``.
    gene_names : list of str or None
        Gene names matching columns of ``normalized_data``. If None, uses
        ``gene_0, gene_1, ...``.
    verbose : bool
        Print progress.

    Returns
    -------
    obj
        With ``gene_score_test`` dict populated:
        ``{ct: {f"CC_{cc}": DataFrame}}`` where DataFrame has columns
        ``gene, estimate, pvalue, z_score``.
    """
    from .core import CoProMulti

    test_type = test_type.lower()
    if test_type not in ("lmm", "glm"):
        raise ValueError("test_type must be 'lmm' or 'glm'.")

    if sigma is None:
        sigma = obj.sigma_value_choice
    if sigma is None:
        raise ValueError("No sigma specified and sigma_value_choice not set.")

    if not obj.cell_scores:
        raise ValueError("Cell scores missing. Run compute_gene_and_cell_scores() first.")

    cts = obj.cell_types_of_interest
    n_cc = obj.n_cc
    is_multi = isinstance(obj, CoProMulti)

    # Get metadata for covariates
    if is_multi:
        meta_sub = obj.meta_data_sub
    else:
        meta_sub = obj.meta_data.loc[
            np.isin(obj.cell_types, cts)
        ].reset_index(drop=True) if hasattr(obj, "meta_data") else None

    results = {}

    for ct in cts:
        results[ct] = {}

        # Validate covariate / group columns up-front (order-independent).
        if covariates:
            if meta_sub is None:
                raise ValueError("No metadata available for covariates.")
            missing = [c for c in covariates if c not in meta_sub.columns]
            if missing:
                raise ValueError(f"Covariates not found in metadata: {missing}")
        if test_type == "lmm":
            if meta_sub is None or group_var not in meta_sub.columns:
                raise ValueError(
                    f"LMM requires '{group_var}' in metadata. "
                    f"Use test_type='glm' or provide a valid group_var."
                )

        # Build the design side (X, covariates, group).  For multi-slide objects
        # this MUST be assembled per-slide in obj.slide_list order — the same order
        # the response `cell_score` is concatenated below — so that every row of X
        # aligns with its cell score.  Building X in whole-dataset cell order would
        # misalign the response whenever a cell type's cells are not contiguous by
        # slide (interleaved input, unsorted obs, or auto-sliced blocks).
        if is_multi:
            slide_ids = obj.meta_data_sub["slideID"].values
            included_slides = []
            X_parts = []
            cov_parts = []
            group_parts = []
            for slide in obj.slide_list:
                cs_key = f"cellScores|sigma{sigma}|{slide}|{ct}"
                if cs_key not in obj.cell_scores:
                    # Skip this slide on BOTH sides (design and cell_score) so the
                    # two stay aligned.
                    continue
                mask_s = (obj.cell_types_sub == ct) & (slide_ids == slide)
                included_slides.append(slide)
                X_parts.append(obj.normalized_data_sub[mask_s].astype(float))
                if covariates:
                    cov_parts.append(meta_sub.loc[mask_s, covariates])
                if test_type == "lmm":
                    group_parts.append(meta_sub.loc[mask_s, group_var].values)

            X = np.vstack(X_parts) if X_parts else None
            cov_df = pd.concat(cov_parts, ignore_index=True) if cov_parts else None
            group = np.concatenate(group_parts) if group_parts else None
        else:
            mask = obj.cell_types_sub == ct
            X = obj.normalized_data_sub[mask].astype(float)
            cov_df = None
            if covariates:
                cov_df = meta_sub.loc[mask, covariates].reset_index(drop=True)
            group = None
            if test_type == "lmm":
                group = meta_sub.loc[mask, group_var].values

        for cc in range(n_cc):
            cc_name = f"CC_{cc + 1}"

            # Get cell scores for this ct + cc
            if is_multi:
                # Aggregate across slides in the SAME slide order used to build X.
                cs_parts = [
                    obj.cell_scores[f"cellScores|sigma{sigma}|{slide}|{ct}"][:, cc]
                    for slide in included_slides
                ]
                if not cs_parts:
                    if verbose:
                        print(f"  No cell scores for {ct} CC{cc+1}, skipping.")
                    continue
                cell_score = np.concatenate(cs_parts)
            else:
                cs_key = f"cellScores|sigma{sigma}|{ct}"
                if cs_key not in obj.cell_scores:
                    if verbose:
                        print(f"  No cell scores for {ct} CC{cc+1}, skipping.")
                    continue
                cell_score = obj.cell_scores[cs_key][:, cc]

            if verbose:
                print(f"Testing {ct} {cc_name} ({test_type.upper()}): "
                      f"{X.shape[1]} genes, {len(cell_score)} cells...")

            if test_type == "glm":
                df_result = _test_gene_glm(X, cell_score, cov_df, gene_names)
            else:
                df_result = _test_gene_lmm(X, cell_score, cov_df, group, gene_names)

            results[ct][cc_name] = df_result

            if verbose:
                n_sig = (df_result["pvalue"] < 0.05).sum()
                print(f"  {n_sig} / {len(df_result)} genes significant (p < 0.05)")

    obj.gene_score_test = results
    return obj
