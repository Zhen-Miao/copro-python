"""Plotting functions for CoPro objects."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _import_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


# ---------------------------------------------------------------------------
# 1. Normalized correlation vs sigma
# ---------------------------------------------------------------------------

def plot_norm_corr(obj, cc_index: int = 1, figsize=(7, 4), ax=None):
    """Plot normalized correlation vs sigma value.

    Helps choose the optimal sigma — the peak indicates the best spatial scale.

    Parameters
    ----------
    obj : CoProSingle or CoProMulti
    cc_index : int
        Which CC to plot (1-based).
    figsize : tuple
    ax : matplotlib Axes or None

    Returns
    -------
    fig, ax
    """
    plt = _import_plt()

    if not obj.normalized_correlation:
        raise ValueError("Run compute_normalized_correlation() first.")

    rows = []
    for sigma_name, df in obj.normalized_correlation.items():
        sub = df[df["CC_index"] == cc_index]
        for _, row in sub.iterrows():
            rows.append(row.to_dict())
    plot_df = pd.DataFrame(rows)

    if plot_df.empty:
        raise ValueError(f"No data for CC_index={cc_index}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Group by cell type pair
    if "cell_type_1" in plot_df.columns and "cell_type_2" in plot_df.columns:
        plot_df["pair"] = plot_df["cell_type_1"] + " - " + plot_df["cell_type_2"]
    else:
        plot_df["pair"] = "within"

    for pair, grp in plot_df.groupby("pair"):
        grp = grp.sort_values("sigma")
        ax.plot(grp["sigma"], grp["normalized_correlation"], "o-", label=pair)

    # Mark the chosen sigma
    if obj.sigma_value_choice is not None:
        ax.axvline(obj.sigma_value_choice, color="red", linestyle="--", alpha=0.6,
                   label=f"chosen σ={obj.sigma_value_choice}")

    ax.set_xlabel("Sigma (σ)")
    ax.set_ylabel("Normalized Correlation")
    ax.set_title(f"Normalized Correlation vs Sigma (CC{cc_index})")
    ax.legend()
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 2. Cell scores in situ (spatial scatter)
# ---------------------------------------------------------------------------

def plot_cell_scores_in_situ(
    obj,
    sigma: float | None = None,
    cc_index: int = 1,
    cell_type: str | None = None,
    cmap: str = "RdBu_r",
    point_size: float = 1,
    figsize=(7, 6),
    ax=None,
):
    """Scatter plot of cell scores on spatial coordinates.

    Parameters
    ----------
    obj : CoProSingle or CoProMulti
    sigma : float or None
        Defaults to sigma_value_choice.
    cc_index : int
        1-based CC index.
    cell_type : str or None
        If None, plots all cell types in subplots.
    cmap : str
        Matplotlib colormap.
    point_size : float
    figsize : tuple
    ax : matplotlib Axes or None (ignored when cell_type is None and multiple types exist)

    Returns
    -------
    fig, axes
    """
    plt = _import_plt()

    if sigma is None:
        sigma = obj.sigma_value_choice
    if sigma is None:
        raise ValueError("No sigma specified and sigma_value_choice not set.")
    if not obj.cell_scores:
        raise ValueError("Run compute_gene_and_cell_scores() first.")

    cts = [cell_type] if cell_type else obj.cell_types_of_interest
    cc = cc_index - 1  # 0-based

    from .core import CoProMulti
    is_multi = isinstance(obj, CoProMulti)

    if len(cts) == 1 or ax is not None:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        for ct in cts:
            mask = obj.cell_types_sub == ct
            loc = obj.location_data_sub.loc[mask]

            if is_multi:
                scores = _gather_multi_scores(obj, sigma, ct, cc)
            else:
                cs_key = f"cellScores|sigma{sigma}|{ct}"
                if cs_key not in obj.cell_scores:
                    continue
                scores = obj.cell_scores[cs_key][:, cc]

            sc = ax.scatter(loc["x"].values, loc["y"].values,
                            c=scores, s=point_size, cmap=cmap, alpha=0.8)
            plt.colorbar(sc, ax=ax, label=f"Cell Score (CC{cc_index})")

        title = f"{cts[0]} — CC{cc_index}" if len(cts) == 1 else f"All types — CC{cc_index}"
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        return fig, ax

    # Multiple cell types → subplots
    n = len(cts)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols / 2, figsize[1] * nrows / 2))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).ravel()

    for i, ct in enumerate(cts):
        axc = axes[i]
        mask = obj.cell_types_sub == ct
        loc = obj.location_data_sub.loc[mask]

        if is_multi:
            scores = _gather_multi_scores(obj, sigma, ct, cc)
        else:
            cs_key = f"cellScores|sigma{sigma}|{ct}"
            if cs_key not in obj.cell_scores:
                continue
            scores = obj.cell_scores[cs_key][:, cc]

        sc = axc.scatter(loc["x"].values, loc["y"].values,
                         c=scores, s=point_size, cmap=cmap, alpha=0.8)
        plt.colorbar(sc, ax=axc, label="Score")
        axc.set_title(f"{ct}")
        axc.set_xlabel("x")
        axc.set_ylabel("y")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Cell Scores in Situ (CC{cc_index}, σ={sigma})", fontsize=13)
    fig.tight_layout()
    return fig, axes


def _gather_multi_scores(obj, sigma, ct, cc):
    """Concatenate per-slide cell scores for a cell type."""
    parts = []
    for slide in obj.slide_list:
        cs_key = f"cellScores|sigma{sigma}|{slide}|{ct}"
        if cs_key in obj.cell_scores:
            parts.append(obj.cell_scores[cs_key][:, cc])
    return np.concatenate(parts) if parts else np.array([])


# ---------------------------------------------------------------------------
# 3. Correlation scatter: AK vs B
# ---------------------------------------------------------------------------

def plot_correlation_scatter(
    obj,
    cell_type_1: str,
    cell_type_2: str | None = None,
    sigma: float | None = None,
    cc_index: int = 1,
    point_size: float = 3,
    figsize=(6, 5),
    ax=None,
):
    """Scatter plot of kernel-weighted scores (AK) vs scores (B).

    For cross-type: plots ``A_score^T @ K`` vs ``B_score``.
    For within-type (cell_type_2=None or same): plots ``A_score^T @ K`` vs ``A_score``.

    Parameters
    ----------
    obj : CoProSingle
    cell_type_1, cell_type_2 : str
    sigma : float or None
    cc_index : int (1-based)

    Returns
    -------
    fig, ax
    """
    plt = _import_plt()
    from .correlation import _get_kernel_for_pair

    if sigma is None:
        sigma = obj.sigma_value_choice
    if sigma is None:
        raise ValueError("No sigma specified and sigma_value_choice not set.")
    if not obj.cell_scores:
        raise ValueError("Run compute_gene_and_cell_scores() first.")

    cc = cc_index - 1
    within = cell_type_2 is None or cell_type_2 == cell_type_1
    ct_a = cell_type_1
    ct_b = cell_type_1 if within else cell_type_2

    from .core import CoProMulti
    is_multi = isinstance(obj, CoProMulti)

    # Gather scores
    if is_multi:
        score_a = _gather_multi_scores(obj, sigma, ct_a, cc)
        score_b = _gather_multi_scores(obj, sigma, ct_b, cc)
        if len(score_a) == 0 or len(score_b) == 0:
            raise ValueError(f"Cell scores not found for sigma={sigma}.")
        # For multi-slide, aggregate kernel across slides
        # We plot per-slide and overlay
        _plot_multi_corr_scatter = True
    else:
        cs_key_a = f"cellScores|sigma{sigma}|{ct_a}"
        cs_key_b = f"cellScores|sigma{sigma}|{ct_b}"
        if cs_key_a not in obj.cell_scores or cs_key_b not in obj.cell_scores:
            raise ValueError(f"Cell scores not found for sigma={sigma}.")
        score_a = obj.cell_scores[cs_key_a][:, cc]
        score_b = obj.cell_scores[cs_key_b][:, cc]
        _plot_multi_corr_scatter = False

    if _plot_multi_corr_scatter:
        # Multi-slide: compute AK per slide and concatenate
        all_AK = []
        all_B = []
        for slide in obj.slide_list:
            cs_key_a_s = f"cellScores|sigma{sigma}|{slide}|{ct_a}"
            cs_key_b_s = f"cellScores|sigma{sigma}|{slide}|{ct_b}"
            if cs_key_a_s not in obj.cell_scores or cs_key_b_s not in obj.cell_scores:
                continue
            sa = obj.cell_scores[cs_key_a_s][:, cc]
            sb = obj.cell_scores[cs_key_b_s][:, cc]
            try:
                K = _get_kernel_for_pair(obj.kernel_matrices, sigma, ct_a, ct_b, slide)
            except KeyError:
                continue
            all_AK.append(sa @ K)
            all_B.append(sb)
        AK = np.concatenate(all_AK)
        score_b = np.concatenate(all_B)
    else:
        K = _get_kernel_for_pair(obj.kernel_matrices, sigma, ct_a, ct_b)
        AK = score_a @ K

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(AK, score_b, s=point_size, alpha=0.5)
    ax.set_xlabel(f"{ct_a} score × K")
    ax.set_ylabel(f"{ct_b} score")

    # Add correlation
    from .correlation import _safe_pearsonr
    r = _safe_pearsonr(np.asarray(AK, dtype=float), np.asarray(score_b, dtype=float))
    ax.set_title(f"{ct_a}{'↔' + ct_b if not within else ' (within)'} "
                 f"CC{cc_index}, r={r:.3f}")

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Gene score volcano plot
# ---------------------------------------------------------------------------

def plot_gene_scores(
    obj,
    cell_type: str,
    cc_name: str = "CC_1",
    top_n: int = 10,
    pval_threshold: float = 0.05,
    figsize=(8, 5),
    ax=None,
):
    """Volcano plot of gene test results (estimate vs -log10 p-value).

    Parameters
    ----------
    obj : CoProSingle or CoProMulti
        Must have ``gene_score_test`` from ``test_gene_scores()``.
    cell_type : str
    cc_name : str
        e.g. "CC_1".
    top_n : int
        Label this many top genes.
    pval_threshold : float
        Significance line.

    Returns
    -------
    fig, ax
    """
    plt = _import_plt()

    if not hasattr(obj, "gene_score_test") or not obj.gene_score_test:
        raise ValueError("Run test_gene_scores() first.")
    if cell_type not in obj.gene_score_test:
        raise ValueError(f"No results for cell type '{cell_type}'.")
    if cc_name not in obj.gene_score_test[cell_type]:
        raise ValueError(f"No results for {cc_name}.")

    df = obj.gene_score_test[cell_type][cc_name].copy()
    df["neg_log10_p"] = -np.log10(df["pvalue"].clip(lower=1e-300))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sig = df["pvalue"] < pval_threshold
    ax.scatter(df.loc[~sig, "estimate"], df.loc[~sig, "neg_log10_p"],
               s=10, c="grey", alpha=0.5, label="NS")
    ax.scatter(df.loc[sig, "estimate"], df.loc[sig, "neg_log10_p"],
               s=15, c="steelblue", alpha=0.7, label=f"p < {pval_threshold}")

    # Significance line
    ax.axhline(-np.log10(pval_threshold), color="red", linestyle="--", alpha=0.5)

    # Label top genes
    top = df.nlargest(top_n, "neg_log10_p")
    for _, row in top.iterrows():
        ax.annotate(row["gene"], (row["estimate"], row["neg_log10_p"]),
                    fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Estimate (β)")
    ax.set_ylabel("-log₁₀(p-value)")
    ax.set_title(f"Gene Scores — {cell_type} {cc_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 5. Permutation null distribution
# ---------------------------------------------------------------------------

def plot_permutation(
    obj,
    cc_index: int = 1,
    cell_type_1: str | None = None,
    cell_type_2: str | None = None,
    bins: int = 20,
    figsize=(6, 4),
    ax=None,
):
    """Histogram of permutation null distribution with observed value.

    Parameters
    ----------
    obj : CoProSingle
        Must have ``normalized_correlation_permu``.
    cc_index : int (1-based)
    cell_type_1, cell_type_2 : str or None
    bins : int

    Returns
    -------
    fig, ax
    """
    plt = _import_plt()
    from .permutation import calculate_pvalue

    result = calculate_pvalue(obj, cc_index, cell_type_1, cell_type_2)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.hist(result["permu_values"], bins=bins, color="grey", alpha=0.7,
            edgecolor="white", label="Null distribution")
    ax.axvline(result["observed"], color="red", linewidth=2,
               label=f"Observed = {result['observed']:.4f}")

    ax.set_xlabel("Normalized Correlation")
    ax.set_ylabel("Count")
    pair_str = f"{result['cell_type_1']}–{result['cell_type_2']}"
    ax.set_title(f"Permutation Test — {pair_str} CC{cc_index}\n"
                 f"p = {result['p_value']:.4f} (n={result['n_permu']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 6. Bidirectional correlation bar chart
# ---------------------------------------------------------------------------

def plot_bidir_correlation(
    obj,
    sigma: float | None = None,
    include_self: bool = True,
    figsize=(8, 4),
    ax=None,
):
    """Bar chart of bidirectional correlations per CC.

    Parameters
    ----------
    obj : CoProSingle or CoProMulti
    sigma : float or None
    include_self : bool
        Include self-type bidir correlations if available.

    Returns
    -------
    fig, ax
    """
    plt = _import_plt()

    if sigma is None:
        sigma = obj.sigma_value_choice
    if sigma is None:
        raise ValueError("No sigma specified and sigma_value_choice not set.")

    sigma_name = f"sigma_{sigma}"

    rows = []

    # Cross-type
    if obj.bidir_correlation and sigma_name in obj.bidir_correlation:
        df = obj.bidir_correlation[sigma_name]
        for _, row in df.iterrows():
            label = f"{row['cell_type_1']}↔{row['cell_type_2']}"
            rows.append({"pair": label, "CC_index": int(row["CC_index"]),
                         "correlation": row["bidir_correlation"], "type": "cross"})

    # Self-type
    if include_self and hasattr(obj, "self_bidir_correlation") and obj.self_bidir_correlation:
        if sigma_name in obj.self_bidir_correlation:
            df_self = obj.self_bidir_correlation[sigma_name]
            for _, row in df_self.iterrows():
                label = f"{row['cell_type']}↔{row['cell_type']}"
                rows.append({"pair": label, "CC_index": int(row["CC_index"]),
                             "correlation": row["self_bidir_correlation"], "type": "self"})

    if not rows:
        raise ValueError(f"No bidirectional correlation data for sigma={sigma}.")

    plot_df = pd.DataFrame(rows)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ccs = sorted(plot_df["CC_index"].unique())
    pairs = plot_df["pair"].unique()
    n_pairs = len(pairs)
    x = np.arange(n_pairs)
    width = 0.8 / len(ccs)

    colors_cross = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    colors_self = ["#8DA0CB", "#FC8D62", "#66C2A5", "#E78AC3"]

    for i, cc in enumerate(ccs):
        cc_data = plot_df[plot_df["CC_index"] == cc]
        vals = []
        bar_colors = []
        for p in pairs:
            sub = cc_data[cc_data["pair"] == p]
            val = float(sub["correlation"].values[0]) if len(sub) > 0 else 0
            vals.append(val)
            is_self = sub["type"].values[0] == "self" if len(sub) > 0 else False
            bar_colors.append(colors_self[i % len(colors_self)] if is_self
                              else colors_cross[i % len(colors_cross)])
        ax.bar(x + i * width, vals, width, label=f"CC{cc}",
               color=bar_colors, edgecolor="white", alpha=0.85)

    ax.set_xticks(x + width * (len(ccs) - 1) / 2)
    ax.set_xticklabels(pairs, rotation=30, ha="right")
    ax.set_ylabel("Bidirectional Correlation")
    ax.set_title(f"Bidirectional Correlation (σ={sigma})")
    ax.legend()
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 7. Top genes bar chart (by regression gene weight)
# ---------------------------------------------------------------------------

def plot_top_genes(
    obj,
    cell_type: str,
    cc_index: int = 1,
    gene_names=None,
    top_n: int = 20,
    colors_pos: str = "#e41a1c",
    colors_neg: str = "#377eb8",
    title: str | None = None,
    figsize=(5, 6),
    ax=None,
):
    """Horizontal bar chart of top genes ranked by absolute regression weight.

    Parameters
    ----------
    obj : CoProSingle or CoProMulti
        Must have ``gene_scores_regression`` from
        ``compute_regression_gene_scores()``.
    cell_type : str
    cc_index : int
        1-based CC index.
    gene_names : array-like or None
        Gene names matching the gene dimension.  If *None*, integer indices
        are used.
    top_n : int
    colors_pos, colors_neg : str
        Bar colours for positive / negative weights.
    title : str or None
    figsize : tuple
    ax : matplotlib Axes or None

    Returns
    -------
    fig, ax
    """
    plt = _import_plt()

    if not hasattr(obj, "gene_scores_regression") or not obj.gene_scores_regression:
        raise ValueError("Run compute_regression_gene_scores() first.")

    sigma = obj.sigma_value_choice
    gs_key = f"geneScores|sigma{sigma}|{cell_type}"
    if gs_key not in obj.gene_scores_regression:
        raise ValueError(
            f"No regression gene scores for '{cell_type}' at sigma={sigma}."
        )

    gs = obj.gene_scores_regression[gs_key][:, cc_index - 1]

    if gene_names is None:
        gene_names = np.arange(len(gs))
    gene_names = np.asarray(gene_names)

    gene_df = pd.DataFrame({"gene": gene_names, "weight": gs})
    gene_df = gene_df.reindex(
        gene_df["weight"].abs().sort_values(ascending=False).index
    )
    top = gene_df.head(top_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colors = [colors_pos if w > 0 else colors_neg for w in top["weight"]]
    ax.barh(
        top["gene"].values[::-1],
        top["weight"].values[::-1],
        color=colors[::-1],
    )
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Gene weight")
    ax.set_title(title or f"Top {top_n} {cell_type} genes (CC{cc_index})")

    fig.tight_layout()
    return fig, ax
