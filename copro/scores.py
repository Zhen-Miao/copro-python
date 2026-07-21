"""compute_gene_and_cell_scores() — project CCA weights back to cell and gene space."""

from __future__ import annotations

import numpy as np

from .core import CoProSingle
from .skrcca import _prepare_pc_matrices


def compute_gene_and_cell_scores(obj):
    """Compute cell scores and gene scores for each sigma × cell type × CC.

    Dispatches to multi-slide version for CoProMulti objects.

    Single-slide:
      Cell scores  : X_scaled @ w   → (n_cells_ct, n_cc)
      Gene scores  : (w * (1/sdev))^T @ rotation^T   → (n_genes, n_cc)
      Keys: 'cellScores|sigma{s}|{ct}' and 'geneScores|sigma{s}|{ct}'.

    Multi-slide:
      Cell scores per slide: 'cellScores|sigma{s}|{slide}|{ct}'
      Gene scores shared: 'geneScores|sigma{s}|{ct}'
    """
    from .core import CoProMulti
    if isinstance(obj, CoProMulti):
        return _compute_scores_multi(obj)

    # --- Single-slide path ---
    cts = obj.cell_types_of_interest
    if not cts:
        raise ValueError("No cell types of interest.")
    if not obj.skr_cca_out:
        raise ValueError("CCA results missing. Run run_skr_cca() first.")
    if not any(
        obj.skr_cca_out.get(f"sigma_{sigma}") is not None
        for sigma in obj.sigma_values
    ):
        if any(str(key).startswith("gscca_sigma_") for key in obj.skr_cca_out):
            raise ValueError(
                "Only gene-space CCA results are present. run_gene_space_cca() "
                "already stores gene and cell scores directly."
            )
        raise ValueError("No usable PCA-space skrCCA results are present.")
    if not obj.pca_global:
        raise ValueError("PCA results missing. Run compute_pca() first.")

    scale_pcs = getattr(obj, "scale_pcs", True)
    n_cc = obj.n_cc

    X_dict = _prepare_pc_matrices(obj, scale_pcs, cts)

    cell_scores = dict(obj.cell_scores)
    gene_scores = dict(obj.gene_scores)

    for sigma in obj.sigma_values:
        sigma_name = f"sigma_{sigma}"
        w_sigma = obj.skr_cca_out.get(sigma_name)
        if w_sigma is None:
            continue

        for ct in cts:
            W = w_sigma[ct]          # (n_pca, n_cc)
            X = X_dict[ct]           # (n_cells, n_pca)
            pca = obj.pca_global[ct]
            rotation = pca["rotation"]  # (n_genes, n_pca)
            sdev = pca["sdev"]          # (n_pca,)

            # Cell scores
            cs = X @ W   # (n_cells, n_cc)
            cell_key = f"cellScores|sigma{sigma}|{ct}"
            cell_scores[cell_key] = cs

            # Gene scores — matches R: matrix(w * (1/sdev), nrow=1) %*% t(rotation)
            # gene_score = R %*% diag(1/sdev) %*% w  (regression coefficient)
            # This inverts the sdev scaling applied during PCA whitening.
            if scale_pcs:
                sdev_safe = sdev.copy()
                sdev_safe[sdev_safe < 1e-10] = 1.0
                sdev_use = 1.0 / sdev_safe
            else:
                sdev_use = np.ones_like(sdev)

            gs = rotation @ (W * sdev_use[:, np.newaxis])  # (n_genes, n_cc)
            gene_key = f"geneScores|sigma{sigma}|{ct}"
            gene_scores[gene_key] = gs

    obj.cell_scores = cell_scores
    obj.gene_scores = gene_scores
    return obj


def _compute_scores_multi(obj):
    """Multi-slide scores. Cell scores per slide, gene scores shared."""
    from .skrcca import _prepare_pc_matrices_multi
    cts = obj.cell_types_of_interest
    slides = obj.slide_list
    scale_pcs = getattr(obj, "scale_pcs", True)
    n_cc = obj.n_cc
    if not any(
        obj.skr_cca_out.get(f"sigma_{sigma}") is not None
        for sigma in obj.sigma_values
    ):
        if any(str(key).startswith("gscca_sigma_") for key in obj.skr_cca_out):
            raise ValueError(
                "Only gene-space CCA results are present. run_gene_space_cca() "
                "already stores gene and cell scores directly."
            )
        raise ValueError("No usable PCA-space skrCCA results are present.")

    X_list_all = _prepare_pc_matrices_multi(obj, scale_pcs, cts)

    cell_scores = dict(obj.cell_scores)
    gene_scores = dict(obj.gene_scores)

    for sigma in obj.sigma_values:
        sigma_name = f"sigma_{sigma}"
        w_sigma = obj.skr_cca_out.get(sigma_name)
        if w_sigma is None:
            continue

        for ct in cts:
            W = w_sigma[ct]  # (n_pca, n_cc)
            pca = obj.pca_global[ct]
            rotation = pca["rotation"]  # (n_genes, n_pca)
            sdev = pca["sdev"]
            if scale_pcs:
                sdev_safe = sdev.copy()
                sdev_safe[sdev_safe < 1e-10] = 1.0
                sdev_use = 1.0 / sdev_safe
            else:
                sdev_use = np.ones_like(sdev)

            # Gene scores: shared (no slide in key)
            gs = rotation @ (W * sdev_use[:, np.newaxis])
            gene_scores[f"geneScores|sigma{sigma}|{ct}"] = gs

            # Cell scores: per slide
            for slide in slides:
                X = X_list_all[slide].get(ct)
                if X is None:
                    continue
                cs = X @ W
                cell_scores[f"cellScores|sigma{sigma}|{slide}|{ct}"] = cs

    obj.cell_scores = cell_scores
    obj.gene_scores = gene_scores
    return obj


# ---------------------------------------------------------------------------
# Regression-based gene scores
# ---------------------------------------------------------------------------


def compute_regression_gene_scores(obj, sigma=None, verbose=True):
    """Compute regression-based gene scores for each sigma × cell type × CC.

    Instead of back-projecting CCA weights through PCA loadings, this
    regresses the raw gene expression onto the cell scores:

        beta_g = cov(gene_g, cellScore) / var(cellScore)

    This avoids collinearity issues and typically produces more robust
    weights for score transfer to new datasets.

    Results are stored in ``obj.gene_scores_regression`` with the same key
    format as ``obj.gene_scores``.

    Parameters
    ----------
    obj : CoProSingle or CoProMulti
        Must have cell scores computed (call ``compute_gene_and_cell_scores``
        first).
    sigma : list of float or None
        Sigma values to process. If None, uses all in ``obj.sigma_values``.
    verbose : bool
        Print progress messages.

    Returns
    -------
    obj
        The input object with ``gene_scores_regression`` populated.
    """
    from .core import CoProMulti
    if isinstance(obj, CoProMulti):
        return _compute_regression_gene_scores_multi(obj, sigma, verbose)

    cts = obj.cell_types_of_interest
    if not cts:
        raise ValueError("No cell types of interest.")
    if not obj.cell_scores:
        raise ValueError("Cell scores missing. Run compute_gene_and_cell_scores() first.")

    sigmas = sigma if sigma is not None else obj.sigma_values
    n_cc = obj.n_cc

    gene_scores_reg = {}

    for sig in sigmas:
        for ct in cts:
            cs_key = f"cellScores|sigma{sig}|{ct}"
            if cs_key not in obj.cell_scores:
                continue
            cs = obj.cell_scores[cs_key]  # (n_cells_ct, n_cc)

            # Get expression matrix for this cell type
            mask = obj.cell_types_sub == ct
            X = obj.normalized_data_sub[mask].astype(float)  # (n_cells_ct, n_genes)
            n_genes = X.shape[1]

            gs_reg = np.zeros((n_genes, n_cc))

            for cc in range(n_cc):
                cs_cc = cs[:, cc]
                cs_c = cs_cc - cs_cc.mean()
                denom = np.sum(cs_c ** 2)

                if denom < 1e-12:
                    if verbose:
                        print(f"  Warning: zero-variance cell scores for "
                              f"sigma={sig}, {ct}, CC{cc+1}. Betas set to 0.")
                    continue

                # Center gene expression
                X_c = X - X.mean(axis=0)
                # beta_g = X_c^T @ cs_c / denom
                gs_reg[:, cc] = (X_c.T @ cs_c) / denom

            gs_key = f"geneScores|sigma{sig}|{ct}"
            gene_scores_reg[gs_key] = gs_reg

            if verbose:
                print(f"Regression gene scores computed for sigma={sig}, {ct}")

    obj.gene_scores_regression = gene_scores_reg
    return obj


def _compute_regression_gene_scores_multi(obj, sigma, verbose):
    """Multi-slide regression gene scores. Uses per-slide cell scores."""
    cts = obj.cell_types_of_interest
    slides = obj.slide_list
    sigmas = sigma if sigma is not None else obj.sigma_values
    n_cc = obj.n_cc

    gene_scores_reg = {}

    for sig in sigmas:
        for ct in cts:
            # Gather cell scores and expression across all slides
            all_cs = []
            all_X = []
            for slide in slides:
                cs_key = f"cellScores|sigma{sig}|{slide}|{ct}"
                if cs_key not in obj.cell_scores:
                    continue
                cs_slide = obj.cell_scores[cs_key]

                # Get expression for this slide + cell type
                slide_ids = obj.meta_data_sub["slideID"].values
                mask = (obj.cell_types_sub == ct) & (slide_ids == slide)
                X_slide = obj.normalized_data_sub[mask].astype(float)

                all_cs.append(cs_slide)
                all_X.append(X_slide)

            if not all_cs:
                continue

            cs = np.vstack(all_cs)   # (n_cells_all, n_cc)
            X = np.vstack(all_X)     # (n_cells_all, n_genes)
            n_genes = X.shape[1]

            gs_reg = np.zeros((n_genes, n_cc))
            for cc in range(n_cc):
                cs_cc = cs[:, cc]
                cs_c = cs_cc - cs_cc.mean()
                denom = np.sum(cs_c ** 2)

                if denom < 1e-12:
                    if verbose:
                        print(f"  Warning: zero-variance cell scores for "
                              f"sigma={sig}, {ct}, CC{cc+1}. Betas set to 0.")
                    continue

                X_c = X - X.mean(axis=0)
                gs_reg[:, cc] = (X_c.T @ cs_c) / denom

            gs_key = f"geneScores|sigma{sig}|{ct}"
            gene_scores_reg[gs_key] = gs_reg

            if verbose:
                print(f"Regression gene scores computed for sigma={sig}, {ct}")

    obj.gene_scores_regression = gene_scores_reg
    return obj
