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
      Gene scores  : (w * sdev)^T @ rotation^T   → (n_genes, n_cc)
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
    if not obj.pca_global:
        raise ValueError("PCA results missing. Run compute_pca() first.")

    scale_pcs = getattr(obj, "scale_pcs", True)
    n_cc = obj.n_cc

    X_dict = _prepare_pc_matrices(obj, scale_pcs, cts)

    cell_scores = {}
    gene_scores = {}

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

            # Gene scores — matches R: matrix(w * sdev, nrow=1) %*% t(rotation)
            # = (w * sdev)^T @ rotation^T = rotation @ (w * sdev)  [per column]
            if scale_pcs:
                sdev_use = sdev
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

    X_list_all = _prepare_pc_matrices_multi(obj, scale_pcs, cts)

    cell_scores = {}
    gene_scores = {}

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
            sdev_use = sdev if scale_pcs else np.ones_like(sdev)

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
