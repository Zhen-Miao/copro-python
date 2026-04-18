"""CoPro Python — Spatial Kernel Restricted CCA for spatial transcriptomics."""

from .core import CoProSingle, CoProMulti, subset_data, create_copro
from .pca import compute_pca
from .distance import compute_distance, compute_self_distance
from .kernel import compute_kernel_matrix, compute_self_kernel
from .skrcca import run_skr_cca, run_skr_cca_supervised
from .correlation import compute_normalized_correlation, compute_bidir_correlation, compute_self_bidir_correlation
from .scores import compute_gene_and_cell_scores, compute_regression_gene_scores
from .transfer import (
    quantile_normalize,
    get_transfer_cell_scores,
    get_transfer_norm_corr,
    get_transfer_bidir_corr,
)
from .permutation import (
    run_skr_cca_permu,
    compute_normalized_correlation_permu,
    calculate_pvalue,
)
from .gene_test import test_gene_scores
from .plotting import (
    plot_norm_corr,
    plot_cell_scores_in_situ,
    plot_correlation_scatter,
    plot_gene_scores,
    plot_permutation,
    plot_bidir_correlation,
    plot_top_genes,
)
from .data import copro_download_data

__all__ = [
    "CoProSingle",
    "CoProMulti",
    "subset_data",
    "create_copro",
    "compute_pca",
    "compute_distance",
    "compute_self_distance",
    "compute_kernel_matrix",
    "compute_self_kernel",
    "run_skr_cca",
    "run_skr_cca_supervised",
    "compute_normalized_correlation",
    "compute_bidir_correlation",
    "compute_self_bidir_correlation",
    "compute_gene_and_cell_scores",
    "compute_regression_gene_scores",
    "quantile_normalize",
    "get_transfer_cell_scores",
    "get_transfer_norm_corr",
    "get_transfer_bidir_corr",
    "run_skr_cca_permu",
    "compute_normalized_correlation_permu",
    "calculate_pvalue",
    "test_gene_scores",
    "plot_norm_corr",
    "plot_cell_scores_in_situ",
    "plot_correlation_scatter",
    "plot_gene_scores",
    "plot_permutation",
    "plot_bidir_correlation",
    "plot_top_genes",
    "copro_download_data",
]

__version__ = "0.1.1"
