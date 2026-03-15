"""CoPro Python — Spatial Kernel-based Reduced Rank CCA for spatial transcriptomics."""

from .core import CoProSingle, CoProMulti, subset_data
from .pca import compute_pca
from .distance import compute_distance
from .kernel import compute_kernel_matrix
from .skrcca import run_skr_cca
from .correlation import compute_normalized_correlation
from .scores import compute_gene_and_cell_scores

__all__ = [
    "CoProSingle",
    "CoProMulti",
    "subset_data",
    "compute_pca",
    "compute_distance",
    "compute_kernel_matrix",
    "run_skr_cca",
    "compute_normalized_correlation",
    "compute_gene_and_cell_scores",
]

__version__ = "0.1.0"
