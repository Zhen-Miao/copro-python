"""CoProSingle and CoProMulti dataclasses — state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CoProSingle:
    # Input data
    normalized_data: np.ndarray          # cells × genes
    location_data: pd.DataFrame          # cells × {x, y, ...}
    meta_data: pd.DataFrame
    cell_types: np.ndarray               # per-cell label vector

    # Set by subset_data
    cell_types_of_interest: list = field(default_factory=list)
    normalized_data_sub: Optional[np.ndarray] = None
    location_data_sub: Optional[pd.DataFrame] = None
    cell_types_sub: Optional[np.ndarray] = None

    # Computed results (keyed dicts)
    pca_global: dict = field(default_factory=dict)        # ct → dict with components/scores/sdev
    distances: dict = field(default_factory=dict)         # flat keys: "dist|A|B"
    kernel_matrices: dict = field(default_factory=dict)   # flat keys: "kernel|sigma0.1|A|B"
    sigma_values: list = field(default_factory=list)
    skr_cca_out: dict = field(default_factory=dict)       # "sigma_0.1" → {ct: w_matrix}
    normalized_correlation: dict = field(default_factory=dict)
    sigma_value_choice: Optional[float] = None
    cell_scores: dict = field(default_factory=dict)
    gene_scores: dict = field(default_factory=dict)
    n_cc: int = 2
    n_pca: int = 30
    scale_pcs: bool = True


@dataclass
class CoProMulti:
    """Multi-slide CoPro object. meta_data must have a 'slideID' column."""
    # Input data
    normalized_data: np.ndarray
    location_data: pd.DataFrame
    meta_data: pd.DataFrame
    cell_types: np.ndarray

    # Slide list
    slide_list: list = field(default_factory=list)        # ordered list of slide IDs

    # Set by subset_data
    cell_types_of_interest: list = field(default_factory=list)
    normalized_data_sub: Optional[np.ndarray] = None
    location_data_sub: Optional[pd.DataFrame] = None
    cell_types_sub: Optional[np.ndarray] = None
    meta_data_sub: Optional[pd.DataFrame] = None         # subset of meta_data (with slideID)

    # PCA
    pca_global: dict = field(default_factory=dict)        # ct → global PCA dict (rotation, sdev)
    pca_results: dict = field(default_factory=dict)       # slide → {ct → scores matrix}

    # Computed results
    distances: dict = field(default_factory=dict)         # flat keys: "dist|{slide}|A|B"
    kernel_matrices: dict = field(default_factory=dict)   # flat keys: "kernel|sigma0.1|{slide}|A|B"
    sigma_values: list = field(default_factory=list)
    skr_cca_out: dict = field(default_factory=dict)       # "sigma_0.1" → {ct: w_matrix} (shared)
    normalized_correlation: dict = field(default_factory=dict)
    sigma_value_choice: Optional[float] = None
    cell_scores: dict = field(default_factory=dict)       # "cellScores|sigma0.1|{slide}|{ct}"
    gene_scores: dict = field(default_factory=dict)       # "geneScores|sigma0.1|{ct}" (shared)
    n_cc: int = 2
    n_pca: int = 30
    scale_pcs: bool = True


def subset_data(obj, cell_types_of_interest: list, min_cells: int = 10):
    """Filter data to listed cell types. Works for both CoProSingle and CoProMulti."""
    if isinstance(obj, CoProMulti):
        return _subset_data_multi(obj, cell_types_of_interest, min_cells)
    else:
        return _subset_data_single(obj, cell_types_of_interest, min_cells)


def _subset_data_single(obj: CoProSingle, cell_types_of_interest: list, min_cells: int) -> CoProSingle:
    for ct in cell_types_of_interest:
        n = np.sum(obj.cell_types == ct)
        if n < min_cells:
            raise ValueError(
                f"Cell type '{ct}' has only {n} cells (minimum {min_cells} required)."
            )
    mask = np.isin(obj.cell_types, cell_types_of_interest)
    obj.cell_types_of_interest = list(cell_types_of_interest)
    obj.normalized_data_sub = obj.normalized_data[mask]
    obj.location_data_sub = obj.location_data.loc[mask].reset_index(drop=True)
    obj.cell_types_sub = obj.cell_types[mask]
    return obj


def _subset_data_multi(obj: CoProMulti, cell_types_of_interest: list, min_cells: int) -> CoProMulti:
    """Subset multi-slide object. Checks per-slide cell counts."""
    if "slideID" not in obj.meta_data.columns:
        raise ValueError("meta_data must have a 'slideID' column for CoProMulti.")

    # Discover slide list from meta_data if not set
    if not obj.slide_list:
        obj.slide_list = sorted(obj.meta_data["slideID"].unique().tolist())

    for ct in cell_types_of_interest:
        # Check total across all slides
        n_total = np.sum(obj.cell_types == ct)
        if n_total < min_cells:
            raise ValueError(
                f"Cell type '{ct}' has only {n_total} cells total (minimum {min_cells} required)."
            )

    mask = np.isin(obj.cell_types, cell_types_of_interest)
    obj.cell_types_of_interest = list(cell_types_of_interest)
    obj.normalized_data_sub = obj.normalized_data[mask]
    obj.location_data_sub = obj.location_data.loc[mask].reset_index(drop=True)
    obj.cell_types_sub = obj.cell_types[mask]
    obj.meta_data_sub = obj.meta_data.loc[mask].reset_index(drop=True)
    return obj
