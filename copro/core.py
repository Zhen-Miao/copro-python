"""CoProSingle and CoProMulti dataclasses — state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
    gene_scores_regression: dict = field(default_factory=dict)
    bidir_correlation: dict = field(default_factory=dict)
    n_cc: int = 2
    n_pca: int = 30
    scale_pcs: bool = True

    # Permutation testing
    cell_permu: dict = field(default_factory=dict)
    skr_cca_permu_out: dict = field(default_factory=dict)
    normalized_correlation_permu: dict = field(default_factory=dict)

    # Gene score testing (GLM / LMM)
    gene_score_test: dict = field(default_factory=dict)


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
    gene_scores_regression: dict = field(default_factory=dict)
    bidir_correlation: dict = field(default_factory=dict)
    n_cc: int = 2
    n_pca: int = 30
    scale_pcs: bool = True

    # Permutation testing
    cell_permu: dict = field(default_factory=dict)
    skr_cca_permu_out: dict = field(default_factory=dict)
    normalized_correlation_permu: dict = field(default_factory=dict)

    # Gene score testing (GLM / LMM)
    gene_score_test: dict = field(default_factory=dict)


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


# ---------------------------------------------------------------------------
# Auto-K spatial blocking
# ---------------------------------------------------------------------------

def _partition_by_location(
    location_data: pd.DataFrame,
    n: int,
    max_cell: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Partition locations into approximately equal-sized spatial blocks via k-means.

    Recursively subdivides any cluster that still exceeds *max_cell*.

    Parameters
    ----------
    location_data : pd.DataFrame
        Must contain columns ``x`` and ``y`` (optionally ``z``).
    n : int
        Target number of clusters (>= 1).
    max_cell : int
        Maximum cells per block.
    rng : np.random.RandomState
        Seeded RNG instance (avoids mutating global state).

    Returns
    -------
    np.ndarray of str
        Block labels, length ``len(location_data)``.
    """
    cols_lower = [c.lower() for c in location_data.columns]
    use_cols = ["x", "y"]
    if "z" in cols_lower:
        use_cols.append("z")
    col_idx = [cols_lower.index(c) for c in use_cols]
    loc = location_data.iloc[:, col_idx].to_numpy(dtype=np.float64)

    m = loc.shape[0]
    if m == 0:
        return np.array([], dtype=str)
    if n <= 1 or m <= max_cell:
        return np.full(m, "1", dtype=object)

    loc_scaled = StandardScaler().fit_transform(loc)

    km = KMeans(
        n_clusters=n, max_iter=50, n_init=5, random_state=rng.randint(2**31)
    )
    labels = km.fit_predict(loc_scaled).astype(str)

    # Recursive refinement for clusters still exceeding max_cell
    unique_labels, counts = np.unique(labels, return_counts=True)
    if np.any(counts > max_cell):
        new_labels = np.empty(m, dtype=object)
        next_id = 1
        for cl in unique_labels:
            idx = np.where(labels == cl)[0]
            if len(idx) <= max_cell:
                new_labels[idx] = str(next_id)
                next_id += 1
            else:
                extra = math.ceil(len(idx) / max_cell)
                sub_df = pd.DataFrame(
                    loc[idx], columns=use_cols
                )
                sub_labels = _partition_by_location(sub_df, extra, max_cell, rng)
                for sl in np.unique(sub_labels):
                    sub_idx = idx[sub_labels == sl]
                    new_labels[sub_idx] = str(next_id)
                    next_id += 1
        labels = new_labels

    return labels


def create_copro(
    normalized_data: np.ndarray,
    location_data: pd.DataFrame,
    meta_data: pd.DataFrame,
    cell_types: np.ndarray,
    slide_id: Optional[np.ndarray] = None,
    max_cell: int = 50_000,
    plot: bool = False,
    seed: int = 1,
):
    """Create a CoPro object, automatically choosing Single vs Multi.

    Oversized slides (more than *max_cell* cells) are spatially partitioned
    into sub-blocks via k-means so that every block respects the limit.

    Parameters
    ----------
    normalized_data : np.ndarray
        Cells-by-genes expression matrix.
    location_data : pd.DataFrame
        Must contain columns ``x`` and ``y`` (optionally ``z``).
    meta_data : pd.DataFrame
        Per-cell annotations.
    cell_types : np.ndarray
        Cell-type label for every cell.
    slide_id : np.ndarray or None
        Slide identifier for every cell.  If ``None``, all cells are treated
        as belonging to a single slide.
    max_cell : int
        Maximum number of cells per (sub-)slide.
    plot : bool
        If ``True``, show a scatter plot of the auto-slicing result
        (requires *matplotlib*).
    seed : int
        Random seed for reproducible k-means partitioning.

    Returns
    -------
    CoProSingle or CoProMulti
    """
    n_cells = normalized_data.shape[0]
    if (
        len(cell_types) != n_cells
        or len(meta_data) != n_cells
        or len(location_data) != n_cells
    ):
        raise ValueError("Input data dimensions do not match.")

    if slide_id is not None:
        slide_id = np.asarray(slide_id, dtype=str)
        if len(slide_id) != n_cells:
            raise ValueError("Input data dimensions do not match.")
    else:
        slide_id = np.full(n_cells, "slice1", dtype=object)
        slide_id = slide_id.astype(str)

    unique_slides = np.unique(slide_id)

    # Fast path: single slide, small enough → CoProSingle
    if len(unique_slides) == 1 and n_cells <= max_cell:
        return CoProSingle(
            normalized_data=normalized_data,
            location_data=location_data,
            meta_data=meta_data,
            cell_types=cell_types,
        )

    # Identify oversized slides
    oversized = [
        sid for sid in unique_slides if np.sum(slide_id == sid) > max_cell
    ]
    if oversized:
        print(
            f"Slide(s) exceeding cell limit ({max_cell}): "
            f"{', '.join(oversized)}. Performing auto slicing..."
        )

    rng = np.random.RandomState(seed)
    t0 = time.time()
    new_slide_id = slide_id.astype(object)  # object dtype avoids string truncation

    for sid in unique_slides:
        idx = np.where(slide_id == sid)[0]
        if len(idx) > max_cell:
            n_target = math.ceil(len(idx) / max_cell)
            block = _partition_by_location(
                location_data.iloc[idx].reset_index(drop=True),
                n_target,
                max_cell,
                rng,
            )
            new_slide_id[idx] = np.array(
                [f"{sid}_blk{b}" for b in block], dtype=str
            )

    if oversized:
        elapsed = time.time() - t0
        n_new = len(np.unique(new_slide_id[np.isin(slide_id, oversized)]))
        print(
            f"Auto slicing done in {elapsed:.2f}s; "
            f"{len(oversized)} oversized slide(s) split into {n_new} slices; "
            f"total slides: {len(np.unique(new_slide_id))}"
        )

    if plot:
        _plot_auto_slicing(location_data, slide_id, new_slide_id)

    # Inject slideID into meta_data
    meta_data = meta_data.copy()
    meta_data["slideID"] = new_slide_id

    return CoProMulti(
        normalized_data=normalized_data,
        location_data=location_data,
        meta_data=meta_data,
        cell_types=cell_types,
        slide_list=sorted(np.unique(new_slide_id).tolist()),
    )


def _plot_auto_slicing(location_data, slide_id, new_slide_id):
    """Scatter plot of the auto-slicing result (one panel per original slide)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping auto-slicing plot.")
        return

    for sid in np.unique(slide_id):
        mask = slide_id == sid
        x = location_data["x"].values[mask]
        y = location_data["y"].values[mask]
        labels = new_slide_id[mask]

        fig, ax = plt.subplots(figsize=(7, 6))
        for lbl in np.unique(labels):
            sel = labels == lbl
            ax.scatter(x[sel], y[sel], s=1, alpha=0.7, label=lbl)
        ax.set_title(f"Auto-slicing for slide: {sid}")
        ax.legend(title="Block ID", markerscale=5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tight_layout()
        plt.show()
