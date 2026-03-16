# CoPro Python

**Unsupervised detection of coordinated spatial progressions in spatial transcriptomics**

CoPro detects **coordinated spatial progressions** between cell types in spatial transcriptomics data. Given the spatial positions and gene expression profiles of cells, CoPro finds a low-dimensional axis along which cells of one type are spatially co-organized with cells of another type — or within a single cell type — revealing continuous tissue structure that discrete clustering misses.

The method is built on **Spatial Kernel Restricted CCA (SkrCCA)**: a power-method optimization that maximizes a spatially-weighted cross-covariance between cell type-specific PC score matrices.

!!! note "R package"
    The original R implementation is available at [github.com/Zhen-Miao/CoPro](https://github.com/Zhen-Miao/CoPro).
    See the [R ↔ Python reference](r_to_python.md) for a full function mapping.

## Quick start

```python
import copro as cp

obj = cp.CoProSingle(
    normalized_data=expr_norm,   # np.ndarray (cells × genes)
    location_data=location,      # pd.DataFrame with columns "x", "y"
    meta_data=meta,
    cell_types=cell_types,       # np.ndarray of per-cell type labels
)

obj = cp.subset_data(obj, ["Cell type A", "Cell type B"])
obj = cp.compute_pca(obj, n_pca=30)
obj = cp.compute_distance(obj, normalize=False)
obj = cp.compute_kernel_matrix(obj, sigma_values=[0.1, 0.2, 0.5])
obj = cp.run_skr_cca(obj, scale_pcs=True, n_cc=2)
obj = cp.compute_normalized_correlation(obj)
obj = cp.compute_gene_and_cell_scores(obj)

sigma = obj.sigma_value_choice
scores_A = obj.cell_scores[f"cellScores|sigma{sigma}|Cell type A"][:, 0]
```

## Tutorials

- [**One cell type (Organoid)**](tutorials/one_cell_type.md) — within-type spatial self-organization of epithelial cells along the crypt–villus axis (seqFISH)
- [**Two cell types (MERFISH)**](tutorials/two_cell_types.md) — cross-type co-progression between D1 and D2 striatal neurons (Zhang et al. *Nature* 2023)

## Citation

If you use CoPro in your research, please cite:

> Miao Z. et al. *CoPro: Unsupervised detection of coordinated spatial progressions in spatial transcriptomics* (in preparation).
