# CoPro (Python)

**Unsupervised detection of coordinated spatial progressions in spatial transcriptomics**

[![Python](https://img.shields.io/badge/python-≥3.10-blue)](https://www.python.org)
[![PyPI](https://img.shields.io/badge/pypi-pycopro-orange)](https://pypi.org/project/pycopro/)
[![GitHub](https://img.shields.io/badge/github-Zhen--Miao%2Fcopro--python-lightgrey)](https://github.com/Zhen-Miao/copro-python)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

CoPro detects **coordinated spatial progressions** between cell types in spatial transcriptomics data. Given the spatial positions and gene expression profiles of cells, CoPro finds a low-dimensional axis along which cells of one type are spatially co-organized with cells of another type — or within a single cell type — revealing continuous tissue structure that discrete clustering misses.

The method is built on **Spatial Kernel Restricted CCA (SkrCCA)**, which maximizes spatially weighted cross-covariance between cell type-specific score matrices. Two-cell-type analyses use an exact multi-axis SVD; larger multi-set analyses use projection-deflated optimization.

> **R package:** The original R implementation is available at [github.com/Zhen-Miao/CoPro](https://github.com/Zhen-Miao/CoPro).

---

## Installation

```bash
pip install pycopro
```

Or install from source:

```bash
git clone https://github.com/Zhen-Miao/copro-python.git
cd copro-python
pip install -e .
```

**Requirements:** Python ≥ 3.10, NumPy, SciPy, pandas, scikit-learn.

---

## Quick start

```python
import numpy as np
import pandas as pd
import copro as cp

# Inputs
# expr_norm : np.ndarray  (cells × genes, normalized)
# location  : pd.DataFrame with columns "x", "y"
# cell_types: np.ndarray  (per-cell type labels)

obj = cp.CoProSingle(
    normalized_data=expr_norm,
    location_data=location,
    meta_data=meta,
    cell_types=cell_types,
)

obj = cp.subset_data(obj, ["Cell type A", "Cell type B"])
obj = cp.compute_pca(obj, n_pca=30)
obj = cp.compute_distance(obj, normalize=False)
obj = cp.compute_kernel_matrix(obj, sigma_values=[0.1, 0.2, 0.5])
obj = cp.run_skr_cca(obj, scale_pcs=True, n_cc=2)
obj = cp.compute_normalized_correlation(obj)
obj = cp.compute_gene_and_cell_scores(obj)

print("Selected sigma:", obj.sigma_value_choice)

# Cell scores for the optimal sigma
sigma = obj.sigma_value_choice
scores_A = obj.cell_scores[f"cellScores|sigma{sigma}|Cell type A"][:, 0]
scores_B = obj.cell_scores[f"cellScores|sigma{sigma}|Cell type B"][:, 0]
```

For AnnData input, use `cp.from_anndata(adata, cell_type_key="cell_type", slide_key="sample")`. Large spatial datasets can skip dense distance construction and call `cp.compute_sparse_kernel(...)` directly.

---

## How it works

CoPro runs a seven-step pipeline:

| Step | Function | Description |
|------|----------|-------------|
| 1 | `subset_data` | Filter to cell types of interest |
| 2 | `compute_pca` | Truncated PCA per cell type (ARPACK, matching R IRLBA); sparse input stays sparse |
| 3 | `compute_distance` | Pairwise Euclidean distances (within and between types) |
| 4 | `compute_kernel_matrix` | Dense, sparse, or automatic Gaussian RBF kernels |
| 5 | `run_skr_cca` | Exact two-type SVD or multi-set optimization over σ values |
| 6 | `compute_normalized_correlation` | Whitened-Frobenius normalized correlation per σ and CC |
| 7 | `compute_gene_and_cell_scores` | Project CCA weights to cell and gene space |

Sigma selection is automatic: the σ that maximizes the mean CC1 normalized correlation is chosen as `obj.sigma_value_choice`.

**Multi-slide support:** Use `CoProMulti` for datasets spanning multiple tissue sections. CCA weights are learned jointly across slides; cell scores are computed per slide.

**Gene-space CCA:** `run_gene_space_cca` learns batch-robust gene weights directly across slides. Streaming mode does not retain dense distances or kernels and, for sparse expression input, standardizes only one slide at a time.

**Permutation inference:** fixed-sigma, fair-sigma, and conditional step-down tests are available through `run_skr_cca_permu`, `run_skr_cca_permu_fair_sigma`, and `run_skr_cca_permu_conditional`.

---

## Tutorials

| Notebook | Dataset | Description |
|----------|---------|-------------|
| [`tutorials/tutorial_one_cell_type.ipynb`](tutorials/tutorial_one_cell_type.ipynb) | Intestinal organoid (seqFISH) | Within-type spatial self-organization of epithelial cells along the crypt–villus axis |
| [`tutorials/tutorial_two_cell_types.ipynb`](tutorials/tutorial_two_cell_types.ipynb) | Mouse brain MERFISH (Zhang et al. *Nature* 2023) | Cross-type co-progression between D1 and D2 striatal neurons |

---

## API reference

### Data containers

**`CoProSingle(normalized_data, location_data, meta_data, cell_types)`**
Single-slide state container. All pipeline functions modify and return this object.

**`CoProMulti(normalized_data, location_data, meta_data, cell_types)`**
Multi-slide variant. Requires a `slideID` column in `meta_data`.

### Pipeline functions

```python
cp.subset_data(obj, cell_types_of_interest)
cp.compute_pca(obj, n_pca=30, center=True, scale=True, center_per_slide=True)
cp.compute_distance(obj, normalize=False)
cp.compute_kernel_matrix(obj, sigma_values, upper_quantile=0.85, lower_limit=5e-7)
cp.run_skr_cca(obj, scale_pcs=True, n_cc=2, max_iter=500, tol=1e-5)
cp.compute_normalized_correlation(obj, tol=1e-4)
cp.compute_gene_and_cell_scores(obj)
```

Large-data and multi-slide additions:

```python
cp.from_anndata(adata, cell_type_key="cell_type", slide_key="sample")
cp.compute_sparse_kernel(obj, sigma_values=[0.05, 0.1, 0.2])
cp.run_gene_space_cca(obj, sigma=0.1, streaming=True)
cp.run_skr_cca_permu_conditional(obj, n_permu=999)
```

### Key output slots

| Attribute | Type | Description |
|-----------|------|-------------|
| `obj.sigma_value_choice` | `float` | Automatically selected σ |
| `obj.normalized_correlation` | `dict[str, DataFrame]` | Normalized correlation per σ and CC |
| `obj.cell_scores` | `dict[str, ndarray]` | Cell scores, keyed `"cellScores\|sigma{s}\|{ct}"` |
| `obj.gene_scores` | `dict[str, ndarray]` | Gene scores, keyed `"geneScores\|sigma{s}\|{ct}"` |
| `obj.kernel_matrices` | `dict[str, ndarray or csr_matrix]` | Kernel matrices, keyed `"kernel\|sigma{s}\|{ct_i}\|{ct_j}"` |

---

## Numerical equivalence with R

The Python implementation is numerically validated against the R package on simulation and real datasets. Key design choices ensuring equivalence:

- PCA uses `scipy.sparse.linalg.svds` (ARPACK), the same Krylov-subspace family as R's `irlba::prcomp_irlba`
- Sign convention matches `prcomp_irlba` via `sklearn.utils.extmath.svd_flip`
- Distance and kernel computations are algebraically identical to R's `fields::rdist` + Gaussian RBF
- Exact sparse fixed-radius kernels agree with the dense path to floating-point precision
- Normalized correlation uses the same centered, matched-self-kernel whitened-Frobenius null scale as R
- Two-type canonical axes come from the same exact PC-space SVD as R
- Cell score Pearson |r| vs R ≥ 0.9999 across all tested datasets and sigma values

---

## Citation

If you use CoPro in your research, please cite:

> Miao Z, Qu Y, Huang S, Laux L, Peters S, Aristel A, Zhang Z, Niedernhofer L, McMahon A, Kim J, Zhang NR (2026). *Dissecting the coordinated progression of cell states in spatial transcriptomics with CoPro.* bioRxiv 2026.04.17.719309. https://doi.org/10.64898/2026.04.17.719309

**MERFISH tutorial data:**
> Zhang, M., Pan, X., Jung, W. et al. Molecularly defined and spatially resolved cell atlas of the whole mouse brain. *Nature* 624, 343–354 (2023). https://doi.org/10.1038/s41586-023-06808-9

---

## License

MIT
