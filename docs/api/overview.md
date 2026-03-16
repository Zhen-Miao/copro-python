# API Overview

CoPro exposes a **stateful pipeline**: each function takes a `CoProSingle` (or `CoProMulti`) object, augments it with new results, and returns it. The typical call order is:

```python
import copro as cp

obj = cp.CoProSingle(...)           # 1. Create object
obj = cp.subset_data(obj, [...])    # 2. Filter cell types
obj = cp.compute_pca(obj)           # 3. PCA
obj = cp.compute_distance(obj)      # 4. Distances
obj = cp.compute_kernel_matrix(obj) # 5. Kernel matrices
obj = cp.run_skr_cca(obj)           # 6. SkrCCA optimization
obj = cp.compute_normalized_correlation(obj)  # 7. Normalized correlation
obj = cp.compute_gene_and_cell_scores(obj)    # 8. Scores
```

## Key output slots

| Attribute | Content |
|-----------|---------|
| `obj.sigma_value_choice` | Automatically selected σ (maximizes mean CC1 correlation) |
| `obj.normalized_correlation` | `dict[str, DataFrame]` — one DataFrame per σ with columns `sigma`, `cell_type_1`, `cell_type_2`, `CC_index`, `normalized_correlation` |
| `obj.cell_scores` | `dict[str, ndarray]` — key format: `"cellScores\|sigma{s}\|{ct}"` → shape `(n_cells, n_cc)` |
| `obj.gene_scores` | `dict[str, ndarray]` — key format: `"geneScores\|sigma{s}\|{ct}"` → shape `(n_genes, n_cc)` |
| `obj.kernel_matrices` | `dict[str, ndarray]` — key format: `"kernel\|sigma{s}\|{ct_i}\|{ct_j}"` |
| `obj.pca_global` | `dict[str, dict]` — per cell type: `scores`, `sdev`, `rotation`, `components` |
