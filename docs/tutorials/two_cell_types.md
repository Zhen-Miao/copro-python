# Tutorial: Two Cell Types (Brain MERFISH)

This tutorial demonstrates how to run CoPro on a **two cell type** dataset.
We use a brain MERFISH dataset from Zhang et al. *Nature* 2023, focusing on the
co-progression between **D1 neurons** and **D2 neurons** in the striatum.

**What CoPro finds here:** A coordinated spatial gradient shared between D1 and D2 neurons,
reflecting their positional identity along the striatum axis.

The full executable notebook is available at
[`tutorials/tutorial_two_cell_types.ipynb`](https://github.com/Zhen-Miao/copro-python/blob/main/tutorials/tutorial_two_cell_types.ipynb).

---

## 1. Load packages

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copro as cp
```

## 2. Set parameters

```python
N_PCA        = 40
CELL_TYPE_A  = "061 STR D1 Gaba"
CELL_TYPE_B  = "062 STR D2 Gaba"
CELL_TYPES   = [CELL_TYPE_A, CELL_TYPE_B]
SIGMA_VALUES = [0.1, 0.14, 0.2, 0.5]
N_CC         = 2
```

## 3. Load data

```python
EXPR_FILE        = "path/to/expression.csv"
META_FILE        = "path/to/metadata.csv"
CELL_TYPE_COLUMN = "subclass"

expr_df = pd.read_csv(EXPR_FILE, index_col=0)
meta    = pd.read_csv(META_FILE, index_col=0)

# If metadata uses a different row identifier, align it to the expression matrix.
# Uncomment and adapt if needed:
# meta.index = meta["cell_label"].astype(str)

# Subset to cell types of interest (positional filter — index-safe)
keep_mask = meta[CELL_TYPE_COLUMN].isin(CELL_TYPES).values
expr_df   = expr_df.iloc[keep_mask]
meta      = meta.iloc[keep_mask]

meta["x"] = pd.to_numeric(meta["x"])
meta["y"] = pd.to_numeric(meta["y"])
```

## 4. Preprocess

```python
# Use data as-is if already normalized; otherwise apply log1p:
# expr_norm = np.log1p(expr_df.values.astype(float))
expr_norm  = expr_df.values.astype(float)
location   = meta[["x", "y"]].reset_index(drop=True)
cell_types = meta[CELL_TYPE_COLUMN].values
```

## 5. Run CoPro pipeline

```python
obj = cp.CoProSingle(
    normalized_data=expr_norm,
    location_data=location,
    meta_data=meta.reset_index(drop=True),
    cell_types=cell_types,
)

obj = cp.subset_data(obj, CELL_TYPES)
obj = cp.compute_pca(obj, n_pca=N_PCA)
obj = cp.compute_distance(obj, normalize=False)
obj = cp.compute_kernel_matrix(obj, sigma_values=SIGMA_VALUES,
                               upper_quantile=0.85, lower_limit=5e-7)
obj = cp.run_skr_cca(obj, scale_pcs=True, n_cc=N_CC)
obj = cp.compute_normalized_correlation(obj)
obj = cp.compute_gene_and_cell_scores(obj)

print(f"Selected sigma: {obj.sigma_value_choice}")
```

## 6. Visualize results

### Cross-type spatial correlation scatter

```python
sigma     = obj.sigma_value_choice
scores_A  = obj.cell_scores[f"cellScores|sigma{sigma}|{CELL_TYPE_A}"][:, 0]
scores_B  = obj.cell_scores[f"cellScores|sigma{sigma}|{CELL_TYPE_B}"][:, 0]
K_AB      = obj.kernel_matrices[f"kernel|sigma{sigma}|{CELL_TYPE_A}|{CELL_TYPE_B}"]
KA_scores = K_AB.T @ scores_A   # kernel-smoothed A scores projected into B space

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(KA_scores, scores_B, s=2, alpha=0.4, color="steelblue")
ax.set_xlabel(f"{CELL_TYPE_A} × K_AB")
ax.set_ylabel(f"{CELL_TYPE_B} cell scores")
plt.tight_layout(); plt.show()
```

### Cell scores in situ

```python
mask_A = obj.cell_types_sub == CELL_TYPE_A
mask_B = obj.cell_types_sub == CELL_TYPE_B
loc_A  = obj.location_data_sub[mask_A].reset_index(drop=True)
loc_B  = obj.location_data_sub[mask_B].reset_index(drop=True)
vabs   = np.percentile(np.abs(np.concatenate([scores_A, scores_B])), 99)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, scores, loc, ct in [(axes[0], scores_A, loc_A, CELL_TYPE_A),
                             (axes[1], scores_B, loc_B, CELL_TYPE_B)]:
    sc = ax.scatter(loc["x"], loc["y"], c=scores, cmap="RdBu_r",
                    s=4, vmin=-vabs, vmax=vabs)
    plt.colorbar(sc, ax=ax, label="Cell score (CC1)")
    ax.set_title(ct, fontsize=9); ax.set_aspect("equal")
plt.tight_layout(); plt.show()
```

---

## Reference

Zhang, M., Pan, X., Jung, W. et al. Molecularly defined and spatially resolved cell atlas
of the whole mouse brain. *Nature* 624, 343–354 (2023).
<https://doi.org/10.1038/s41586-023-06808-9>
