# Tutorial: One Cell Type (Intestinal Organoid)

This tutorial demonstrates how to run CoPro on a **single cell type** dataset.
We use an intestinal organoid dataset profiled by seqFISH, where all cells are Epithelial.
CoPro detects the **within-type spatial self-organization**: cells are arranged in a
coordinated spatial progression even within a single cell type.

**What CoPro finds here:** The spatial co-progression of epithelial cells captures the
crypt–villus axis of the organoid — cells along the same developmental trajectory cluster
spatially.

The full executable notebook is available at
[`tutorials/tutorial_one_cell_type.ipynb`](https://github.com/Zhen-Miao/copro-python/blob/main/tutorials/tutorial_one_cell_type.ipynb).

---

## 1. Load packages

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copro as cp

print(f"copro version: {cp.__version__}")
```

## 2. Set parameters

```python
N_PCA        = 30
CELL_TYPE    = "Epithelial"
SIGMA_VALUES = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
N_CC         = 4
```

## 3. Load data

```python
EXPR_FILE = "path/to/cell_by_gene.csv"   # cells × genes
META_FILE = "path/to/cell_attributes.csv"  # must have columns: x, y

expr_df = pd.read_csv(EXPR_FILE, index_col=0)
meta    = pd.read_csv(META_FILE, index_col=0)

# Align rows
shared  = expr_df.index.intersection(meta.index)
expr_df = expr_df.loc[shared]
meta    = meta.loc[shared]
```

## 4. Preprocess

### Cap expression outliers

```python
def cap_outliers(df, upper_quantile=0.95):
    upper = df.quantile(upper_quantile, axis=0)
    return df.clip(upper=upper, axis=1)

expr_capped = cap_outliers(expr_df, upper_quantile=0.95)
```

### Log1p normalization

```python
expr_norm = np.log1p(expr_capped.values.astype(float))
```

### Spatial coordinates

```python
# Scale by 5000 to convert pixels → mm (adjust to your data)
location = pd.DataFrame({
    "x": meta["center_x"].values / 5000,
    "y": meta["center_y"].values / 5000,
})
cell_types = np.array([CELL_TYPE] * len(meta))
```

## 5. Run CoPro pipeline

```python
obj = cp.CoProSingle(
    normalized_data=expr_norm,
    location_data=location,
    meta_data=meta.reset_index(drop=True),
    cell_types=cell_types,
)

obj = cp.subset_data(obj, [CELL_TYPE])
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

### Sigma selection

```python
nc_frames = [df for df in obj.normalized_correlation.values() if df is not None]
nc_df = pd.concat(nc_frames, ignore_index=True)

cc_indices = sorted(nc_df["CC_index"].unique())
fig, axes = plt.subplots(1, len(cc_indices), figsize=(4 * len(cc_indices), 3.5))
for ax, cc in zip(axes, cc_indices):
    sub = nc_df[nc_df["CC_index"] == cc].sort_values("sigma")
    ax.plot(sub["sigma"], sub["normalized_correlation"], "o-", color="steelblue")
    ax.axvline(obj.sigma_value_choice, color="tomato", linestyle="--")
    ax.set_title(f"CC{cc}"); ax.set_xlabel("sigma")
plt.tight_layout(); plt.show()
```

### Cell scores in situ

```python
sigma  = obj.sigma_value_choice
scores = obj.cell_scores[f"cellScores|sigma{sigma}|{CELL_TYPE}"][:, 0]
mask   = obj.cell_types_sub == CELL_TYPE
loc    = obj.location_data_sub[mask].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(5.5, 5))
sc = ax.scatter(loc["x"], loc["y"], c=scores, cmap="RdBu_r", s=6,
                vmin=-np.percentile(np.abs(scores), 99),
                vmax= np.percentile(np.abs(scores), 99))
plt.colorbar(sc, ax=ax, label="Cell score (CC1)")
ax.set_aspect("equal"); plt.tight_layout(); plt.show()
```
