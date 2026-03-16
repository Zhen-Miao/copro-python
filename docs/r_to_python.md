# R ↔ Python Reference

This page maps every R CoPro function and slot to its Python equivalent.
Users familiar with the R package can use this as a migration guide.

---

## Object construction

| R | Python | Notes |
|---|--------|-------|
| `newCoProSingle(normalizedData, locationData, metaData, cellTypes)` | `cp.CoProSingle(normalized_data, location_data, meta_data, cell_types)` | Arguments use `snake_case` in Python |
| `newCoProMulti(...)` | `cp.CoProMulti(...)` | Requires `slideID` column in `meta_data` |

---

## Pipeline functions

| R | Python | Key argument changes |
|---|--------|----------------------|
| `subsetData(obj, cellTypesOfInterest)` | `cp.subset_data(obj, cell_types_of_interest)` | |
| `computePCA(obj, nPCA, center, scale.)` | `cp.compute_pca(obj, n_pca, center, scale)` | `scale.` → `scale` |
| `computeDistance(obj, distType, normalizeDistance)` | `cp.compute_distance(obj, normalize)` | `distType` fixed to `"Euclidean2D"` |
| `computeKernelMatrix(obj, sigmaValues, upperQuantile, lowerLimit)` | `cp.compute_kernel_matrix(obj, sigma_values, upper_quantile, lower_limit)` | |
| `runSkrCCA(obj, scalePCs, nCC, maxIter, tol)` | `cp.run_skr_cca(obj, scale_pcs, n_cc, max_iter, tol)` | |
| `computeNormalizedCorrelation(obj, tol)` | `cp.compute_normalized_correlation(obj, tol)` | |
| `computeGeneAndCellScores(obj)` | `cp.compute_gene_and_cell_scores(obj)` | |

---

## Accessor functions

R provides helper functions to extract results; in Python these are accessed directly as object attributes.

| R | Python |
|---|--------|
| `getNormCorr(obj)` | `pd.concat(obj.normalized_correlation.values())` |
| `getCellScoresInSitu(obj, sigmaValueChoice)` | See below |
| `getCorrOneType(obj, sigmaValueChoice, cellTypeA, ccIndex)` | See below |
| `getCorrTwoTypes(obj, sigmaValueChoice, cellTypeA, cellTypeB)` | See below |

### `getCellScoresInSitu` equivalent

=== "One cell type"

    ```python
    sigma  = obj.sigma_value_choice
    ct     = "Epithelial"
    scores = obj.cell_scores[f"cellScores|sigma{sigma}|{ct}"][:, 0]
    mask   = obj.cell_types_sub == ct
    loc    = obj.location_data_sub[mask].reset_index(drop=True)

    # loc has columns "x", "y"; scores is the CC1 cell score vector
    ```

=== "Two cell types"

    ```python
    sigma    = obj.sigma_value_choice
    scores_A = obj.cell_scores[f"cellScores|sigma{sigma}|{cell_type_A}"][:, 0]
    scores_B = obj.cell_scores[f"cellScores|sigma{sigma}|{cell_type_B}"][:, 0]

    mask_A = obj.cell_types_sub == cell_type_A
    mask_B = obj.cell_types_sub == cell_type_B
    loc_A  = obj.location_data_sub[mask_A].reset_index(drop=True)
    loc_B  = obj.location_data_sub[mask_B].reset_index(drop=True)
    ```

### `getCorrOneType` equivalent

```python
sigma    = obj.sigma_value_choice   # or any sigma value
ct       = "Epithelial"
scores   = obj.cell_scores[f"cellScores|sigma{sigma}|{ct}"][:, 0]
K        = obj.kernel_matrices[f"kernel|sigma{sigma}|{ct}|{ct}"]
k_scores = K @ scores               # kernel-smoothed scores (AK in R)

# Scatter: k_scores (x-axis) vs scores (y-axis)
```

### `getCorrTwoTypes` equivalent

```python
sigma     = obj.sigma_value_choice
scores_A  = obj.cell_scores[f"cellScores|sigma{sigma}|{cell_type_A}"][:, 0]
scores_B  = obj.cell_scores[f"cellScores|sigma{sigma}|{cell_type_B}"][:, 0]
K_AB      = obj.kernel_matrices[f"kernel|sigma{sigma}|{cell_type_A}|{cell_type_B}"]
KA_scores = K_AB.T @ scores_A       # AK column in R's output

# Scatter: KA_scores (x-axis) vs scores_B (y-axis)
```

---

## Object slots

| R slot | Python attribute | Description |
|--------|-----------------|-------------|
| `obj@normalizedData` | `obj.normalized_data` | Input expression matrix |
| `obj@locationData` | `obj.location_data` | Input spatial coordinates |
| `obj@metaData` | `obj.meta_data` | Input metadata |
| `obj@cellTypes` | `obj.cell_types` | Input cell type labels |
| `obj@cellTypesOfInterest` | `obj.cell_types_of_interest` | Subset cell types |
| `obj@normalizedDataSub` | `obj.normalized_data_sub` | Subsetted expression |
| `obj@locationDataSub` | `obj.location_data_sub` | Subsetted coordinates |
| `obj@pcaGlobal[[ct]]$x` | `obj.pca_global[ct]["scores"]` | PC scores (cells × PCs) |
| `obj@pcaGlobal[[ct]]$sdev` | `obj.pca_global[ct]["sdev"]` | PC standard deviations |
| `obj@pcaGlobal[[ct]]$rotation` | `obj.pca_global[ct]["rotation"]` | Gene loadings (genes × PCs) |
| `obj@kernelMatrices[["kernel\|sigma0.1\|A\|B"]]` | `obj.kernel_matrices["kernel\|sigma0.1\|A\|B"]` | Kernel matrix |
| `obj@skrCCAOut[["sigma_0.1"]][["A"]]` | `obj.skr_cca_out["sigma_0.1"]["A"]` | CCA weight matrix (PCs × CCs) |
| `obj@normalizedCorrelation` | `obj.normalized_correlation` | Dict of DataFrames per σ |
| `obj@sigmaValueChoice` | `obj.sigma_value_choice` | Selected σ |
| `obj@cellScores[["cellScores\|sigma0.1\|A"]]` | `obj.cell_scores["cellScores\|sigma0.1\|A"]` | Cell scores (cells × CCs) |
| `obj@geneScores[["geneScores\|sigma0.1\|A"]]` | `obj.gene_scores["geneScores\|sigma0.1\|A"]` | Gene scores (genes × CCs) |

---

## Naming conventions

| Convention | R | Python |
|-----------|---|--------|
| Function names | `camelCase` | `snake_case` |
| Argument names | `camelCase` | `snake_case` |
| Slot access | `@` operator | `.` attribute |
| Object class | S4 (`CoProSingle`) | dataclass (`CoProSingle`) |
| Key separator in dict names | `\|` | `\|` (identical) |

---

## Features not yet in Python

| R function | Status |
|------------|--------|
| `runSkrCCAPermu()` | Not yet implemented |
| `computeNormalizedCorrelationPermu()` | Not yet implemented |
| `distType = "Morphology-Aware"` | Not yet implemented (`"Euclidean2D"` only) |
