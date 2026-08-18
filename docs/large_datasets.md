# Handling large spatial datasets

The main memory risk in a large CoPro run is the dense cell-by-cell distance
matrix. Subset to the cell types of interest first, keep imaging-panel PCA
small (typically 10–15 components), and build sparse kernels directly from
coordinates.

## Keep the expression matrix sparse

Pass `normalized_data` as a SciPy sparse matrix (or an AnnData whose `X` is
sparse) and `compute_pca()` will never densify it. Centering would destroy
sparsity, so the standardized matrix is applied as a matrix-free operator and
the truncated SVD only ever asks for products with it — the same technique R
CoPro uses via `irlba`'s `center`/`scale.` arguments. Results are identical to
the dense path to machine precision, including the component sign convention.

On a 60,000-cell × 3,000-gene block at 8% density, `compute_pca(n_pca=30)`
drops from 5.4 GB peak RSS / 47 s to 1.1 GB / 12 s. The saving grows with the
block size, and even for a fairly dense block (~34% non-zero) peak memory is
still roughly halved at comparable runtime.

```python
import copro as cp

obj = cp.from_anndata(
    adata,
    cell_type_key="cell_type",
    slide_key="sample",
)
obj = cp.subset_data(obj, ["Type A", "Type B"])
obj = cp.compute_pca(obj, n_pca=15)

# No compute_distance() call: omitted Gaussian values would already be below
# lower_limit, so the fixed-radius CSR result is equivalent to the dense path.
obj = cp.compute_sparse_kernel(
    obj,
    sigma_values=[0.05, 0.1, 0.2],
    dist_type="Euclidean2D",
)
obj = cp.run_skr_cca(obj, n_cc=2)
```

`compute_kernel_matrix(method="auto")` chooses the sparse path when a
per-slide cell-type block reaches `auto_threshold` (5,000 by default), or when
the estimated total dense workload reaches `auto_threshold ** 2`. For a small
workload, automatic mode selects the dense path and therefore requires
`compute_distance()` first. Use `method="sparse"` or `compute_sparse_kernel()`
to force coordinate-to-kernel construction.

By default `compute_kernel_matrix()` clears stored distances after building
kernels. Pass `drop_distances=False` only when you need to inspect or reuse the
dense distances.

For multi-slide gene-space CCA, streaming mode reduces each slide/pair to a
gene-by-gene covariance before moving on, without populating the distance or
kernel caches:

```python
obj = cp.run_gene_space_cca(
    obj,
    sigma=0.1,
    n_cc=2,
    streaming=True,
    distance_args={"normalization_scope": "global"},
)
```

Global distance normalization is the default and matches the slot-based
pipeline. Use `normalization_scope="per_slide"` only when slides genuinely use
different coordinate scales.

When `normalized_data_sub` is a SciPy sparse matrix, streaming mode preserves
that sparse representation during filtering and clipping. It materializes
standardized expression for only one slide at a time while building
covariances, and one slide/cell-type block at a time while writing scores; it
never creates a complete dense cells-by-genes copy. Peak memory still includes
the retained gene-by-gene covariance matrices, so aggressive gene filtering
may be necessary when the gene set is very large.
