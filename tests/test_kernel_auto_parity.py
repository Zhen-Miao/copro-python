"""Dense/sparse parity for the default ``compute_kernel_matrix`` entry point.

These tests lock in the parity fixes made in ``copro/kernel.py``:

* Facet B -- the sparse branch's default ``dist_type`` now resolves to
  ``Euclidean2D`` (matching ``compute_distance``), so on x/y/z data the dense
  branch (2D distances from ``compute_distance``) and the sparse branch produce
  the SAME kernels for the identical default call.
* Facet A -- the dense branch consumes the precomputed ``obj.distances`` as-is,
  so the distance-processing options are honored only by the sparse branch. With
  CONSISTENT options the two branches agree elementwise; when the precomputed
  distances disagree with the requested normalization the dense branch now warns
  instead of silently returning inconsistent kernels.
* Defect 2 -- single-slide self blocks with <= 5 cells are skipped like the
  dense path, and a 1-cell within block raises an actionable error.

Together these make ``method='auto'`` result-invariant across the dense/sparse
size threshold for the same arguments.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from copro.core import CoProSingle, subset_data
from copro.distance import compute_distance, compute_self_distance
from copro.kernel import compute_kernel_matrix, compute_self_kernel


SIGMAS = [0.5, 1.0]


# ---------------------------------------------------------------------------
# Object builders and comparison helpers (mirroring tests/test_sparse_kernel.py)
# ---------------------------------------------------------------------------
def _xyz_single_object(n_per_type: int = 24, seed: int = 20260721) -> CoProSingle:
    """Single-slide, two-cell-type object carrying an x/y/z coordinate frame.

    The z column is what makes the pre-fix sparse branch default to Euclidean3D
    while the dense branch used Euclidean2D distances (Facet B).
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_type
    location = pd.DataFrame(
        {
            "x": rng.uniform(0, 10, n),
            "y": rng.uniform(0, 8, n),
            "z": rng.uniform(0, 3, n),
        }
    )
    obj = CoProSingle(
        normalized_data=rng.normal(size=(n, 6)),
        location_data=location,
        meta_data=pd.DataFrame(index=np.arange(n)),
        cell_types=np.array(["A"] * n_per_type + ["B"] * n_per_type),
    )
    return subset_data(obj, ["A", "B"])


def _uneven_single_object(sizes: dict, seed: int = 41) -> CoProSingle:
    """Single-slide object with per-type cell counts given by ``sizes``.

    Each cell type is placed around its own well-separated centre so distance
    blocks are well conditioned; ``min_cells`` is lowered to 1 so 1-5 cell types
    survive ``subset_data`` (the regime that exposes Defect 2).
    """
    rng = np.random.default_rng(seed)
    cell_types: list[str] = []
    coords = []
    for ct, n in sizes.items():
        cell_types.extend([ct] * n)
        center = rng.uniform(0, 50, size=2)
        coords.append(rng.normal(center, 1.0, size=(n, 2)))
    location = pd.DataFrame(np.vstack(coords), columns=["x", "y"])
    total = len(cell_types)
    obj = CoProSingle(
        normalized_data=rng.normal(size=(total, 4)),
        location_data=location,
        meta_data=pd.DataFrame(index=np.arange(total)),
        cell_types=np.asarray(cell_types),
    )
    return subset_data(obj, list(sizes.keys()), min_cells=1)


def _dense(mat):
    return mat.toarray() if sparse.issparse(mat) else np.asarray(mat)


def _assert_kernels_equal(obj_a, obj_b, atol: float = 2e-14) -> None:
    keys_a = set(obj_a.kernel_matrices)
    keys_b = set(obj_b.kernel_matrices)
    assert keys_a == keys_b, f"kernel key sets differ: {keys_a ^ keys_b}"
    for key, a in obj_a.kernel_matrices.items():
        np.testing.assert_allclose(
            _dense(a), _dense(obj_b.kernel_matrices[key]), atol=atol, err_msg=key
        )


def _self_types(mapping) -> set:
    """Cell types with a within-type ('|ct|ct') key in a distances/kernels dict."""
    return {
        k.split("|")[-1]
        for k in mapping
        if k.split("|")[-1] == k.split("|")[-2]
    }


# ---------------------------------------------------------------------------
# (a) DEFAULT arguments on x/y/z data -> dense == sparse, and auto agrees
# ---------------------------------------------------------------------------
def test_auto_parity_defaults_on_xyz_data():
    base = _xyz_single_object()

    # Dense branch: compute_distance() default is Euclidean2D.
    dense = compute_distance(deepcopy(base))
    dense = compute_kernel_matrix(
        dense, SIGMAS, method="dense", min_ave_cell_neighbor=1
    )

    # Sparse branch with DEFAULT dist_type (=None). Before the Facet B fix this
    # defaulted to Euclidean3D on z-bearing data and diverged from the dense 2D
    # kernels for the same default call.
    sparse_obj = compute_kernel_matrix(
        deepcopy(base), SIGMAS, method="sparse", min_ave_cell_neighbor=1
    )

    _assert_kernels_equal(dense, sparse_obj)

    # method='auto' must return the SAME kernels whichever branch it selects.
    auto_dense = compute_kernel_matrix(
        compute_distance(deepcopy(base)),
        SIGMAS,
        method="auto",
        auto_threshold=10_000,  # small workload -> dense
        min_ave_cell_neighbor=1,
    )
    auto_sparse = compute_kernel_matrix(
        deepcopy(base),
        SIGMAS,
        method="auto",
        auto_threshold=1,  # any block -> sparse
        min_ave_cell_neighbor=1,
    )
    assert all(
        isinstance(v, np.ndarray) for v in auto_dense.kernel_matrices.values()
    )
    assert all(sparse.issparse(v) for v in auto_sparse.kernel_matrices.values())
    _assert_kernels_equal(auto_dense, sparse_obj)
    _assert_kernels_equal(auto_sparse, dense)


# ---------------------------------------------------------------------------
# (b) normalize=False parity under consistent options + non-silent mismatch
# ---------------------------------------------------------------------------
def test_auto_parity_normalize_false_consistent_and_mismatch_warns():
    base = _xyz_single_object()

    # Consistent options: distances built with normalize=False, kernels asked
    # with normalize_distance=False -> genuine elementwise parity.
    dense = compute_distance(deepcopy(base), normalize=False)
    dense = compute_kernel_matrix(
        dense,
        SIGMAS,
        method="dense",
        normalize_distance=False,
        min_ave_cell_neighbor=1,
    )
    sparse_obj = compute_kernel_matrix(
        deepcopy(base),
        SIGMAS,
        method="sparse",
        normalize_distance=False,
        min_ave_cell_neighbor=1,
    )
    _assert_kernels_equal(dense, sparse_obj)
    assert dense.distance_scale_factor == pytest.approx(
        sparse_obj.distance_scale_factor
    )

    # Mismatch: distances built with normalize=False but the DEFAULT call
    # requests normalize_distance=True. The dense branch reuses the
    # un-normalized distances (it cannot honor normalize_distance without
    # rebuilding distances), so it must WARN instead of silently returning
    # kernels that differ from the sparse branch -- the Facet A fix for the
    # "silent no-op" reproduction (dense frob 0.548 vs sparse 9.998).
    mismatched = compute_distance(deepcopy(base), normalize=False)
    with pytest.warns(UserWarning, match="honored only on the sparse branch"):
        compute_kernel_matrix(
            mismatched, SIGMAS, method="dense", min_ave_cell_neighbor=1
        )


# ---------------------------------------------------------------------------
# (c) Explicit 3D elementwise dense/sparse parity (stronger than type/shape)
# ---------------------------------------------------------------------------
def test_auto_parity_explicit_euclidean3d_elementwise():
    base = _xyz_single_object()

    dense = compute_distance(deepcopy(base), dist_type="Euclidean3D")
    dense = compute_kernel_matrix(
        dense,
        SIGMAS,
        method="dense",
        dist_type="Euclidean3D",
        min_ave_cell_neighbor=1,
    )
    sparse_obj = compute_kernel_matrix(
        deepcopy(base),
        SIGMAS,
        method="sparse",
        dist_type="Euclidean3D",
        min_ave_cell_neighbor=1,
    )
    _assert_kernels_equal(dense, sparse_obj)

    # Sanity: dist_type actually changes the result, so the 3D parity above is
    # not vacuously equal to the 2D case.
    sparse_2d = compute_kernel_matrix(
        deepcopy(base),
        SIGMAS,
        method="sparse",
        dist_type="Euclidean2D",
        min_ave_cell_neighbor=1,
    )
    key = "kernel|sigma1.0|A|B"
    assert not np.allclose(
        sparse_obj.kernel_matrices[key].toarray(),
        sparse_2d.kernel_matrices[key].toarray(),
    )


# ---------------------------------------------------------------------------
# (d) Defect 2: single-slide self-block skipping and 1-cell error message
# ---------------------------------------------------------------------------
def test_defect2_single_slide_self_blocks_skip_like_dense():
    base = _uneven_single_object({"A": 20, "B": 18, "C": 3, "D": 1})

    # Dense reference: compute_self_distance() (distance.py) skips any cell type
    # with <= 5 cells, so C (3) and D (1) never get a self block.
    dense_ref = compute_self_distance(deepcopy(base), verbose=False)
    assert _self_types(dense_ref.distances) == {"A", "B"}

    # Sparse self-kernel must skip the SAME small self blocks. Before the fix a
    # single-slide object never applied the <= 5 guard, so C produced a
    # sparse-only self-kernel and D (1 cell) reached _low_percentile_block with
    # total = 0 and aborted the entire self-kernel run.
    sparse_obj = compute_self_kernel(
        deepcopy(base),
        SIGMAS,
        method="sparse",
        min_ave_cell_neighbor=1,
        verbose=False,
    )
    sparse_self = _self_types(sparse_obj.kernel_matrices)
    assert sparse_self == {"A", "B"}
    assert sparse_self == _self_types(dense_ref.distances)


def test_defect2_one_cell_within_block_raises_actionable_message():
    # A lone within/self block with a single cell can no longer be skipped (it is
    # the only block), so it must raise a CLEAR, actionable error rather than the
    # opaque "too few cells for distances" abort.
    obj = _uneven_single_object({"C": 1}, seed=7)
    with pytest.raises(ValueError, match="Increase min_cells"):
        compute_kernel_matrix(obj, [1.0], method="sparse", min_ave_cell_neighbor=1)
