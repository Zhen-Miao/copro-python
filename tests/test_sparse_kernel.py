from copy import deepcopy
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from copro.core import CoProMulti, CoProSingle, subset_data
from copro.distance import compute_distance, compute_self_distance
from copro.kernel import (
    compute_kernel_matrix,
    compute_self_kernel,
    compute_sparse_kernel,
)


SIGMAS = [0.5, 1.0]


def _single_object(with_z: bool = False) -> CoProSingle:
    rng = np.random.default_rng(20260720)
    n_per_type = 24
    n = 2 * n_per_type
    location = pd.DataFrame(
        {
            "x": rng.uniform(0, 10, n),
            "y": rng.uniform(0, 8, n),
        }
    )
    if with_z:
        location["z"] = rng.uniform(0, 3, n)
    # Exercise the dense-compatible zero-distance replacement path.
    location.loc[1, ["x", "y"]] = location.loc[0, ["x", "y"]].to_numpy()
    location.loc[n_per_type + 1, ["x", "y"]] = location.loc[
        n_per_type, ["x", "y"]
    ].to_numpy()
    if with_z:
        location.loc[1, "z"] = location.loc[0, "z"]
        location.loc[n_per_type + 1, "z"] = location.loc[n_per_type, "z"]

    obj = CoProSingle(
        normalized_data=rng.normal(size=(n, 6)),
        location_data=location,
        meta_data=pd.DataFrame(index=np.arange(n)),
        cell_types=np.array(["A"] * n_per_type + ["B"] * n_per_type),
    )
    return subset_data(obj, ["A", "B"])


def _multi_object() -> CoProMulti:
    rng = np.random.default_rng(1204)
    rows = []
    cell_types = []
    slide_ids = []
    for slide_index, slide in enumerate(["s1", "s2"]):
        for ct_index, ct in enumerate(["A", "B"]):
            xyz = rng.normal(
                loc=[5 * slide_index, 3 * ct_index, slide_index],
                scale=[1.2, 1.0, 0.3],
                size=(12, 3),
            )
            rows.append(xyz)
            cell_types.extend([ct] * len(xyz))
            slide_ids.extend([slide] * len(xyz))
    coords = np.vstack(rows)
    meta = pd.DataFrame({"slideID": slide_ids})
    obj = CoProMulti(
        normalized_data=rng.normal(size=(len(coords), 5)),
        location_data=pd.DataFrame(coords, columns=["x", "y", "z"]),
        meta_data=meta,
        cell_types=np.asarray(cell_types),
        slide_list=["s1", "s2"],
    )
    return subset_data(obj, ["A", "B"])


def _mixed_validity_multi_object(*, all_invalid: bool = False) -> CoProMulti:
    """Two slides: s1 is valid unless requested; s2 is always too separated."""
    rng = np.random.default_rng(99)
    coords = []
    cell_types = []
    slides = []
    configurations = [
        (
            "s1",
            {"A": (0.0, 0.0), "B": (100.0, 100.0)}
            if all_invalid
            else {"A": (0.0, 0.0), "B": (0.05, 0.05)},
        ),
        ("s2", {"A": (0.0, 0.0), "B": (100.0, 100.0)}),
    ]
    for slide, centers in configurations:
        for ct in ["A", "B"]:
            coords.append(rng.normal(centers[ct], 0.03, size=(6, 2)))
            cell_types.extend([ct] * 6)
            slides.extend([slide] * 6)
    location = np.vstack(coords)
    obj = CoProMulti(
        normalized_data=rng.normal(size=(len(location), 4)),
        location_data=pd.DataFrame(location, columns=["x", "y"]),
        meta_data=pd.DataFrame({"slideID": slides}),
        cell_types=np.asarray(cell_types),
        slide_list=["s1", "s2"],
    )
    return subset_data(obj, ["A", "B"])


@pytest.mark.parametrize(
    "normalization",
    ["none", "global", "row", "column"],
)
def test_sparse_matches_dense_single_2d(normalization):
    base = _single_object()
    dense_obj = deepcopy(base)
    sparse_obj = deepcopy(base)

    kwargs = {
        "normalize_kernel": normalization == "global",
        "row_normalize_kernel": normalization == "row",
        "col_normalize_kernel": normalization == "column",
    }
    dense_obj = compute_distance(
        dense_obj,
        dist_type="Euclidean2D",
        normalize=True,
        normalize_target=0.1,
        truncate=True,
    )
    dense_obj = compute_kernel_matrix(
        dense_obj,
        SIGMAS,
        method="dense",
        min_ave_cell_neighbor=1,
        **kwargs,
    )
    sparse_obj = compute_sparse_kernel(
        sparse_obj,
        SIGMAS,
        dist_type="Euclidean2D",
        normalize_distance=True,
        normalize_target=0.1,
        truncate=True,
        min_ave_cell_neighbor=1,
        **kwargs,
    )

    assert sparse_obj.distance_scale_factor == pytest.approx(
        dense_obj.distance_scale_factor
    )
    for sigma in SIGMAS:
        key = f"kernel|sigma{sigma}|A|B"
        observed = sparse_obj.kernel_matrices[key]
        assert sparse.isspmatrix_csr(observed)
        np.testing.assert_allclose(
            observed.toarray(), dense_obj.kernel_matrices[key], atol=2e-14
        )


def test_sparse_multi_slide_3d_and_scaled_coordinates():
    obj = compute_sparse_kernel(
        _multi_object(),
        [0.8],
        dist_type="Euclidean3D",
        x_dist_scale=1.5,
        y_dist_scale=0.75,
        z_dist_scale=2.0,
        normalize_target=0.2,
        min_ave_cell_neighbor=1,
    )

    assert obj.distances == {}
    assert np.isfinite(obj.distance_scale_factor)
    for slide in obj.slide_list:
        key = f"kernel|sigma0.8|{slide}|A|B"
        assert sparse.isspmatrix_csr(obj.kernel_matrices[key])
        assert obj.kernel_matrices[key].shape == (12, 12)


def test_auto_dispatch_and_drop_distances():
    dense_obj = compute_distance(_single_object(), normalize=False)
    dense_obj = compute_kernel_matrix(
        dense_obj,
        [1.0],
        method="auto",
        drop_distances=False,
        auto_threshold=10_000,
        min_ave_cell_neighbor=1,
    )
    assert isinstance(dense_obj.kernel_matrices["kernel|sigma1.0|A|B"], np.ndarray)
    assert dense_obj.distances

    dropped_obj = compute_distance(_single_object(), normalize=False)
    dropped_obj = compute_kernel_matrix(
        dropped_obj,
        [1.0],
        method="dense",
        drop_distances=True,
        min_ave_cell_neighbor=1,
    )
    assert dropped_obj.distances == {}

    sparse_obj = compute_kernel_matrix(
        _single_object(),
        [1.0],
        method="auto",
        auto_threshold=1,
        normalize_distance=False,
        min_ave_cell_neighbor=1,
    )
    assert sparse.issparse(sparse_obj.kernel_matrices["kernel|sigma1.0|A|B"])


def test_sparse_auto_self_kernels_match_dense_and_persist_scale():
    base = _single_object()
    dense_obj = compute_self_distance(
        deepcopy(base),
        normalize=True,
        normalize_target=0.1,
        truncate=True,
        verbose=False,
    )
    dense_obj = compute_self_kernel(
        dense_obj,
        SIGMAS,
        method="dense",
        min_ave_cell_neighbor=1,
        verbose=False,
    )

    sparse_obj = compute_self_kernel(
        deepcopy(base),
        SIGMAS,
        method="auto",  # Missing self distances must select the sparse path.
        auto_threshold=10_000,
        normalize_distance=True,
        normalize_target=0.1,
        min_ave_cell_neighbor=1,
        verbose=False,
    )
    assert sparse_obj.distance_scale_factor == pytest.approx(
        dense_obj.distance_scale_factor
    )
    for sigma in SIGMAS:
        for ct in ["A", "B"]:
            key = f"kernel|sigma{sigma}|{ct}|{ct}"
            observed = sparse_obj.kernel_matrices[key]
            assert sparse.isspmatrix_csr(observed)
            assert np.all(observed.diagonal() == 0)
            np.testing.assert_allclose(
                observed.toarray(), dense_obj.kernel_matrices[key], atol=2e-14
            )


def test_sparse_rejects_fully_degenerate_coordinates():
    obj = _single_object()
    obj.location_data_sub.loc[:, ["x", "y"]] = 1.0
    with pytest.raises(ValueError, match="coincident|low-distance percentile"):
        compute_sparse_kernel(
            obj,
            [1.0],
            normalize_target=0.1,
            min_ave_cell_neighbor=1,
        )


def test_multi_slide_dense_sparse_and_auto_keep_sigma_when_any_block_is_valid():
    base = _mixed_validity_multi_object()
    dense_input = compute_distance(deepcopy(base), normalize=False)
    auto_dense_input = compute_distance(deepcopy(base), normalize=False)

    results = {}
    with pytest.warns(UserWarning, match="s2.*too sparse"):
        results["dense"] = compute_kernel_matrix(
            dense_input,
            [0.1],
            method="dense",
            min_ave_cell_neighbor=1,
        )
    with pytest.warns(UserWarning, match="s2.*too sparse"):
        results["sparse"] = compute_kernel_matrix(
            deepcopy(base),
            [0.1],
            method="sparse",
            normalize_distance=False,
            min_ave_cell_neighbor=1,
        )
    with pytest.warns(UserWarning, match="s2.*too sparse"):
        results["auto_dense"] = compute_kernel_matrix(
            auto_dense_input,
            [0.1],
            method="auto",
            auto_threshold=10_000,
            min_ave_cell_neighbor=1,
        )
    with pytest.warns(UserWarning, match="s2.*too sparse"):
        results["auto_sparse"] = compute_kernel_matrix(
            deepcopy(base),
            [0.1],
            method="auto",
            auto_threshold=1,
            normalize_distance=False,
            min_ave_cell_neighbor=1,
        )

    valid_key = "kernel|sigma0.1|s1|A|B"
    invalid_key = "kernel|sigma0.1|s2|A|B"
    for obj in results.values():
        assert obj.sigma_values == [0.1]
        assert valid_key in obj.kernel_matrices
        assert invalid_key not in obj.kernel_matrices

    expected = results["dense"].kernel_matrices[valid_key]
    for name, obj in results.items():
        observed = obj.kernel_matrices[valid_key]
        if sparse.issparse(observed):
            observed = observed.toarray()
        np.testing.assert_allclose(observed, expected, atol=2e-14, err_msg=name)


@pytest.mark.parametrize(
    ("method", "auto_threshold"),
    [
        ("dense", 5000),
        ("sparse", 5000),
        ("auto", 10_000),  # auto -> dense
        ("auto", 1),       # auto -> sparse
    ],
)
def test_multi_slide_single_sigma_fails_only_when_all_blocks_invalid(
    method, auto_threshold
):
    obj = _mixed_validity_multi_object(all_invalid=True)
    if method == "dense" or (method == "auto" and auto_threshold > 1):
        obj = compute_distance(obj, normalize=False)

    with pytest.warns(UserWarning, match="too sparse"):
        with pytest.raises(ValueError, match="no valid kernel blocks.*larger sigma"):
            compute_kernel_matrix(
                obj,
                [0.1],
                method=method,
                auto_threshold=auto_threshold,
                normalize_distance=False,
                min_ave_cell_neighbor=1,
            )


@pytest.mark.parametrize(
    ("method", "auto_threshold"),
    [
        ("dense", 5000),
        ("sparse", 5000),
        ("auto", 10_000),
        ("auto", 1),
    ],
)
def test_multi_slide_removes_only_all_invalid_sigma_when_alternative_survives(
    method, auto_threshold
):
    obj = _mixed_validity_multi_object(all_invalid=True)
    if method == "dense" or (method == "auto" and auto_threshold > 1):
        obj = compute_distance(obj, normalize=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = compute_kernel_matrix(
            obj,
            [0.1, 100.0],
            method=method,
            auto_threshold=auto_threshold,
            normalize_distance=False,
            min_ave_cell_neighbor=1,
        )
    assert any("Removing sigma=0.1" in str(item.message) for item in caught)
    assert obj.sigma_values == [100.0]
    assert not any(key.startswith("kernel|sigma0.1|") for key in obj.kernel_matrices)
    assert any(key.startswith("kernel|sigma100.0|") for key in obj.kernel_matrices)


def test_invalid_self_sigma_preserves_existing_cross_kernels_and_overwrite_is_explicit():
    obj = _multi_object()
    cross_key = "kernel|sigma0.001|s1|A|B"
    obj.kernel_matrices[cross_key] = sparse.eye(12, format="csr")
    obj.sigma_values = [0.001]

    with pytest.warns(UserWarning, match="too sparse"):
        obj = compute_self_kernel(
            obj,
            [0.001],
            method="sparse",
            normalize_distance=False,
            min_ave_cell_neighbor=1,
            verbose=False,
        )
    assert cross_key in obj.kernel_matrices
    assert not any(
        key.startswith("kernel|sigma0.001|") and key.split("|")[-2] == key.split("|")[-1]
        for key in obj.kernel_matrices
    )

    overwrite_obj = _multi_object()
    overwrite_obj.kernel_matrices[cross_key] = sparse.eye(12, format="csr")
    overwrite_obj.sigma_values = [1.0]
    overwrite_obj = compute_self_kernel(
        overwrite_obj,
        [1.0],
        method="sparse",
        normalize_target=0.1,
        min_ave_cell_neighbor=1,
        overwrite=True,
        verbose=False,
    )
    assert cross_key not in overwrite_obj.kernel_matrices
