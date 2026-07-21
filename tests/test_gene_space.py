"""Focused tests for multi-slide gene-space CCA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from copro.core import CoProMulti, CoProSingle, subset_data
from copro.distance import compute_distance
from copro.gene_space import (
    _prepare_gene_space_data,
    compute_genespace_objective,
    optimize_genespace_avg_corr,
    optimize_genespace_avg_corr_n,
    run_gene_space_cca,
)
from copro.kernel import compute_kernel_matrix
from copro.transfer import get_transfer_cell_scores


def _covariance_problem():
    slides = ["s1", "s2"]
    cell_types = ["A", "B"]
    strengths = (0.9, 0.45, 0.1, 0.02)
    self_covariances = {
        slide: {ct: np.eye(4) for ct in cell_types}
        for slide in slides
    }
    cross_covariances = {
        "s1": {"A-B": np.diag(strengths)},
        # Exercise reverse-pair lookup as well as averaging across slides.
        "s2": {"B-A": np.diag(strengths)},
    }
    return self_covariances, cross_covariances, slides, cell_types


def _make_multi(seed=17, counts=None):
    if counts is None:
        counts = {
            "s1": {"A": 12, "B": 12},
            "s2": {"A": 12, "B": 12},
        }
    rng = np.random.default_rng(seed)
    expression = []
    locations = []
    metadata = []
    labels = []
    loadings = np.asarray([0.50, -0.35, 0.25, 0.15, -0.10, 0.08])
    for slide_index, (slide, by_type) in enumerate(counts.items()):
        for type_index, (cell_type, n_cells) in enumerate(by_type.items()):
            position = np.linspace(0.04, 0.96, n_cells)
            common = np.sin(2 * np.pi * position)
            noise = rng.normal(scale=0.10, size=(n_cells, len(loadings)))
            block = np.exp(
                1.0
                + 0.08 * slide_index
                + common[:, None] * loadings[None, :]
                + noise
            )
            expression.append(block)
            locations.extend(
                {
                    "x": float(x + 0.012 * type_index),
                    "y": float(0.2 * np.cos(2 * np.pi * x) + 0.03 * type_index),
                }
                for x in position
            )
            metadata.extend({"slideID": slide} for _ in range(n_cells))
            labels.extend([cell_type] * n_cells)

    obj = CoProMulti(
        normalized_data=np.vstack(expression),
        location_data=pd.DataFrame(locations),
        meta_data=pd.DataFrame(metadata),
        cell_types=np.asarray(labels, dtype=object),
    )
    obj.gene_names = [f"g{i}" for i in range(obj.normalized_data.shape[1])]
    return subset_data(obj, ["A", "B"])


def test_first_component_is_deterministic_and_recovers_ranked_signal():
    self_cov, cross_cov, slides, cell_types = _covariance_problem()

    first = optimize_genespace_avg_corr(
        self_cov, cross_cov, slides, cell_types,
        max_iter=1000, tol=1e-10, verbose=False, random_state=9,
    )
    again = optimize_genespace_avg_corr(
        self_cov, cross_cov, slides, cell_types,
        max_iter=1000, tol=1e-10, verbose=False, random_state=9,
    )

    for cell_type in cell_types:
        np.testing.assert_allclose(first[cell_type], again[cell_type])
        np.testing.assert_allclose(np.linalg.norm(first[cell_type]), 1.0)
        assert abs(first[cell_type][0, 0]) > 0.999
    assert compute_genespace_objective(
        first, self_cov, cross_cov, slides, cell_types
    ) == pytest.approx(0.9, abs=1e-8)


def test_later_components_are_deterministic_and_euclidean_orthogonal():
    self_cov, cross_cov, slides, cell_types = _covariance_problem()
    first = optimize_genespace_avg_corr(
        self_cov, cross_cov, slides, cell_types,
        max_iter=1000, tol=1e-10, verbose=False, random_state=4,
    )
    weights = optimize_genespace_avg_corr_n(
        self_cov, cross_cov, slides, cell_types, first,
        n_cc=3, max_iter=1000, tol=1e-10, verbose=False, random_state=5,
    )
    again = optimize_genespace_avg_corr_n(
        self_cov, cross_cov, slides, cell_types, first,
        n_cc=3, max_iter=1000, tol=1e-10, verbose=False, random_state=5,
    )

    for cell_type in cell_types:
        np.testing.assert_allclose(weights[cell_type], again[cell_type])
        np.testing.assert_allclose(
            weights[cell_type].T @ weights[cell_type], np.eye(3), atol=1e-7
        )
        assert abs(weights[cell_type][1, 1]) > 0.999
        assert abs(weights[cell_type][2, 2]) > 0.999


def test_preparation_drops_incomplete_slides_and_handles_constant_blocks():
    obj = _make_multi(
        counts={
            "good": {"A": 12, "B": 12},
            "too_small": {"A": 12, "B": 9},
        }
    )
    # A gene that is constant only within one valid slide/type block must not
    # create NaNs after sample-standardization.
    mask = (
        (obj.meta_data_sub["slideID"].to_numpy() == "good")
        & (obj.cell_types_sub == "A")
    )
    obj.normalized_data_sub[mask, 0] = 2.0

    with pytest.warns(RuntimeWarning, match="too_small.*dropped"):
        prepared = _prepare_gene_space_data(
            obj, clip=100.0, min_prevalence=0, min_cells=1,
            cell_types=["A", "B"], slides=["good", "too_small"],
        )

    assert prepared["slides"] == ["good"]
    assert prepared["genes"].tolist() == [f"g{i}" for i in range(6)]
    assert np.all(prepared["z_by_slide"]["good"]["A"][:, 0] == 0)
    assert all(
        np.isfinite(block).all()
        for block in prepared["z_by_slide"]["good"].values()
    )


def test_sparse_preparation_matches_dense_without_full_densification(monkeypatch):
    dense_obj = _make_multi(seed=29)
    sparse_obj = _make_multi(seed=29)
    expected = _prepare_gene_space_data(
        dense_obj, clip="quantile", min_prevalence=0, min_cells=1,
        cell_types=["A", "B"], slides=["s1", "s2"],
    )
    sparse_obj.normalized_data_sub = sparse.csr_matrix(
        sparse_obj.normalized_data_sub
    )
    full_shape = sparse_obj.normalized_data_sub.shape
    densified_shapes = []
    original_toarray = sparse.csr_matrix.toarray

    def recording_toarray(self, *args, **kwargs):
        densified_shapes.append(self.shape)
        return original_toarray(self, *args, **kwargs)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", recording_toarray)
    observed = _prepare_gene_space_data(
        sparse_obj, clip="quantile", min_prevalence=0, min_cells=1,
        cell_types=["A", "B"], slides=["s1", "s2"],
    )

    assert full_shape not in densified_shapes
    assert densified_shapes == [(12, 6)] * 4
    assert observed["genes"].tolist() == expected["genes"].tolist()
    assert observed["clip_value"] == pytest.approx(expected["clip_value"])
    for slide in ("s1", "s2"):
        for cell_type in ("A", "B"):
            np.testing.assert_allclose(
                observed["z_by_slide"][slide][cell_type],
                expected["z_by_slide"][slide][cell_type],
            )


def test_slot_based_run_stores_gene_and_slide_standardized_cell_scores():
    obj = _make_multi()
    obj.skr_cca_out["existing"] = {"sentinel": np.ones((1, 1))}
    compute_distance(obj, normalize=True, truncate=True)
    compute_kernel_matrix(
        obj, [0.3], method="dense", min_ave_cell_neighbor=1,
    )

    result = run_gene_space_cca(
        obj, sigma=0.3, n_cc=2, clip=100.0,
        min_prevalence=0, min_cells=1, max_iter=1200, tol=1e-8,
        verbose=False, random_state=23,
    )

    assert result is obj
    assert "existing" in obj.skr_cca_out
    assert "gscca_sigma_0.3" in obj.skr_cca_out
    assert obj.gene_space_genes == [f"g{i}" for i in range(6)]
    np.testing.assert_array_equal(obj.gene_space_gene_indices, np.arange(6))
    for cell_type in ("A", "B"):
        assert obj.gene_scores[f"geneScores|sigma0.3|{cell_type}"].shape == (6, 2)
        scores = obj.cell_scores[f"cellScores|sigma0.3|{cell_type}"]
        ct_slides = obj.meta_data_sub.loc[
            obj.cell_types_sub == cell_type, "slideID"
        ].to_numpy()
        for slide in ("s1", "s2"):
            block = scores[ct_slides == slide]
            np.testing.assert_allclose(block.mean(axis=0), 0, atol=1e-12)
            np.testing.assert_allclose(block.std(axis=0, ddof=1), 1, atol=1e-12)
    assert "cellScore_sigma_0.3_cc_index_1" in obj.meta_data_sub.columns
    assert "cellScore_sigma_0.3_cc_index_2" in obj.meta_data_sub.columns


def test_streaming_matches_slot_based_global_normalization_without_caches():
    slot_obj = _make_multi(seed=31)
    stream_obj = _make_multi(seed=31)
    compute_distance(slot_obj, normalize=True, truncate=True)
    compute_kernel_matrix(
        slot_obj, [0.3], method="dense", min_ave_cell_neighbor=1,
    )
    run_gene_space_cca(
        slot_obj, sigma=0.3, n_cc=2, clip=100.0,
        min_prevalence=0, min_cells=1, max_iter=1200, tol=1e-8,
        verbose=False, random_state=101,
    )
    run_gene_space_cca(
        stream_obj, sigma=0.3, n_cc=2, clip=100.0,
        min_prevalence=0, min_cells=1, max_iter=1200, tol=1e-8,
        streaming=True,
        kernel_args={"min_ave_cell_neighbor": 1},
        verbose=False, random_state=101,
    )

    assert stream_obj.distances == {}
    assert stream_obj.kernel_matrices == {}
    assert stream_obj.sigma_values == [0.3]
    for cell_type in ("A", "B"):
        np.testing.assert_allclose(
            slot_obj.gene_scores[f"geneScores|sigma0.3|{cell_type}"],
            stream_obj.gene_scores[f"geneScores|sigma0.3|{cell_type}"],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            slot_obj.cell_scores[f"cellScores|sigma0.3|{cell_type}"],
            stream_obj.cell_scores[f"cellScores|sigma0.3|{cell_type}"],
            atol=1e-10,
        )


def test_sparse_streaming_never_densifies_complete_expression(monkeypatch):
    obj = _make_multi(seed=37)
    obj.normalized_data_sub = sparse.csr_matrix(obj.normalized_data_sub)
    full_shape = obj.normalized_data_sub.shape
    densified_shapes = []
    original_toarray = sparse.csr_matrix.toarray

    def recording_toarray(self, *args, **kwargs):
        densified_shapes.append(self.shape)
        return original_toarray(self, *args, **kwargs)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", recording_toarray)
    run_gene_space_cca(
        obj, sigma=0.3, n_cc=1, clip=100.0,
        min_prevalence=0, min_cells=1, max_iter=1200, tol=1e-8,
        streaming=True,
        kernel_args={"min_ave_cell_neighbor": 1},
        verbose=False, random_state=18,
    )

    assert densified_shapes
    assert full_shape not in densified_shapes
    assert all(shape == (12, 6) for shape in densified_shapes)
    assert np.isfinite(obj.cell_scores["cellScores|sigma0.3|A"]).all()


def test_gene_space_requires_multi_slide_object():
    obj = CoProSingle(
        normalized_data=np.ones((20, 3)),
        location_data=pd.DataFrame({"x": np.arange(20), "y": np.arange(20)}),
        meta_data=pd.DataFrame(index=np.arange(20)),
        cell_types=np.asarray(["A"] * 10 + ["B"] * 10),
    )
    with pytest.raises(TypeError, match="CoProMulti"):
        run_gene_space_cca(obj, sigma=0.3, verbose=False)


def test_filtered_gene_space_weights_transfer_by_gene_identity():
    reference = _make_multi(seed=44)
    target = _make_multi(seed=45)
    # Force one gene below the global prevalence/absolute-count filter while
    # retaining the full six-gene expression matrices on both objects.
    reference.normalized_data_sub[:, 5] = 0.0
    compute_distance(reference, normalize=True, truncate=True)
    compute_kernel_matrix(
        reference, [0.3], method="dense", min_ave_cell_neighbor=1
    )
    run_gene_space_cca(
        reference, sigma=0.3, n_cc=1, clip=100.0,
        min_prevalence=0.1, min_cells=2, max_iter=1000, tol=1e-8,
        verbose=False, random_state=12,
    )
    assert reference.gene_space_genes == [f"g{i}" for i in range(5)]

    transferred = get_transfer_cell_scores(
        reference, target, 0.3, gene_score_type="gene_space",
        use_quantile_normalization=False, verbose=False,
    )
    assert set(transferred) == {"A", "B"}
    assert transferred["A"].shape == (24, 1)
    assert transferred["B"].shape == (24, 1)
    assert np.isfinite(transferred["A"]).all()
