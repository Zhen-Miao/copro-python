"""Multi-slide parity tests for transferred-score correlations."""

import numpy as np
import pandas as pd
import pytest

from copro.core import CoProMulti
from copro.correlation import (
    _compute_bidir_corrs_all_cc,
    _get_kernel_for_pair,
    _kernel_normalizer,
)
from copro.transfer import get_transfer_bidir_corr, get_transfer_norm_corr


def _make_multi_target():
    slides = np.array(["s1"] * 6 + ["s2"] * 6)
    cell_types = np.array(
        ["A"] * 3 + ["B"] * 3 + ["A"] * 3 + ["B"] * 3
    )
    meta = pd.DataFrame({"slideID": slides})
    obj = CoProMulti(
        normalized_data=np.zeros((12, 2)),
        location_data=pd.DataFrame(
            {"x": np.arange(12, dtype=float), "y": np.zeros(12)}
        ),
        meta_data=meta.copy(),
        cell_types=cell_types,
        slide_list=["s1", "s2"],
    )
    obj.normalized_data_sub = obj.normalized_data.copy()
    obj.location_data_sub = obj.location_data.copy()
    obj.cell_types_sub = cell_types.copy()
    obj.meta_data_sub = meta.copy()
    obj.cell_types_of_interest = ["A", "B"]

    cross = {
        (1.0, "s1"): np.array(
            [[2.0, 0.4, 0.1], [0.2, 1.8, 0.5], [0.1, 0.3, 1.5]]
        ),
        (1.0, "s2"): np.array(
            [[1.2, 0.8, 0.2], [0.5, 1.5, 0.4], [0.2, 0.9, 1.7]]
        ),
        (2.0, "s1"): np.array(
            [[0.2, 0.6, 1.8], [1.6, 0.3, 0.4], [0.5, 1.7, 0.2]]
        ),
        (2.0, "s2"): np.array(
            [[0.4, 1.5, 0.3], [0.2, 0.5, 1.6], [1.7, 0.4, 0.2]]
        ),
    }
    self_a = np.array(
        [[1.0, 0.4, 0.1], [0.4, 1.0, 0.3], [0.1, 0.3, 1.0]]
    )
    self_b = np.array(
        [[1.0, 0.2, 0.3], [0.2, 1.0, 0.5], [0.3, 0.5, 1.0]]
    )
    for (sigma, slide), K in cross.items():
        obj.kernel_matrices[f"kernel|sigma{sigma}|{slide}|A|B"] = K
        # Make target sigma selection affect both the cross-kernel and its
        # matched-sigma whitening operators.
        scale = 1.0 if sigma == 1.0 else 0.65
        obj.kernel_matrices[f"kernel|sigma{sigma}|{slide}|A|A"] = self_a * scale
        obj.kernel_matrices[f"kernel|sigma{sigma}|{slide}|B|B"] = self_b * scale

    scores = {
        # Rows are ordered as the A/B subsets of cell_types_sub: s1 then s2.
        "A": np.array(
            [[-1.0, 0.2], [0.4, 1.1], [1.3, -0.5],
             [0.7, -1.2], [-0.8, 0.3], [1.5, 0.9]]
        ),
        "B": np.array(
            [[0.8, -0.4], [-1.1, 1.2], [1.6, 0.5],
             [-0.6, 1.3], [1.4, -0.8], [0.2, 0.6]]
        ),
    }
    return obj, scores


def _slide_scores(scores, slide):
    index = slice(0, 3) if slide == "s1" else slice(3, 6)
    return scores["A"][index], scores["B"][index]


def test_multi_transfer_per_slide_results_use_slide_keyed_kernels():
    obj, scores = _make_multi_target()
    norm = get_transfer_norm_corr(obj, scores, sigma_choice=1.0, verbose=False)
    bidir = get_transfer_bidir_corr(
        obj, scores, sigma_choice=1.0, normalize_K="none",
        filter_kernel=False, verbose=False,
    )

    assert len(norm) == 4
    assert len(bidir) == 4
    assert set(norm["slideID"]) == {"s1", "s2"}
    assert set(bidir["slideID"]) == {"s1", "s2"}
    assert norm["normalized_correlation"].notna().all()
    assert bidir["bidir_correlation"].notna().all()

    for slide in obj.slide_list:
        K = _get_kernel_for_pair(obj.kernel_matrices, 1.0, "A", "B", slide)
        A, B = _slide_scores(scores, slide)
        expected = _compute_bidir_corrs_all_cc(A, B, K, "none")
        actual = bidir.loc[
            bidir["slideID"] == slide, "bidir_correlation"
        ].to_numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_multi_transfer_aggregate_matches_r_formulas():
    obj, scores = _make_multi_target()
    norm = get_transfer_norm_corr(
        obj, scores, sigma_choice=1.0,
        calculation_mode="aggregate", verbose=False,
    )
    bidir_per_slide = get_transfer_bidir_corr(
        obj, scores, sigma_choice=1.0, normalize_K="none",
        filter_kernel=False, calculation_mode="perSlide", verbose=False,
    )
    bidir_aggregate = get_transfer_bidir_corr(
        obj, scores, sigma_choice=1.0, normalize_K="none",
        filter_kernel=False, calculation_mode="aggregate", verbose=False,
    )

    assert list(norm.columns)[-1] == "aggregate_correlation"
    assert "slideID" not in norm
    assert len(norm) == 2

    expected_norm = []
    for cc in range(2):
        numerator = norm_a = norm_b = kernel_norm = 0.0
        for slide in obj.slide_list:
            K = _get_kernel_for_pair(
                obj.kernel_matrices, 1.0, "A", "B", slide
            )
            A, B = _slide_scores(scores, slide)
            numerator += float(A[:, cc] @ (K @ B[:, cc]))
            norm_a += float(np.sum(A[:, cc] ** 2))
            norm_b += float(np.sum(B[:, cc] ** 2))
            kernel_norm += _kernel_normalizer(
                obj.kernel_matrices, 1.0, "A", "B", slide
            )
        expected_norm.append(
            numerator / (np.sqrt(norm_a) * np.sqrt(norm_b) * (kernel_norm / 2))
        )
    np.testing.assert_allclose(
        norm["aggregate_correlation"], expected_norm, atol=1e-12
    )

    expected_bidir = (
        bidir_per_slide.groupby("CC_index")["bidir_correlation"].mean().to_numpy()
    )
    np.testing.assert_allclose(
        bidir_aggregate["aggregate_correlation"], expected_bidir, atol=1e-12
    )


@pytest.mark.parametrize(
    "function,value_column,kwargs",
    [
        (get_transfer_norm_corr, "normalized_correlation", {}),
        (
            get_transfer_bidir_corr,
            "bidir_correlation",
            {"normalize_K": "none", "filter_kernel": False},
        ),
    ],
)
def test_transfer_target_sigma_is_used_but_reference_sigma_labels_output(
    function, value_column, kwargs,
):
    obj, scores = _make_multi_target()
    expected = function(
        obj, scores, sigma_choice=2.0, verbose=False, **kwargs
    )
    with pytest.warns(UserWarning, match="different sigma"):
        transferred = function(
            obj,
            scores,
            sigma_choice=9.0,
            sigma_choice_tar=2.0,
            verbose=False,
            **kwargs,
        )

    assert set(transferred["sigma"]) == {9.0}
    np.testing.assert_allclose(
        transferred[value_column], expected[value_column], atol=1e-12
    )


def test_invalid_multi_transfer_calculation_mode_is_rejected():
    obj, scores = _make_multi_target()
    with pytest.raises(ValueError, match="calculation_mode"):
        get_transfer_norm_corr(
            obj, scores, sigma_choice=1.0,
            calculation_mode="pooled", verbose=False,
        )
