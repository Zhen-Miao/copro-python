"""Focused tests for modern permutation inference."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

import copro.permutation as permutation
from copro.optimization import _compute_Y_resi, _solve_two_type_svd


def _make_permutation_object(n_cc=2, seed=41):
    rng = np.random.RandomState(seed)
    n_cells = 10
    n_features = 4
    cts = ["A", "B"]
    scores = {
        "A": rng.normal(size=(n_cells, n_features)),
        "B": rng.normal(size=(n_cells, n_features)),
    }
    for ct in cts:
        scores[ct] -= scores[ct].mean(axis=0, keepdims=True)
        scores[ct] /= scores[ct].std(axis=0, ddof=1, keepdims=True)

    loc_a = rng.uniform([0, 0], [5, 8], size=(n_cells, 2))
    loc_b = rng.uniform([0, 0], [5, 8], size=(n_cells, 2))
    sigma_values = [0.5, 1.0]
    kernels = {}
    for sigma in sigma_values:
        for ct1, ct2, xy1, xy2 in (
            ("A", "A", loc_a, loc_a),
            ("B", "B", loc_b, loc_b),
            ("A", "B", loc_a, loc_b),
        ):
            distance = cdist(xy1, xy2)
            kernels[f"kernel|sigma{sigma}|{ct1}|{ct2}"] = np.exp(
                -(distance**2) / (2 * sigma**2)
            )

    obj = SimpleNamespace(
        cell_types_of_interest=cts,
        cell_types_sub=np.array(["A"] * n_cells + ["B"] * n_cells),
        location_data_sub=pd.DataFrame(
            np.vstack([loc_a, loc_b]), columns=["x", "y"]
        ),
        pca_global={
            ct: {"scores": scores[ct], "sdev": np.ones(n_features)} for ct in cts
        },
        kernel_matrices=kernels,
        sigma_values=sigma_values,
        scale_pcs=True,
        n_cc=n_cc,
        distance_scale_factor=1.0,
        skr_cca_out={},
        normalized_correlation={},
        cell_permu={},
        skr_cca_permu_out={},
        normalized_correlation_permu={},
    )

    for sigma in sigma_values:
        y_resi = _compute_Y_resi(scores, kernels, sigma, cts)
        weights = _solve_two_type_svd(y_resi, cts, n_cc=n_cc)
        obj.skr_cca_out[f"sigma_{sigma}"] = weights
        info = permutation._get_ncorr_kernel_info(kernels, sigma, cts)
        rows = []
        for axis in range(n_cc):
            w_axis = {ct: weights[ct][:, axis : axis + 1] for ct in cts}
            value = permutation._compute_ncorr_quick(
                scores,
                w_axis,
                kernels,
                sigma,
                cts,
                kernel_info=info,
                y_resi=y_resi,
            )
            rows.append(
                {
                    "sigma": sigma,
                    "cell_type_1": "A",
                    "cell_type_2": "B",
                    "CC_index": axis + 1,
                    "normalized_correlation": value,
                }
            )
        obj.normalized_correlation[f"sigma_{sigma}"] = pd.DataFrame(rows)

    obj.sigma_value_choice = max(
        sigma_values,
        key=lambda sigma: obj.normalized_correlation[f"sigma_{sigma}"].loc[
            0, "normalized_correlation"
        ],
    )
    return obj, scores


def _correlation_frame(values, sigma=0.5):
    return pd.DataFrame(
        {
            "sigma": sigma,
            "cell_type_1": ["A", "A"],
            "cell_type_2": ["B", "C"],
            "CC_index": [1, 1],
            "normalized_correlation": values,
        }
    )


def test_permutation_correlation_uses_whitened_frobenius_normalizer(monkeypatch):
    obj, scores = _make_permutation_object(n_cc=1)
    sigma = obj.sigma_value_choice
    weights = obj.skr_cca_out[f"sigma_{sigma}"]
    obj.skr_cca_permu_out = {"permu_1": weights}
    obj.cell_permu = {
        ct: np.arange(len(scores[ct]), dtype=int)[:, None]
        for ct in obj.cell_types_of_interest
    }
    expected = permutation._compute_ncorr_quick(
        scores,
        weights,
        obj.kernel_matrices,
        sigma,
        obj.cell_types_of_interest,
    )

    calls = []
    original = permutation._kernel_normalizer

    def recording_normalizer(*args, **kwargs):
        calls.append(args[1:4])
        return original(*args, **kwargs)

    monkeypatch.setattr(permutation, "_kernel_normalizer", recording_normalizer)
    permutation.compute_normalized_correlation_permu(obj, verbose=False)

    actual = obj.normalized_correlation_permu["permu_1"].loc[
        0, "normalized_correlation"
    ]
    assert actual == pytest.approx(expected, abs=1e-12)
    assert calls == [(sigma, "A", "B")]


def test_calculate_pvalue_is_phipson_smyth_and_maxes_consistently():
    obj = SimpleNamespace(
        normalized_correlation={
            "sigma_0.5": _correlation_frame([2.0, 3.0], sigma=0.5),
            "sigma_1.0": _correlation_frame([4.0, 1.0], sigma=1.0),
        },
        normalized_correlation_permu={
            "permu_1": _correlation_frame([0.0, 5.0]),
            "permu_2": _correlation_frame([2.0, 3.0]),
        },
    )

    greater = permutation.calculate_pvalue(obj, alternative="greater")
    less = permutation.calculate_pvalue(obj, alternative="less")
    two_sided = permutation.calculate_pvalue(obj, alternative="two_sided")

    np.testing.assert_array_equal(greater["permu_values"], [5.0, 3.0])
    assert greater["observed"] == 4.0
    assert greater["p_value"] == pytest.approx(2 / 3)
    assert less["p_value"] == pytest.approx(2 / 3)
    assert two_sided["p_value"] == 1.0
    assert greater["mc_floor"] == pytest.approx(1 / 3)
    assert greater["pair_aggregation"] == "max"

    selected = permutation.calculate_pvalue(
        obj, cell_type_1="A", cell_type_2="B"
    )
    assert selected["observed"] == 4.0
    np.testing.assert_array_equal(selected["permu_values"], [0.0, 2.0])
    assert selected["pair_aggregation"] == "selected"


def test_sigma_aware_bins_use_stored_distance_scale_factor():
    obj = SimpleNamespace(
        location_data_sub=pd.DataFrame(
            {"x": [0.0, 10.0], "y": [0.0, 20.0]}
        ),
        distance_scale_factor=2.0,
    )
    bins = permutation._sigma_aware_bins(obj, sigma=2.0, verbose=False)
    assert bins == {"num_bins_x": 5, "num_bins_y": 10, "scale_factor": 2.0}

    obj.distance_scale_factor = None
    with pytest.warns(UserWarning, match="falling back"):
        fallback = permutation._sigma_aware_bins(obj, sigma=2.0, verbose=False)
    assert fallback["num_bins_x"] == fallback["num_bins_y"] == 10
    assert np.isnan(fallback["scale_factor"])


def test_bin_permutation_prepares_each_permuted_cell_type_once(monkeypatch):
    obj, _ = _make_permutation_object(n_cc=1)
    calls = 0
    original = permutation._prepare_spatial_resampling

    def recording_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        permutation, "_prepare_spatial_resampling", recording_prepare
    )
    draws = permutation._get_cell_permu(
        obj,
        permu_method="bin",
        n_permu=5,
        cts=obj.cell_types_of_interest,
        permu_which="second_only",
        num_bins_x=2,
        num_bins_y=2,
        match_quantile=False,
        rng=np.random.RandomState(9),
    )
    assert calls == 1
    np.testing.assert_array_equal(draws["A"][:, 0], np.arange(10))
    assert draws["B"].shape == (10, 5)
    assert np.all((draws["B"] >= 0) & (draws["B"] < 10))


def test_legacy_fixed_sigma_api_remains_runnable():
    obj, _ = _make_permutation_object(n_cc=2)
    with pytest.warns(UserWarning, match="n_permu"):
        permutation.run_skr_cca_permu(
            obj,
            n_permu=3,
            permu_method="global",
            seed=12,
            verbose=False,
        )
    assert list(obj.skr_cca_permu_out) == ["permu_1", "permu_2", "permu_3"]
    permutation.compute_normalized_correlation_permu(obj, verbose=False)
    result = permutation.calculate_pvalue(obj, cc_index=1)
    assert 0 < result["p_value"] <= 1
    assert result["mc_floor"] == 0.25
    assert obj.n_permu == 3


def test_conditional_first_axis_matches_fair_sigma_with_same_draws(monkeypatch):
    fair, _ = _make_permutation_object(n_cc=1)
    conditional = deepcopy(fair)
    calls = 0
    original = permutation._kernel_normalizer

    def recording_normalizer(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(permutation, "_kernel_normalizer", recording_normalizer)
    permutation.run_skr_cca_permu_fair_sigma(
        fair,
        n_permu=5,
        permu_method="global",
        seed=202,
        verbose=False,
    )
    # The expensive normalizer is cached once per sigma, not per draw.
    assert calls == len(fair.sigma_values)

    monkeypatch.setattr(permutation, "_kernel_normalizer", original)
    permutation.run_skr_cca_permu_conditional(
        conditional,
        n_permu=5,
        permu_method="global",
        seed=202,
        verbose=False,
    )
    fair_null = np.array(
        [
            frame.loc[0, "normalized_correlation"]
            for frame in fair.normalized_correlation_permu.values()
        ]
    )
    conditional_null = conditional.conditional_permu["perm_stats"][:, 0]
    np.testing.assert_allclose(fair_null, conditional_null, atol=1e-12)
    assert permutation.calculate_pvalue(fair)["p_value"] == pytest.approx(
        permutation.calculate_pvalue_stepdown(conditional).loc[0, "p_raw"]
    )


def test_conditional_axes_reproduce_deflation_and_stepdown_is_monotone():
    obj, scores = _make_permutation_object(n_cc=2)
    sigma = obj.sigma_value_choice
    weights = obj.skr_cca_out[f"sigma_{sigma}"]
    y_resi = _compute_Y_resi(
        scores, obj.kernel_matrices, sigma, obj.cell_types_of_interest
    )
    fit = permutation._fit_conditional_axis(
        scores,
        obj.kernel_matrices,
        sigma,
        obj.cell_types_of_interest,
        w_lower=weights,
        k_minus_1=1,
        y_resi=y_resi,
    )
    expected = obj.normalized_correlation[f"sigma_{sigma}"].loc[
        lambda frame: frame["CC_index"] == 2, "normalized_correlation"
    ].iloc[0]
    assert fit["ncorr"] == pytest.approx(expected, abs=1e-10)

    permutation.run_skr_cca_permu_conditional(
        obj,
        n_permu=6,
        permu_method="global",
        alpha=0.2,
        seed=11,
        verbose=False,
    )
    table = permutation.calculate_pvalue_stepdown(obj)
    assert np.all(np.diff(table["p_stepdown"]) >= -1e-12)
    assert np.all(table["p_stepdown"] >= table["p_raw"] - 1e-12)
    assert np.all(table["p_raw"] >= table["mc_floor"] - 1e-12)
    assert np.all(table["p_raw"] > 0)
    assert table.attrs["n_significant_axes"] == int(table["significant"].sum())
    false_axes = np.flatnonzero(~table["significant"].to_numpy())
    if len(false_axes):
        assert not table["significant"].iloc[false_axes[0] :].any()


def test_stepdown_reader_requires_conditional_results():
    with pytest.raises(ValueError, match="conditional"):
        permutation.calculate_pvalue_stepdown(SimpleNamespace())
