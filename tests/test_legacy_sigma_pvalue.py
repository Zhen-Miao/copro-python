"""Regression tests for DEFECT 2: legacy fixed-sigma p-value sigma-matching.

The legacy null (compute_normalized_correlation_permu) scores every draw at a
single sigma, but calculate_pvalue historically maxed the OBSERVED statistic
over every stored sigma. When the observed run used more than one sigma this
made observed = max-over-many-sigmas while each null = single sigma, biasing the
p-value downward (anti-conservative). calculate_pvalue now restricts the
observed maximum to the sigma set the nulls were scored at. The fair-sigma /
conditional nulls span every sigma per draw, so they are unaffected.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

import copro.permutation as permutation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_permutation_modern import _make_permutation_object  # noqa: E402


def _frame(sigma: float, value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sigma": [sigma],
            "cell_type_1": ["A"],
            "cell_type_2": ["B"],
            "CC_index": [1],
            "normalized_correlation": [value],
        }
    )


def _legacy_object():
    """Observed spans sigma 0.5 and 1.0; the null is legacy single-sigma (0.5)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        normalized_correlation={
            "sigma_0.5": _frame(0.5, 0.3),
            "sigma_1.0": _frame(1.0, 0.9),  # larger, but nulls never scored here
        },
        normalized_correlation_permu={
            "permu_1": _frame(0.5, 0.35),
            "permu_2": _frame(0.5, 0.40),
            "permu_3": _frame(0.5, 0.20),
            "permu_4": _frame(0.5, 0.32),
        },
    )


def test_legacy_single_sigma_null_restricts_observed_to_null_sigma():
    obj = _legacy_object()
    result = permutation.calculate_pvalue(obj, alternative="greater")

    # Observed is maxed only over the null's sigma (0.5): value 0.3, NOT the
    # larger observed value 0.9 that lives at sigma=1.0 where no null was scored.
    assert result["observed"] == pytest.approx(0.3)
    np.testing.assert_allclose(result["permu_values"], [0.35, 0.40, 0.20, 0.32])

    # Fair (sigma-matched) p-value: #{null >= 0.3} = 3 of 4.
    assert result["p_value"] == pytest.approx((1 + 3) / (4 + 1))  # 0.8

    # The old anti-conservative behavior (observed 0.9) would have given
    # #{null >= 0.9} = 0 -> p = 1/5 = 0.2. The fix is strictly more conservative.
    anti_conservative_p = (1 + 0) / (4 + 1)
    assert result["p_value"] > anti_conservative_p


def test_null_spanning_all_sigmas_leaves_observed_maximum_unchanged():
    from types import SimpleNamespace

    # A fair-sigma-style null: draws realize both sigmas, so the observed
    # maximum is taken over the full sigma set (0.5 and 1.0) and remains 0.9.
    obj = SimpleNamespace(
        normalized_correlation={
            "sigma_0.5": _frame(0.5, 0.3),
            "sigma_1.0": _frame(1.0, 0.9),
        },
        normalized_correlation_permu={
            "permu_1": _frame(0.5, 0.35),
            "permu_2": _frame(1.0, 0.80),
            "permu_3": _frame(0.5, 0.20),
            "permu_4": _frame(1.0, 0.85),
        },
    )
    result = permutation.calculate_pvalue(obj, alternative="greater")

    assert result["observed"] == pytest.approx(0.9)
    assert result["p_value"] == pytest.approx((1 + 0) / (4 + 1))  # 0.2


def test_fair_sigma_and_conditional_paths_unaffected():
    """The fair-sigma workflow's p-value still equals the conditional CC1 p_raw.

    Both re-maximize over every sigma per draw, so the sigma-matching in
    calculate_pvalue is a no-op for them.
    """
    from copy import deepcopy

    fair, _ = _make_permutation_object(n_cc=1)
    conditional = deepcopy(fair)

    permutation.run_skr_cca_permu_fair_sigma(
        fair, n_permu=6, permu_method="global", seed=7, verbose=False
    )
    permutation.run_skr_cca_permu_conditional(
        conditional, n_permu=6, permu_method="global", seed=7, verbose=False
    )

    # The fair-sigma null frames span more than one sigma, so the observed
    # maximum is unrestricted and equals the full max-over-sigma observed stat.
    null_sigmas = {
        float(frame.loc[0, "sigma"])
        for frame in fair.normalized_correlation_permu.values()
    }
    observed_full = max(
        fair.normalized_correlation[f"sigma_{s}"]
        .loc[lambda d: d["CC_index"] == 1, "normalized_correlation"]
        .max()
        for s in fair.sigma_values
    )
    result = permutation.calculate_pvalue(fair, alternative="greater")
    if len(null_sigmas) == len(fair.sigma_values):
        assert result["observed"] == pytest.approx(observed_full)

    stepdown = permutation.calculate_pvalue_stepdown(conditional)
    assert result["p_value"] == pytest.approx(stepdown.loc[0, "p_raw"])
