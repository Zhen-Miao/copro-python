"""Focused validation tests for multi-slide SkrCCA inputs."""

import numpy as np
import pandas as pd
import pytest

from copro import CoProMulti, run_skr_cca


def _multi_with_missing_pca_type(missing_slide: str) -> CoProMulti:
    obj = CoProMulti(
        normalized_data=np.zeros((8, 2)),
        location_data=pd.DataFrame(
            {"x": np.arange(8, dtype=float), "y": np.arange(8, dtype=float)}
        ),
        meta_data=pd.DataFrame({"slideID": ["s1"] * 4 + ["s2"] * 4}),
        cell_types=np.array(["A", "A", "B", "B"] * 2),
        slide_list=["s1", "s2"],
        cell_types_of_interest=["A", "B"],
    )
    obj.pca_global = {
        "A": {"sdev": np.ones(2)},
        "B": {"sdev": np.ones(2)},
    }
    scores = np.eye(2)
    obj.pca_results = {
        "s1": {"A": scores.copy(), "B": scores.copy()},
        "s2": {"A": scores.copy(), "B": scores.copy()},
    }
    del obj.pca_results[missing_slide]["B"]
    obj.sigma_values = [1.0]
    obj.kernel_matrices = {
        f"kernel|sigma1.0|{slide}|A|B": np.eye(2)
        for slide in obj.slide_list
    }
    return obj


def test_multi_skrcca_rejects_type_missing_from_first_slide():
    obj = _multi_with_missing_pca_type("s1")

    with pytest.raises(
        ValueError,
        match=r"Missing PCA data for slide 's1', cell type 'B'",
    ):
        run_skr_cca(obj, n_cc=1, scale_pcs=True, verbose=False)

    assert obj.skr_cca_out == {}


def test_multi_skrcca_rejects_type_missing_from_later_slide_without_scaling():
    obj = _multi_with_missing_pca_type("s2")

    with pytest.raises(
        ValueError,
        match=r"Missing PCA data for slide 's2', cell type 'B'",
    ):
        run_skr_cca(obj, n_cc=1, scale_pcs=False, verbose=False)

    assert obj.skr_cca_out == {}
