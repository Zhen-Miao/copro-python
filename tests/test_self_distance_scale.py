"""Regression tests for DEFECT 1: compute_self_distance must not clobber the
CROSS-type distance_scale_factor set by compute_distance.

R parity: ``computeSelfDistance`` (compute_self_distance_kernel.R) never writes
``@distanceScaleFactor``; the cross-type factor set by ``computeDistance``
(11_compute_distance.R) is what ``.recoverDistanceScaleFactor`` /
``.sigmaAwareBins`` (C_resampling_function.R) read to size permutation patches
in CROSS-normalized units.
"""

import numpy as np
import pandas as pd
import pytest

import copro.permutation as permutation
from copro.core import CoProMulti, CoProSingle, subset_data
from copro.distance import compute_distance, compute_self_distance


def _two_cluster_single(n_per_type: int = 40, seed: int = 0) -> CoProSingle:
    """Two tightly packed, far-apart clusters.

    Cross-type (A-B) distances are large -> small cross scale factor.
    Within-type (A-A / B-B) distances are tiny -> large self scale factor.
    The two factors therefore differ by orders of magnitude, so a regression
    that re-clobbers distance_scale_factor with the self value is caught.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform([0.0, 0.0], [1.0, 1.0], size=(n_per_type, 2))
    b = rng.uniform([100.0, 100.0], [101.0, 101.0], size=(n_per_type, 2))
    coords = np.vstack([a, b])
    n = coords.shape[0]
    obj = CoProSingle(
        normalized_data=rng.normal(size=(n, 5)),
        location_data=pd.DataFrame(coords, columns=["x", "y"]),
        meta_data=pd.DataFrame(index=np.arange(n)),
        cell_types=np.array(["A"] * n_per_type + ["B"] * n_per_type),
    )
    return subset_data(obj, ["A", "B"])


def _two_cluster_multi(n_per: int = 12, seed: int = 3) -> CoProMulti:
    rng = np.random.default_rng(seed)
    coords, cell_types, slides = [], [], []
    for slide in ["s1", "s2"]:
        coords.append(rng.uniform([0.0, 0.0], [1.0, 1.0], size=(n_per, 2)))
        cell_types.extend(["A"] * n_per)
        slides.extend([slide] * n_per)
        coords.append(rng.uniform([100.0, 100.0], [101.0, 101.0], size=(n_per, 2)))
        cell_types.extend(["B"] * n_per)
        slides.extend([slide] * n_per)
    location = np.vstack(coords)
    obj = CoProMulti(
        normalized_data=rng.normal(size=(len(location), 4)),
        location_data=pd.DataFrame(location, columns=["x", "y"]),
        meta_data=pd.DataFrame({"slideID": slides}),
        cell_types=np.asarray(cell_types),
        slide_list=["s1", "s2"],
    )
    return subset_data(obj, ["A", "B"])


def test_self_distance_preserves_cross_scale_factor_single():
    obj = _two_cluster_single()

    compute_distance(obj, normalize=True)
    cross = obj.distance_scale_factor
    assert cross is not None
    assert np.isfinite(cross) and cross > 0

    compute_self_distance(obj, normalize=True, verbose=False)

    # The CROSS factor must survive compute_self_distance unchanged.
    assert obj.distance_scale_factor == pytest.approx(cross)
    # The WITHIN factor lives on its own attribute...
    self_factor = obj.self_distance_scale_factor
    assert self_factor is not None
    assert np.isfinite(self_factor) and self_factor > 0
    # ...and is materially different (tight clusters vs. far-apart clusters).
    assert self_factor != pytest.approx(cross)
    assert self_factor > 10.0 * cross

    # Permutation binning must recover the CROSS factor, not the self factor.
    assert permutation._recover_distance_scale_factor(obj) == pytest.approx(cross)
    bins = permutation._sigma_aware_bins(obj, sigma=0.001, verbose=False)
    assert bins["scale_factor"] == pytest.approx(cross)


def test_self_distance_preserves_cross_scale_factor_multi():
    obj = _two_cluster_multi()

    compute_distance(obj, normalize=True)
    cross = obj.distance_scale_factor
    assert cross is not None and np.isfinite(cross) and cross > 0

    compute_self_distance(obj, normalize=True, verbose=False)

    assert obj.distance_scale_factor == pytest.approx(cross)
    assert obj.self_distance_scale_factor is not None
    assert obj.self_distance_scale_factor != pytest.approx(cross)


def test_self_distance_standalone_still_populates_scale_factor():
    # compute_distance was never called: distance_scale_factor is unset (None),
    # so compute_self_distance falls back to recording the self factor there
    # (there is no cross factor to protect). This keeps the standalone self
    # path consistent with the sparse self-kernel path in kernel.py.
    obj = _two_cluster_single()
    assert obj.distance_scale_factor is None

    compute_self_distance(obj, normalize=True, verbose=False)

    self_factor = obj.self_distance_scale_factor
    assert self_factor is not None and np.isfinite(self_factor) and self_factor > 0
    assert obj.distance_scale_factor == pytest.approx(self_factor)


def test_self_distance_does_not_overwrite_unnormalized_cross_factor():
    # compute_distance(normalize=False) records a cross factor of 1.0 (raw
    # units). compute_self_distance(normalize=True) must not overwrite it, even
    # though the self factor differs, because the cross permutation correlation
    # is defined in those raw units.
    obj = _two_cluster_single()
    compute_distance(obj, normalize=False)
    assert obj.distance_scale_factor == 1.0

    compute_self_distance(obj, normalize=True, verbose=False)

    assert obj.distance_scale_factor == 1.0
    assert obj.self_distance_scale_factor != pytest.approx(1.0)
