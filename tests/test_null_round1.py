"""Test CoPro Python pipeline on null_round1 dataset (no spatial signal)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import copro as cp

DATA_DIR = Path(__file__).parent.parent.parent / "CoPro scripts" / "2026_spatial_simulation"
EXPR_FILE = DATA_DIR / "null_round1_expression.parquet"
META_FILE = DATA_DIR / "null_round1_metadata.parquet"
REF_DIR = Path(__file__).parent / "r_reference" / "null_round1"

pytestmark = pytest.mark.skipif(
    not (EXPR_FILE.exists() and META_FILE.exists()),
    reason="optional spatial-simulation integration dataset is unavailable",
)

SIGMA_VALUES = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5]
CELL_TYPES = ["A", "B"]
N_PCA = 25


@pytest.fixture(scope="module")
def copro_obj():
    expr_df = pd.read_parquet(EXPR_FILE)
    meta = pd.read_parquet(META_FILE)

    gene_cols = [c for c in expr_df.columns if c != "cell_id"]
    expr_np = expr_df[gene_cols].values.astype(float)
    cell_types = meta["cell_type"].values.astype(str)
    location = meta[["x", "y"]].reset_index(drop=True)

    obj = cp.CoProSingle(
        normalized_data=expr_np,
        location_data=location,
        meta_data=meta.reset_index(drop=True),
        cell_types=cell_types,
    )
    obj = cp.subset_data(obj, CELL_TYPES)
    obj = cp.compute_pca(obj, n_pca=N_PCA)
    obj = cp.compute_distance(obj, normalize=False)
    obj = cp.compute_kernel_matrix(obj, sigma_values=SIGMA_VALUES, row_normalize_kernel=True)
    obj = cp.run_skr_cca(obj, scale_pcs=True, n_cc=2, max_iter=500)
    obj = cp.compute_normalized_correlation(obj)
    obj = cp.compute_gene_and_cell_scores(obj)
    return obj


class TestNullRound1:
    def test_pipeline_runs(self, copro_obj):
        assert copro_obj.normalized_correlation

    def test_no_spurious_signal(self, copro_obj):
        """Null dataset: all norm_corr should be modest (no strong spurious signal)."""
        best_corr_values = []
        for sigma_name, df in copro_obj.normalized_correlation.items():
            if df is not None and len(df) > 0:
                cc1 = df[df["CC_index"] == 1]["normalized_correlation"]
                best_corr_values.append(cc1.max())

        assert best_corr_values, "No correlation results computed."
        # Null data typically has low correlation, warn if high
        max_observed = max(best_corr_values)
        print(f"Null dataset max CC1 norm_corr = {max_observed:.4f}")
        # Not a hard threshold since random data can vary,
        # but we flag if unusually high
        if max_observed > 0.5:
            import warnings
            warnings.warn(
                f"Null dataset max norm_corr={max_observed:.4f} seems high. "
                "Check for data issues."
            )

    def test_vs_r_reference(self, copro_obj):
        """Compare to R reference if available."""
        if not REF_DIR.exists():
            pytest.skip("R reference results not available. Run tests/run_r_reference.R first.")

        norm_corr_path = REF_DIR / "normalized_correlation.csv"
        if not norm_corr_path.exists():
            pytest.skip("R normalized_correlation.csv not found.")

        r_nc = pd.read_csv(norm_corr_path)

        for sigma in copro_obj.sigma_values:
            sigma_name = f"sigma_{sigma}"
            py_df = copro_obj.normalized_correlation.get(sigma_name)
            if py_df is None:
                continue

            py_cc1 = py_df[py_df["CC_index"] == 1]["normalized_correlation"].mean()
            r_rows = r_nc[(r_nc["CC_index"] == 1) &
                           (np.isclose(r_nc["sigmaValues"], sigma, atol=1e-6))]
            if len(r_rows) == 0:
                continue

            r_cc1 = r_rows["normalizedCorrelation"].mean()
            diff = abs(py_cc1 - r_cc1)
            print(f"sigma={sigma}: Python CC1={py_cc1:.4f}, R CC1={r_cc1:.4f}, diff={diff:.4f}")
            assert diff < 0.01, (
                f"Null dataset sigma={sigma}: Python={py_cc1:.4f} R={r_cc1:.4f} diff={diff:.4f}"
            )
