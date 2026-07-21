"""Test CoPro Python pipeline on alternative_round1 dataset (signal present)."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import copro as cp

DATA_DIR = Path(__file__).parent.parent.parent / "CoPro scripts" / "2026_spatial_simulation"
EXPR_FILE = DATA_DIR / "alternative_round1_expression.parquet"
META_FILE = DATA_DIR / "alternative_round1_metadata.parquet"
REF_DIR = Path(__file__).parent / "r_reference" / "alt_round1"

pytestmark = pytest.mark.skipif(
    not (EXPR_FILE.exists() and META_FILE.exists()),
    reason="optional spatial-simulation integration dataset is unavailable",
)

SIGMA_VALUES = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5]
CELL_TYPES = ["A", "B"]
N_PCA = 25


@pytest.fixture(scope="module")
def copro_obj():
    """Run full Python pipeline once per module."""
    expr_df = pd.read_parquet(EXPR_FILE)
    meta = pd.read_parquet(META_FILE)

    # Drop non-gene columns (cell_id etc.)
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


class TestAlternativeRound1:
    def test_data_loaded(self, copro_obj):
        assert copro_obj.normalized_data_sub is not None
        assert copro_obj.normalized_data_sub.shape[0] > 0

    def test_pca_computed(self, copro_obj):
        for ct in CELL_TYPES:
            assert ct in copro_obj.pca_global
            pca = copro_obj.pca_global[ct]
            assert pca["scores"].shape[1] == N_PCA
            assert len(pca["sdev"]) == N_PCA

    def test_kernel_matrices_exist(self, copro_obj):
        assert len(copro_obj.kernel_matrices) > 0

    def test_cca_results_exist(self, copro_obj):
        assert len(copro_obj.skr_cca_out) > 0
        for sigma_name, w_dict in copro_obj.skr_cca_out.items():
            if w_dict is not None:
                for ct in CELL_TYPES:
                    assert ct in w_dict
                    assert w_dict[ct].shape[1] == 2  # n_cc=2

    def test_signal_detected(self, copro_obj):
        """Alternative dataset should have norm_corr > null-dataset baseline for CC1.

        The spectral-norm-normalized correlation is typically small in absolute terms
        (~0.01–0.05 for these simulations). What matters is that it's meaningfully
        higher than the null dataset's best (~0.017). We use 0.025 as a conservative
        threshold that separates signal from noise at sigma=0.05.
        """
        max_cc1 = 0.0
        for sigma_name, df in copro_obj.normalized_correlation.items():
            if df is not None and len(df) > 0:
                cc1_max = df[df["CC_index"] == 1]["normalized_correlation"].max()
                max_cc1 = max(max_cc1, cc1_max)
        assert max_cc1 > 0.025, (
            f"Alternative dataset max CC1 norm_corr={max_cc1:.4f} is too low "
            "(expected > 0.025, comparable to null ~0.017)"
        )

    def test_cell_scores_exist(self, copro_obj):
        for sigma in copro_obj.sigma_values:
            for ct in CELL_TYPES:
                key = f"cellScores|sigma{sigma}|{ct}"
                assert key in copro_obj.cell_scores, f"Missing key: {key}"
                scores = copro_obj.cell_scores[key]
                assert scores.shape[1] == 2

    def test_gene_scores_exist(self, copro_obj):
        for sigma in copro_obj.sigma_values:
            for ct in CELL_TYPES:
                key = f"geneScores|sigma{sigma}|{ct}"
                assert key in copro_obj.gene_scores, f"Missing key: {key}"

    def test_vs_r_reference(self, copro_obj):
        """Compare to R reference if available (sign-flip tolerant)."""
        if not REF_DIR.exists():
            pytest.skip("R reference results not available. Run tests/run_r_reference.R first.")

        from scipy.stats import pearsonr

        norm_corr_path = REF_DIR / "normalized_correlation.csv"
        if not norm_corr_path.exists():
            pytest.skip("R normalized_correlation.csv not found.")

        r_nc = pd.read_csv(norm_corr_path)
        # Compare best sigma's CC1 correlation
        sigma_choice = copro_obj.sigma_value_choice
        sigma_name = f"sigma_{sigma_choice}"
        py_df = copro_obj.normalized_correlation.get(sigma_name)
        if py_df is None:
            pytest.skip(f"No Python result for sigma={sigma_choice}")

        py_cc1 = py_df[py_df["CC_index"] == 1]["normalized_correlation"].mean()

        r_cc1_rows = r_nc[(r_nc["CC_index"] == 1) &
                           (np.isclose(r_nc["sigmaValues"], sigma_choice, atol=1e-6))]
        if len(r_cc1_rows) == 0:
            pytest.skip("No matching R CC1 result for chosen sigma.")

        r_cc1 = r_cc1_rows["normalizedCorrelation"].mean()
        assert abs(py_cc1 - r_cc1) < 0.01, (
            f"Normalized correlation mismatch: Python={py_cc1:.4f}, R={r_cc1:.4f}"
        )

        # Also check all sigmas
        for sigma in SIGMA_VALUES:
            sigma_name = f"sigma_{sigma}"
            py_df_s = copro_obj.normalized_correlation.get(sigma_name)
            if py_df_s is None:
                continue
            py_s = py_df_s[py_df_s["CC_index"] == 1]["normalized_correlation"].mean()
            r_rows_s = r_nc[(r_nc["CC_index"] == 1) & (np.isclose(r_nc["sigmaValues"], sigma, atol=1e-6))]
            if len(r_rows_s) == 0:
                continue
            r_s = r_rows_s["normalizedCorrelation"].mean()
            assert abs(py_s - r_s) < 0.01, (
                f"sigma={sigma}: Python CC1={py_s:.4f}, R CC1={r_s:.4f}, diff={abs(py_s-r_s):.4f}"
            )

        # Cell score Pearson correlation (sign-flip tolerant)
        for ct in CELL_TYPES:
            r_cs_path = REF_DIR / f"cell_scores_sigma_{str(sigma_choice).replace('.', 'p')}_{ct}.csv"
            py_key = f"cellScores|sigma{sigma_choice}|{ct}"
            if not r_cs_path.exists() or py_key not in copro_obj.cell_scores:
                continue
            r_cs = pd.read_csv(r_cs_path, index_col=0).values[:, 0]
            py_cs = copro_obj.cell_scores[py_key][:, 0]
            corr, _ = pearsonr(py_cs, r_cs[:len(py_cs)])
            assert abs(corr) > 0.90, (
                f"Cell score Pearson |r|={abs(corr):.4f} < 0.90 for {ct}"
            )
