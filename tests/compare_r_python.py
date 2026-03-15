"""Side-by-side comparison of R reference vs Python CoPro results.

Usage:
    python tests/compare_r_python.py [alt|null]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
import copro as cp

DATA_DIR = Path(__file__).parent.parent.parent / "CoPro scripts" / "2026_spatial_simulation"
REF_BASE = Path(__file__).parent / "r_reference"
SIGMA_VALUES = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5]
N_PCA = 25
CELL_TYPES = ["A", "B"]


def run_python_pipeline(expr_file, meta_file):
    expr_df = pd.read_parquet(expr_file)
    meta = pd.read_parquet(meta_file)

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


def compare(dataset: str = "alt"):
    if dataset == "alt":
        prefix = "alternative_round1"
        ref_dir = REF_BASE / "alt_round1"
        label = "Alternative (signal present)"
    else:
        prefix = "null_round1"
        ref_dir = REF_BASE / "null_round1"
        label = "Null (no signal)"

    expr_file = DATA_DIR / f"{prefix}_expression.parquet"
    meta_file = DATA_DIR / f"{prefix}_metadata.parquet"

    print(f"\n{'='*60}")
    print(f"Dataset: {label}")
    print(f"{'='*60}")

    print("Running Python pipeline...")
    obj = run_python_pipeline(expr_file, meta_file)

    print(f"\n--- Normalized Correlation (Python) ---")
    for sigma_name in sorted(obj.normalized_correlation):
        df = obj.normalized_correlation[sigma_name]
        if df is not None and len(df) > 0:
            cc1 = df[df["CC_index"] == 1]["normalized_correlation"].values
            print(f"  {sigma_name}: CC1 = {cc1}")

    if ref_dir.exists():
        norm_corr_path = ref_dir / "normalized_correlation.csv"
        if norm_corr_path.exists():
            r_nc = pd.read_csv(norm_corr_path)
            print(f"\n--- Normalized Correlation (R reference) ---")
            for sigma in SIGMA_VALUES:
                r_rows = r_nc[(r_nc["CC_index"] == 1) &
                               (np.isclose(r_nc["sigmaValues"], sigma, atol=1e-6))]
                if len(r_rows) > 0:
                    print(f"  sigma_{sigma}: CC1 = {r_rows['normalizedCorrelation'].values}")

        print(f"\n--- Cell Score Pearson Correlations (sign-flip tolerant) ---")
        sigma_choice = obj.sigma_value_choice
        print(f"  Best sigma (Python): {sigma_choice}")

        sigma_str = str(sigma_choice).replace(".", "p")
        for ct in CELL_TYPES:
            r_cs_path = ref_dir / f"cell_scores_sigma_{sigma_str}_{ct}.csv"
            py_key = f"cellScores|sigma{sigma_choice}|{ct}"
            if not r_cs_path.exists():
                print(f"  {ct}: R reference not found")
                continue
            if py_key not in obj.cell_scores:
                print(f"  {ct}: Python result not found for {py_key}")
                continue

            r_cs = pd.read_csv(r_cs_path, index_col=0).values[:, 0]
            py_cs = obj.cell_scores[py_key][:, 0]
            n = min(len(r_cs), len(py_cs))
            corr, pval = pearsonr(py_cs[:n], r_cs[:n])
            flip = "⚠️ SIGN FLIP" if corr < 0 else "✓"
            print(f"  {ct}: Pearson r = {corr:.4f} (|r| = {abs(corr):.4f}) {flip}")
    else:
        print(f"\n[R reference not available at {ref_dir}]")
        print("Run: Rscript tests/run_r_reference.R")

    print(f"\n  sigma_value_choice (Python) = {obj.sigma_value_choice}")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "alt"
    if dataset not in ("alt", "null"):
        print("Usage: python compare_r_python.py [alt|null]")
        sys.exit(1)
    compare(dataset)
