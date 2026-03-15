"""
Compare Python CoPro tutorial pipelines against R reference outputs.

Runs the organoid (one cell type) and MERFISH (two cell types) pipelines
in Python, then checks concordance with R reference results exported by
run_r_reference_organoid.R and run_r_reference_merfish.R.

Usage:
    python tests/compare_tutorials.py
    (run from repo root: CoPro Claude Code/)
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
import copro as cp

REPO_ROOT = Path(__file__).parent.parent.parent  # CoPro Claude Code/
REF_BASE  = Path(__file__).parent / "r_reference"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _corr_sign_flip(py_vec: np.ndarray, r_vec: np.ndarray) -> float:
    """Pearson |r| (sign-flip allowed)."""
    py_vec = np.asarray(py_vec).ravel()
    r_vec  = np.asarray(r_vec).ravel()
    if py_vec.std() < 1e-10 or r_vec.std() < 1e-10:
        return float("nan")
    r, _ = pearsonr(py_vec, r_vec)
    return abs(r)


def _check(label: str, val: float, threshold: float, *, verbose: bool = True) -> bool:
    ok = val >= threshold
    status = "PASS" if ok else "FAIL"
    if verbose:
        print(f"  [{status}] {label}: {val:.6f}  (threshold ≥ {threshold})")
    return ok


def _compare_nc(py_nc_df: pd.DataFrame, r_nc_df: pd.DataFrame,
                sigma_col: str = "sigma") -> None:
    """Print side-by-side normalized correlation comparison."""
    py = py_nc_df.sort_values([sigma_col, "CC_index"]).reset_index(drop=True)
    r  = r_nc_df.rename(columns={
        "sigmaValues": "sigma",
        "normalizedCorrelation": "normalized_correlation",
        "CC_index": "CC_index",
    }).sort_values(["sigma", "CC_index"]).reset_index(drop=True)

    # Align on sigma × CC
    print(f"  {'sigma':>8}  {'CC':>3}  {'R':>10}  {'Python':>10}  {'|diff|':>10}")
    for _, row in r.iterrows():
        s   = row["sigma"]
        cc  = int(row["CC_index"])
        r_val = row["normalized_correlation"]
        py_match = py[(py["sigma"] == s) & (py["CC_index"] == cc)]
        if len(py_match) == 0:
            print(f"  {s:>8}  {cc:>3}  {r_val:>10.5f}  {'N/A':>10}")
            continue
        py_val  = py_match["normalized_correlation"].values[0]
        diff    = abs(r_val - py_val)
        print(f"  {s:>8}  {cc:>3}  {r_val:>10.5f}  {py_val:>10.5f}  {diff:>10.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Tutorial 1: Organoid (one cell type)
# ─────────────────────────────────────────────────────────────────────────────

def run_organoid():
    print("\n" + "=" * 70)
    print("TUTORIAL 1: Organoid (one cell type — Epithelial)")
    print("=" * 70)

    DATA_DIR  = Path.home() / "Library/CloudStorage/Dropbox/DIALOGUE_plus project/Raj_lab_data/72hr/roi1/output"
    EXPR_FILE = DATA_DIR / "cell_by_gene/cell_by_gene.csv"
    META_FILE = DATA_DIR / "attributes/cell_attributes.csv"
    REF_DIR   = REF_BASE / "organoid"

    N_PCA        = 30
    SIGMA_VALUES = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    N_CC         = 4
    CELL_TYPE    = "Epithelial"

    # ── Load and preprocess ───────────────────────────────────────────────────
    print("\nLoading and preprocessing organoid data...")
    expr_df = pd.read_csv(EXPR_FILE, index_col=0)
    meta    = pd.read_csv(META_FILE)
    meta.index = [f"cell_{l}" for l in meta["label"]]
    expr_df.index = meta.index

    # Cap at 95th percentile
    upper = expr_df.quantile(0.95, axis=0)
    expr_capped = expr_df.clip(upper=upper, axis=1)

    # Log1p normalization
    expr_norm = np.log1p(expr_capped.values.astype(float))
    print(f"  Cells: {expr_norm.shape[0]}  Genes: {expr_norm.shape[1]}")

    # Spatial coordinates (scaled by 5000)
    location = pd.DataFrame({"x": meta["center_x"].values / 5000,
                              "y": meta["center_y"].values / 5000})
    cell_types = np.array([CELL_TYPE] * len(meta))

    # ── Python pipeline ───────────────────────────────────────────────────────
    print("Running Python CoPro pipeline...")
    obj = cp.CoProSingle(
        normalized_data=expr_norm,
        location_data=location,
        meta_data=meta.reset_index(drop=True),
        cell_types=cell_types,
    )
    obj = cp.subset_data(obj, [CELL_TYPE])
    obj = cp.compute_pca(obj, n_pca=N_PCA)
    obj = cp.compute_distance(obj, normalize=False)
    obj = cp.compute_kernel_matrix(obj, sigma_values=SIGMA_VALUES,
                                   upper_quantile=0.85, lower_limit=5e-7)
    obj = cp.run_skr_cca(obj, scale_pcs=True, n_cc=N_CC, max_iter=500)
    obj = cp.compute_normalized_correlation(obj)
    obj = cp.compute_gene_and_cell_scores(obj)
    print(f"  Sigma choice (Python): {obj.sigma_value_choice}")

    # ── R reference ───────────────────────────────────────────────────────────
    r_sigma_choice = float(open(REF_DIR / "sigma_choice.txt").read().strip())
    print(f"  Sigma choice (R):      {r_sigma_choice}")

    passes = []

    # ── PCA comparison ────────────────────────────────────────────────────────
    print("\n── PCA comparison ──")
    r_scores = pd.read_csv(REF_DIR / "pca_scores.csv", index_col=0).values
    py_scores = obj.pca_global[CELL_TYPE]["scores"]
    r_sdev    = pd.read_csv(REF_DIR / "pca_sdev.csv", index_col=0)["sdev"].values
    py_sdev   = obj.pca_global[CELL_TYPE]["sdev"]

    for pc in [0, 1, 2, 4, 9, 19, 29]:
        if pc >= py_scores.shape[1]: continue
        r_corr = _corr_sign_flip(py_scores[:, pc], r_scores[:, pc])
        passes.append(_check(f"PC{pc+1} score |r|", r_corr, 0.99))

    sdev_rel_err = np.abs(py_sdev - r_sdev[:len(py_sdev)]) / (r_sdev[:len(py_sdev)] + 1e-12)
    passes.append(_check("sdev max rel err < 0.01", float(1 - sdev_rel_err.max()), 0.99))

    # ── Normalized correlation comparison ─────────────────────────────────────
    print("\n── Normalized correlation ──")
    r_nc = pd.read_csv(REF_DIR / "normalized_correlation.csv")
    nc_frames = [df for df in obj.normalized_correlation.values() if df is not None]
    py_nc = pd.concat(nc_frames, ignore_index=True)
    _compare_nc(py_nc, r_nc)

    # CC1 norm_corr at selected sigma
    r_cc1 = r_nc[(r_nc["sigmaValues"] == r_sigma_choice) & (r_nc["CC_index"] == 1)]["normalizedCorrelation"]
    py_cc1_row = py_nc[(py_nc["sigma"] == obj.sigma_value_choice) & (py_nc["CC_index"] == 1)]["normalized_correlation"]
    if len(r_cc1) > 0 and len(py_cc1_row) > 0:
        diff_nc = abs(float(r_cc1.values[0]) - float(py_cc1_row.values[0]))
        passes.append(_check("NC at sigma_choice: |diff| < 0.02", float(1 - diff_nc), 0.98))

    # ── Cell scores comparison ────────────────────────────────────────────────
    print("\n── Cell scores (CC1) comparison ──")
    sigma = obj.sigma_value_choice
    sigma_str = "sigma_" + str(sigma).replace(".", "p")
    r_cs_file = REF_DIR / f"cell_scores_{sigma_str}.csv"
    if r_cs_file.exists():
        r_cs = pd.read_csv(r_cs_file, index_col=0).values[:, 0]
        py_cs = obj.cell_scores[f"cellScores|sigma{sigma}|{CELL_TYPE}"][:, 0]
        r_corr = _corr_sign_flip(py_cs, r_cs)
        passes.append(_check(f"Cell score CC1 |r|", r_corr, 0.99))
    else:
        print(f"  (no R cell scores file for sigma={sigma}, skipping)")

    print(f"\nOrganoid: {sum(passes)}/{len(passes)} checks passed")
    return passes


# ─────────────────────────────────────────────────────────────────────────────
# Tutorial 2: MERFISH brain (two cell types)
# ─────────────────────────────────────────────────────────────────────────────

def run_merfish():
    print("\n" + "=" * 70)
    print("TUTORIAL 2: Brain MERFISH (two cell types — D1/D2 neurons)")
    print("=" * 70)

    DATA_DIR  = Path.home() / "Library/CloudStorage/Dropbox/Zhuang-ABCA-1.054_1"
    EXPR_FILE = DATA_DIR / "Zhuang_ABCA_1.054_subset_expression_data.csv"
    META_FILE = DATA_DIR / "Zhuang_ABCA_1.054_subset_metadata.csv"
    REF_DIR   = REF_BASE / "merfish"

    N_PCA        = 40
    SIGMA_VALUES = [0.1, 0.14, 0.2, 0.5]
    N_CC         = 2
    CELL_TYPE_A  = "061 STR D1 Gaba"
    CELL_TYPE_B  = "062 STR D2 Gaba"
    CELL_TYPES   = [CELL_TYPE_A, CELL_TYPE_B]

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading MERFISH data...")
    expr_df = pd.read_csv(EXPR_FILE, index_col=0)
    meta    = pd.read_csv(META_FILE, index_col=0)
    # Align: expression uses cell_label as rownames, meta uses integer index
    meta.index = meta["cell_label"].astype(str)

    # Subset to D1 and D2
    keep    = meta["subclass"].isin(CELL_TYPES)
    expr_df = expr_df.loc[keep]
    meta    = meta.loc[keep]
    meta["x"] = pd.to_numeric(meta["x"])
    meta["y"] = pd.to_numeric(meta["y"])
    print(f"  D1: {(meta['subclass']==CELL_TYPE_A).sum()}  D2: {(meta['subclass']==CELL_TYPE_B).sum()}")

    # Expression is already pre-normalized
    expr_norm  = expr_df.values.astype(float)
    location   = pd.DataFrame({"x": meta["x"].values, "y": meta["y"].values})
    cell_types = meta["subclass"].values

    # ── Python pipeline ───────────────────────────────────────────────────────
    print("Running Python CoPro pipeline...")
    obj = cp.CoProSingle(
        normalized_data=expr_norm,
        location_data=location,
        meta_data=meta.reset_index(drop=True),
        cell_types=cell_types,
    )
    obj = cp.subset_data(obj, CELL_TYPES)
    obj = cp.compute_pca(obj, n_pca=N_PCA)
    obj = cp.compute_distance(obj, normalize=False)
    obj = cp.compute_kernel_matrix(obj, sigma_values=SIGMA_VALUES,
                                   upper_quantile=0.85, lower_limit=5e-7)
    obj = cp.run_skr_cca(obj, scale_pcs=True, n_cc=N_CC, max_iter=500)
    obj = cp.compute_normalized_correlation(obj)
    obj = cp.compute_gene_and_cell_scores(obj)
    print(f"  Sigma choice (Python): {obj.sigma_value_choice}")

    # ── R reference ───────────────────────────────────────────────────────────
    r_sigma_choice = float(open(REF_DIR / "sigma_choice.txt").read().strip())
    print(f"  Sigma choice (R):      {r_sigma_choice}")

    passes = []

    # ── PCA comparison ────────────────────────────────────────────────────────
    for ct, ct_safe in [(CELL_TYPE_A, "061_STR_D1_Gaba"),
                         (CELL_TYPE_B, "062_STR_D2_Gaba")]:
        print(f"\n── PCA comparison ({ct}) ──")
        r_scores  = pd.read_csv(REF_DIR / f"pca_scores_{ct_safe}.csv",  index_col=0).values
        r_sdev    = pd.read_csv(REF_DIR / f"pca_sdev_{ct_safe}.csv",    index_col=0)["sdev"].values
        py_scores = obj.pca_global[ct]["scores"]
        py_sdev   = obj.pca_global[ct]["sdev"]

        for pc in [0, 1, 2, 9, 19, 29, 39]:
            if pc >= min(py_scores.shape[1], r_scores.shape[1]): continue
            r_corr = _corr_sign_flip(py_scores[:, pc], r_scores[:, pc])
            passes.append(_check(f"PC{pc+1} score |r|", r_corr, 0.99))

        sdev_rel_err = np.abs(py_sdev - r_sdev[:len(py_sdev)]) / (r_sdev[:len(py_sdev)] + 1e-12)
        passes.append(_check("sdev max rel err < 0.01", float(1 - sdev_rel_err.max()), 0.99))

    # ── Normalized correlation comparison ─────────────────────────────────────
    print("\n── Normalized correlation ──")
    r_nc = pd.read_csv(REF_DIR / "normalized_correlation.csv")
    nc_frames = [df for df in obj.normalized_correlation.values() if df is not None]
    py_nc = pd.concat(nc_frames, ignore_index=True)
    _compare_nc(py_nc, r_nc)

    sigma = obj.sigma_value_choice
    py_cc1 = py_nc[(py_nc["sigma"] == sigma) & (py_nc["CC_index"] == 1)]["normalized_correlation"]
    r_cc1  = r_nc[(r_nc["sigmaValues"] == r_sigma_choice) & (r_nc["CC_index"] == 1)]["normalizedCorrelation"]
    if len(r_cc1) > 0 and len(py_cc1) > 0:
        diff_nc = abs(float(r_cc1.values[0]) - float(py_cc1.values[0]))
        passes.append(_check("NC at sigma_choice: |diff| < 0.02", float(1 - diff_nc), 0.98))

    # ── Cell scores comparison ────────────────────────────────────────────────
    print("\n── Cell scores (CC1) comparison ──")
    sigma_str = "sigma_" + str(sigma).replace(".", "p")
    for ct, ct_safe in [(CELL_TYPE_A, "061_STR_D1_Gaba"),
                         (CELL_TYPE_B, "062_STR_D2_Gaba")]:
        r_cs_file = REF_DIR / f"cell_scores_{sigma_str}_{ct_safe}.csv"
        if r_cs_file.exists():
            r_cs  = pd.read_csv(r_cs_file, index_col=0).values[:, 0]
            py_cs = obj.cell_scores[f"cellScores|sigma{sigma}|{ct}"][:, 0]
            r_corr = _corr_sign_flip(py_cs, r_cs)
            passes.append(_check(f"Cell score CC1 |r| ({ct[:3]})", r_corr, 0.99))
        else:
            print(f"  (no R cell scores file for {ct}, sigma={sigma})")

    print(f"\nMERFISH: {sum(passes)}/{len(passes)} checks passed")
    return passes


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    org_passes = run_organoid()
    mer_passes = run_merfish()

    total = len(org_passes) + len(mer_passes)
    passed = sum(org_passes) + sum(mer_passes)
    print("\n" + "=" * 70)
    print(f"TOTAL: {passed}/{total} checks passed")
    if passed < total:
        sys.exit(1)
