"""Permutation testing for CoPro — null distribution via spatial permutation."""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from .core import CoProSingle
from .correlation import _spectral_norm, _get_kernel_for_pair
from .optimization import optimize_bilinear, optimize_bilinear_n
from .skrcca import _prepare_pc_matrices


# ---------------------------------------------------------------------------
# Spatial resampling helpers
# ---------------------------------------------------------------------------

def resample_spatial(
    location_data: pd.DataFrame,
    num_bins_x: int = 10,
    num_bins_y: int = 10,
    match_quantile: bool = False,
) -> np.ndarray:
    """Bin-wise spatial resampling that preserves local spatial structure.

    Returns an array of permutation indices (0-based).
    """
    x = location_data["x"].values.astype(float)
    y = location_data["y"].values.astype(float)
    n = len(x)

    x_bin = pd.cut(x, bins=num_bins_x, labels=False)
    y_bin = pd.cut(y, bins=num_bins_y, labels=False)

    # Handle NA bins
    x_bin = np.where(np.isnan(x_bin), 0, x_bin).astype(int)
    y_bin = np.where(np.isnan(y_bin), 0, y_bin).astype(int)

    bin_id = x_bin * num_bins_y + y_bin  # unique bin index

    unique_bins = np.unique(bin_id)
    shuffled_bins = np.random.permutation(unique_bins)
    bin_mapping = dict(zip(unique_bins, shuffled_bins))

    # Build bin membership
    bin_to_indices = {}
    for i in range(n):
        b = bin_id[i]
        bin_to_indices.setdefault(b, []).append(i)

    # Build bin coordinates for neighbor lookup
    bin_to_xy = {}
    for b in unique_bins:
        bin_to_xy[b] = (b // num_bins_y, b % num_bins_y)

    perm = np.empty(n, dtype=int)

    for orig_bin in unique_bins:
        target_bin = bin_mapping[orig_bin]
        orig_idx = bin_to_indices[orig_bin]
        n_points = len(orig_idx)

        # Candidate points from target bin
        candidates = list(bin_to_indices.get(target_bin, []))

        # Expand to neighbors if not enough
        if len(candidates) < n_points:
            tx, ty = bin_to_xy[target_bin]
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = tx + dx, ty + dy
                    if 0 <= nx < num_bins_x and 0 <= ny < num_bins_y:
                        nb = nx * num_bins_y + ny
                        if nb in bin_to_indices and nb != target_bin:
                            candidates.extend(bin_to_indices[nb])
                    if len(candidates) >= n_points:
                        break
                if len(candidates) >= n_points:
                    break

        candidates = np.array(candidates)

        if match_quantile and len(candidates) >= n_points:
            sampled = _match_by_quantile(
                x[orig_idx], y[orig_idx],
                x[candidates], y[candidates],
                n_points,
            )
            perm[np.array(orig_idx)] = candidates[sampled]
        else:
            if len(candidates) < n_points:
                sampled = np.random.choice(len(candidates), n_points, replace=True)
            else:
                sampled = np.random.choice(len(candidates), n_points, replace=False)
            perm[np.array(orig_idx)] = candidates[sampled]

    return perm


def _match_by_quantile(
    orig_x, orig_y, cand_x, cand_y, n_points,
):
    """Match cells by within-tile quantile position (greedy nearest-neighbor)."""
    from scipy.stats import rankdata

    n_cand = len(cand_x)

    if len(orig_x) == 1:
        oqx, oqy = np.array([0.5]), np.array([0.5])
    else:
        oqx = rankdata(orig_x, method="average") / (len(orig_x) + 1)
        oqy = rankdata(orig_y, method="average") / (len(orig_y) + 1)

    if n_cand == 1:
        cqx, cqy = np.array([0.5]), np.array([0.5])
    else:
        cqx = rankdata(cand_x, method="average") / (n_cand + 1)
        cqy = rankdata(cand_y, method="average") / (n_cand + 1)

    # Distance matrix (n_points x n_cand)
    dist = (oqx[:, None] - cqx[None, :]) ** 2 + (oqy[:, None] - cqy[None, :]) ** 2

    sampled = np.empty(n_points, dtype=int)
    for i in range(n_points):
        best = np.argmin(dist[i])
        sampled[i] = best
        dist[:, best] = np.inf  # mark used

    return sampled


def generate_toroidal_permutations(
    x: np.ndarray, y: np.ndarray, n_permu: int,
) -> np.ndarray:
    """Generate toroidal (wrap-around) shift permutation indices.

    Returns (n_cells, n_permu) matrix of 0-based indices.
    """
    n = len(x)
    x_range = (x.min(), x.max())
    y_range = (y.min(), y.max())
    x_width = x_range[1] - x_range[0]
    y_width = y_range[1] - y_range[0]

    if x_width < 1e-10 or y_width < 1e-10:
        warnings.warn("Spatial extent very small; toroidal shift may not work well.")
        return np.tile(np.arange(n), (n_permu, 1)).T

    orig_order = np.lexsort((y, x))
    perm_matrix = np.empty((n, n_permu), dtype=int)

    for tt in range(n_permu):
        shift_x = np.random.uniform(x_width * 0.1, x_width * 0.9)
        shift_y = np.random.uniform(y_width * 0.1, y_width * 0.9)

        new_x = ((x - x_range[0] + shift_x) % x_width) + x_range[0]
        new_y = ((y - y_range[0] + shift_y) % y_width) + y_range[0]

        new_order = np.lexsort((new_y, new_x))
        # inv_orig[orig_order[i]] = i
        inv_orig = np.empty(n, dtype=int)
        inv_orig[orig_order] = np.arange(n)
        perm_matrix[:, tt] = new_order[inv_orig]

    return perm_matrix


# ---------------------------------------------------------------------------
# Generate permutation indices
# ---------------------------------------------------------------------------

def _get_cell_permu(
    obj: CoProSingle,
    permu_method: str,
    n_permu: int,
    cts: list,
    permu_which: str,
    num_bins_x: int,
    num_bins_y: int,
    match_quantile: bool,
) -> dict:
    """Generate permutation indices for each cell type.

    Returns dict {ct: ndarray (n_cells, n_permu)} for index-based methods,
    or {ct: {"type": "pc_permute", "seeds": array}} for PC-space method.
    """

    def should_permute(ct_index: int) -> bool:
        if permu_which == "second_only":
            return ct_index > 0
        elif permu_which == "first_only":
            return ct_index == 0
        else:  # "both"
            return True

    cell_permu = {}

    for idx, ct in enumerate(cts):
        mask = obj.cell_types_sub == ct
        n_cell = int(mask.sum())

        if not should_permute(idx):
            # Identity permutation
            cell_permu[ct] = np.tile(np.arange(n_cell), (n_permu, 1)).T
            continue

        if permu_method == "global":
            perm = np.empty((n_cell, n_permu), dtype=int)
            for j in range(n_permu):
                perm[:, j] = np.random.permutation(n_cell)
            cell_permu[ct] = perm

        elif permu_method == "bin":
            loc = obj.location_data_sub[mask].reset_index(drop=True)
            loc = loc.copy()
            loc["cell_idx"] = np.arange(n_cell)
            perm = np.empty((n_cell, n_permu), dtype=int)
            for j in range(n_permu):
                perm[:, j] = resample_spatial(
                    loc, num_bins_x, num_bins_y, match_quantile,
                )
            cell_permu[ct] = perm

        elif permu_method == "toroidal":
            loc = obj.location_data_sub[mask]
            x = loc["x"].values.astype(float)
            y = loc["y"].values.astype(float)
            cell_permu[ct] = generate_toroidal_permutations(x, y, n_permu)

        elif permu_method == "pc":
            seeds = np.random.randint(0, 2**31, size=n_permu)
            cell_permu[ct] = {
                "type": "pc_permute",
                "seeds": seeds,
                "n_cell": n_cell,
            }
        else:
            raise ValueError(
                f"permu_method must be 'bin', 'global', 'toroidal', or 'pc'. "
                f"Got: {permu_method}"
            )

    return cell_permu


def _permute_pc_matrix(pc_mat: np.ndarray, seed: int) -> np.ndarray:
    """Permute values within each PC column independently."""
    rng = np.random.RandomState(seed)
    out = pc_mat.copy()
    for col in range(out.shape[1]):
        rng.shuffle(out[:, col])
    return out


# ---------------------------------------------------------------------------
# Public: run_skr_cca_permu
# ---------------------------------------------------------------------------

def run_skr_cca_permu(
    obj: CoProSingle,
    n_permu: int = 20,
    permu_method: str = "bin",
    permu_which: str = "second_only",
    num_bins_x: int = 10,
    num_bins_y: int = 10,
    match_quantile: bool = False,
    conservative: bool = False,
    max_iter: int = 200,
    tol: float = 1e-5,
    seed: int | None = None,
    verbose: bool = True,
):
    """Run permutation testing for spatial CoPro.

    Generates a null distribution by permuting cell positions and re-running
    CCA optimization for each permutation.

    Parameters
    ----------
    obj : CoProSingle
        Must have CCA already computed (``run_skr_cca`` and
        ``compute_normalized_correlation``).
    n_permu : int
        Number of permutations (default 20; use >= 100 for publication).
    permu_method : str
        ``"bin"`` (default), ``"global"``, ``"toroidal"``, or ``"pc"``.
    permu_which : str
        ``"second_only"`` (default), ``"both"``, or ``"first_only"``.
    num_bins_x, num_bins_y : int
        Bin resolution for ``"bin"`` method (default 10).
    match_quantile : bool
        Match cells by quantile position within bins (default False).
    conservative : bool
        Override settings for lower FPR: ``permu_which="second_only"``,
        15×15 bins, ``match_quantile=True``.
    max_iter : int
        Max CCA iterations per permutation.
    tol : float
        CCA convergence tolerance.
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        Print progress.

    Returns
    -------
    obj
        With ``skr_cca_permu_out`` and ``cell_permu`` populated.
    """
    if not obj.skr_cca_out:
        raise ValueError("Run run_skr_cca() before permutation testing.")
    if obj.sigma_value_choice is None:
        raise ValueError("Run compute_normalized_correlation() first to set sigma_value_choice.")

    if conservative:
        if verbose:
            print("Using CONSERVATIVE mode for lower false positive rate:")
            print("  -> permu_which = 'second_only'")
            print("  -> num_bins = 15×15")
            print("  -> match_quantile = True")
        permu_which = "second_only"
        num_bins_x = 15
        num_bins_y = 15
        match_quantile = True

    if n_permu < 10:
        warnings.warn("n_permu < 10 may give unreliable p-values. Consider >= 100.")

    if seed is not None:
        np.random.seed(seed)

    cts = obj.cell_types_of_interest
    n_cc = obj.n_cc
    scale_pcs = getattr(obj, "scale_pcs", True)
    sigma = obj.sigma_value_choice

    if verbose:
        print(f"Permutation settings: method={permu_method}, "
              f"which={permu_which}, n_permu={n_permu}")
        if permu_method == "bin":
            print(f"  bins={num_bins_x}×{num_bins_y}, "
                  f"match_quantile={match_quantile}")

    # Step 1: Generate permutation indices
    if verbose:
        print("Generating permutation indices...")
    cell_permu = _get_cell_permu(
        obj, permu_method, n_permu, cts, permu_which,
        num_bins_x, num_bins_y, match_quantile,
    )
    obj.cell_permu = cell_permu

    # Step 2: Get PC matrices
    pc_mats = _prepare_pc_matrices(obj, scale_pcs, cts)

    # Step 3: Run CCA for each permutation
    if verbose:
        print(f"Running CCA for {n_permu} permutations...")

    permu_results = {}
    for tt in range(n_permu):
        # Build permuted PC matrices
        pc_local = {}
        for ct in cts:
            cp = cell_permu[ct]
            if isinstance(cp, dict) and cp.get("type") == "pc_permute":
                pc_local[ct] = _permute_pc_matrix(pc_mats[ct], int(cp["seeds"][tt]))
            else:
                pc_local[ct] = pc_mats[ct][cp[:, tt]]

        # Run optimization
        w_dict = optimize_bilinear(
            pc_local, obj.kernel_matrices, sigma,
            max_iter=max_iter, tol=tol, verbose=False,
        )

        if n_cc > 1:
            w_dict = optimize_bilinear_n(
                pc_local, obj.kernel_matrices, sigma, w_dict,
                cts, n_cc=n_cc, max_iter=max_iter, tol=tol, verbose=False,
            )

        permu_results[f"permu_{tt+1}"] = w_dict

        if verbose and ((tt + 1) % 10 == 0 or tt + 1 == n_permu):
            print(f"  Completed {tt+1} of {n_permu} permutations")

    obj.skr_cca_permu_out = permu_results

    if verbose:
        print("Permutation testing complete.")
        print("Run compute_normalized_correlation_permu() to compute p-values.")

    return obj


# ---------------------------------------------------------------------------
# Public: compute_normalized_correlation_permu
# ---------------------------------------------------------------------------

def compute_normalized_correlation_permu(
    obj: CoProSingle,
    tol: float = 1e-4,
    verbose: bool = True,
):
    """Compute normalized correlation for each permutation.

    Parameters
    ----------
    obj : CoProSingle
        Must have permutation results from ``run_skr_cca_permu``.
    tol : float
        SVD tolerance for spectral norm.
    verbose : bool
        Print progress.

    Returns
    -------
    obj
        With ``normalized_correlation_permu`` populated (list of DataFrames).
    """
    if not hasattr(obj, "skr_cca_permu_out") or not obj.skr_cca_permu_out:
        raise ValueError("Run run_skr_cca_permu() first.")

    cts = obj.cell_types_of_interest
    n_cc = obj.n_cc
    scale_pcs = getattr(obj, "scale_pcs", True)
    sigma = obj.sigma_value_choice
    n_permu = len(obj.skr_cca_permu_out)

    pc_mats = _prepare_pc_matrices(obj, scale_pcs, cts)
    cell_permu = obj.cell_permu

    # Cell type pairs
    if len(cts) == 1:
        pairs = [(cts[0], cts[0])]
    else:
        pairs = list(combinations(cts, 2))

    # Precompute spectral norms (once)
    if verbose:
        print("Calculating spectral norms...")
    spec_norms = {}
    for ct_i, ct_j in pairs:
        K = _get_kernel_for_pair(obj.kernel_matrices, sigma, ct_i, ct_j)
        spec_norms[(ct_i, ct_j)] = _spectral_norm(K, tol)

    if verbose:
        print(f"Computing normalized correlations for {n_permu} permutations...")

    correlation_permu = {}

    for tt_idx, (pname, w_permu) in enumerate(obj.skr_cca_permu_out.items()):
        # Reconstruct permuted PC matrices for this permutation
        tt = tt_idx  # 0-based
        pc_local = {}
        for ct in cts:
            cp = cell_permu[ct]
            if isinstance(cp, dict) and cp.get("type") == "pc_permute":
                pc_local[ct] = _permute_pc_matrix(pc_mats[ct], int(cp["seeds"][tt]))
            else:
                pc_local[ct] = pc_mats[ct][cp[:, tt]]

        rows = []
        for ct_i, ct_j in pairs:
            K = _get_kernel_for_pair(obj.kernel_matrices, sigma, ct_i, ct_j)
            norm_K = spec_norms[(ct_i, ct_j)]

            for cc in range(n_cc):
                w1 = w_permu[ct_i][:, cc:cc+1]
                w2 = w_permu[ct_j][:, cc:cc+1]

                A = pc_local[ct_i]
                B = pc_local[ct_j]

                Aw1 = A @ w1
                Bw2 = B @ w2

                numerator = float((Aw1.T @ K @ Bw2).flat[0])
                denom = (float(np.linalg.norm(Aw1)) *
                         float(np.linalg.norm(Bw2)) *
                         norm_K)
                nc = 0.0 if abs(denom) < 1e-9 else numerator / denom

                rows.append({
                    "sigma": sigma,
                    "cell_type_1": ct_i,
                    "cell_type_2": ct_j,
                    "CC_index": cc + 1,
                    "normalized_correlation": nc,
                })

        correlation_permu[pname] = pd.DataFrame(rows)

        if verbose and ((tt_idx + 1) % 20 == 0 or tt_idx + 1 == n_permu):
            print(f"  Completed {tt_idx+1} of {n_permu}")

    obj.normalized_correlation_permu = correlation_permu

    if verbose:
        print("Permutation correlation computation complete.")

    return obj


# ---------------------------------------------------------------------------
# Public: calculate_pvalue
# ---------------------------------------------------------------------------

def calculate_pvalue(
    obj: CoProSingle,
    cc_index: int = 1,
    cell_type_1: str | None = None,
    cell_type_2: str | None = None,
) -> dict:
    """Calculate p-value from observed vs. permutation distribution.

    Parameters
    ----------
    obj : CoProSingle
        Must have both ``normalized_correlation`` and
        ``normalized_correlation_permu``.
    cc_index : int
        Which canonical component (1-based, default 1).
    cell_type_1, cell_type_2 : str or None
        Cell type pair. If None, uses the first pair found.

    Returns
    -------
    dict
        ``observed``, ``permu_values``, ``p_value``, ``n_permu``.
    """
    if not obj.normalized_correlation:
        raise ValueError("Run compute_normalized_correlation() first.")
    if not hasattr(obj, "normalized_correlation_permu") or not obj.normalized_correlation_permu:
        raise ValueError("Run compute_normalized_correlation_permu() first.")

    sigma = obj.sigma_value_choice
    sigma_name = f"sigma_{sigma}"

    # Observed value
    obs_df = obj.normalized_correlation.get(sigma_name)
    if obs_df is None or len(obs_df) == 0:
        raise ValueError(f"No observed correlation for {sigma_name}.")

    if cell_type_1 is None or cell_type_2 is None:
        row0 = obs_df[obs_df["CC_index"] == cc_index].iloc[0]
        cell_type_1 = row0["cell_type_1"]
        cell_type_2 = row0["cell_type_2"]

    mask_obs = (
        (obs_df["CC_index"] == cc_index) &
        (obs_df["cell_type_1"] == cell_type_1) &
        (obs_df["cell_type_2"] == cell_type_2)
    )
    observed = float(obs_df.loc[mask_obs, "normalized_correlation"].values[0])

    # Permutation values
    permu_values = []
    for pname, pdf in obj.normalized_correlation_permu.items():
        mask_p = (
            (pdf["CC_index"] == cc_index) &
            (pdf["cell_type_1"] == cell_type_1) &
            (pdf["cell_type_2"] == cell_type_2)
        )
        if mask_p.any():
            permu_values.append(float(pdf.loc[mask_p, "normalized_correlation"].values[0]))

    permu_values = np.array(permu_values)
    n_permu = len(permu_values)
    p_value = float(np.mean(permu_values >= observed)) if n_permu > 0 else np.nan

    return {
        "observed": observed,
        "permu_values": permu_values,
        "p_value": p_value,
        "n_permu": n_permu,
        "cell_type_1": cell_type_1,
        "cell_type_2": cell_type_2,
        "CC_index": cc_index,
    }
