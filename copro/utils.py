"""Utility functions shared across modules."""

from __future__ import annotations

import numpy as np



def center_scale_matrix(
    matrix: np.ndarray,
    zero_sd_threshold: float = 1e-3,
    nz_proportion_threshold: float = 0.01,
) -> np.ndarray:
    """Center and scale matrix columns — matches R center_scale_matrix_opt.

    Columns with std < zero_sd_threshold or non-zero proportion < nz_proportion_threshold
    are not scaled (scale factor = 1).
    """
    col_means = matrix.mean(axis=0)
    col_sds = matrix.std(axis=0, ddof=1)
    col_nz = np.sum(matrix != 0, axis=0) / matrix.shape[0]

    col_sds_safe = col_sds.copy()
    bad_cols = (col_sds < zero_sd_threshold) | (col_nz < nz_proportion_threshold)
    col_sds_safe[bad_cols] = 1.0

    return (matrix - col_means) / col_sds_safe
