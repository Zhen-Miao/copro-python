# Regenerate the small, repository-owned R parity fixture used by
# test_r_parity_modern.py. Run from the copro-python repository root:
#
#   Rscript tests/r_reference/generate_modern_parity.R

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else
  "tests/r_reference/modern_parity.csv"

if (!requireNamespace("CoPro", quietly = TRUE)) {
  stop("The R CoPro package must be installed to regenerate this fixture.")
}

K <- matrix(c(
  1.2, 0.3, 0.1,
  0.4, 1.1, 0.2,
  0.2, 0.5, 0.9
), nrow = 3, byrow = TRUE)
Rx <- matrix(c(
  1.0, 0.2, 0.1,
  0.2, 0.9, 0.3,
  0.1, 0.3, 1.1
), nrow = 3, byrow = TRUE)
Ry <- matrix(c(
  0.8, 0.1, 0.2,
  0.1, 1.2, 0.4,
  0.2, 0.4, 1.0
), nrow = 3, byrow = TRUE)
A <- matrix(c(
  -1.0, 0.2,
   0.4, 1.1,
   1.3, -0.5
), nrow = 3, byrow = TRUE)
B <- matrix(c(
   0.8, -0.4,
  -1.1, 1.2,
   1.6, 0.5
), nrow = 3, byrow = TRUE)

bidir_none <- CoPro:::.computeAllCCCorrelations(A, B, K, "none")
bidir_row_col <- CoPro:::.computeAllCCCorrelations(A, B, K, "row_or_col")

weights <- list(
  A = matrix(c(0.8, 0.6), ncol = 1),
  B = matrix(c(-0.3, sqrt(0.91)), ncol = 1)
)
self_cov <- list(
  s1 = list(A = diag(c(1.0, 1.5)), B = diag(c(0.8, 1.2))),
  s2 = list(A = diag(c(1.3, 0.7)), B = diag(c(1.1, 0.9)))
)
cross_cov <- list(
  s1 = list(`A-B` = matrix(c(0.7, 0.1, -0.2, 0.5), 2, byrow = TRUE)),
  s2 = list(`A-B` = matrix(c(0.4, -0.3, 0.2, 0.8), 2, byrow = TRUE))
)

values <- c(
  whitened_frob = CoPro:::.whitenedFrobNorm(K, Rx, Ry),
  bidir_none_cc1 = bidir_none[[1]],
  bidir_none_cc2 = bidir_none[[2]],
  bidir_row_col_cc1 = bidir_row_col[[1]],
  bidir_row_col_cc2 = bidir_row_col[[2]],
  gene_space_objective = CoPro:::.compute_p1b_objective(
    weights, self_cov, cross_cov, c("s1", "s2"), c("A", "B")
  )
)

# --- >2-cell-type multi-set deflation parity (DEFECT 1) --------------------
# Pins R's canonical projection-deflation scheme for a 3-cell-type problem
# (used by R whenever n_types > 2 and PCs are whitened, i.e. scale_pcs = TRUE
# / sdev2_list = NULL).  The Python port must reproduce CC1 and CC2 here, and
# additionally give the SAME axes under scale_pcs = FALSE via weighted
# projection (checked separately in tests/test_optimizer_deflation.py).
X_A <- matrix(c(
   0.5, -1.2,  0.3,
   1.1,  0.4, -0.7,
  -0.9,  0.8,  1.0,
   0.2, -0.5,  0.6
), nrow = 4, byrow = TRUE)
X_B <- matrix(c(
  -0.3,  1.0,  0.5,
   0.7, -0.6,  0.9,
   1.2,  0.1, -0.4,
  -0.8,  0.5,  0.2
), nrow = 4, byrow = TRUE)
X_C <- matrix(c(
   0.6,  0.2, -1.1,
  -0.4,  0.9,  0.7,
   0.3, -0.8,  0.4,
   1.0,  0.5, -0.2
), nrow = 4, byrow = TRUE)
K_AB <- matrix(c(
  1.0, 0.2, 0.1, 0.0,
  0.1, 0.9, 0.3, 0.2,
  0.0, 0.4, 1.1, 0.1,
  0.3, 0.1, 0.2, 0.8
), nrow = 4, byrow = TRUE)
K_AC <- matrix(c(
  0.9, 0.1, 0.2, 0.1,
  0.2, 1.0, 0.1, 0.0,
  0.1, 0.3, 0.8, 0.2,
  0.0, 0.2, 0.1, 1.1
), nrow = 4, byrow = TRUE)
K_BC <- matrix(c(
  1.1, 0.0, 0.1, 0.2,
  0.1, 0.8, 0.2, 0.1,
  0.3, 0.1, 1.0, 0.0,
  0.2, 0.2, 0.1, 0.9
), nrow = 4, byrow = TRUE)

mset_sigma <- 0.5
mset_X <- list(A = X_A, B = X_B, C = X_C)
mset_kernels <- list()
mset_kernels[[paste0("kernel|sigma", mset_sigma, "|A|B")]] <- K_AB
mset_kernels[[paste0("kernel|sigma", mset_sigma, "|A|C")]] <- K_AC
mset_kernels[[paste0("kernel|sigma", mset_sigma, "|B|C")]] <- K_BC

mset_w1 <- CoPro::optimize_bilinear(
  mset_X, mset_kernels, mset_sigma,
  max_iter = 20000, tol = 1e-11, sdev2_list = NULL
)
mset_wn <- CoPro::optimize_bilinear_n(
  mset_X, mset_kernels, mset_sigma, w_list = mset_w1,
  cellTypesOfInterest = c("A", "B", "C"), nCC = 2,
  max_iter = 20000, tol = 1e-11, sdev2_list = NULL
)

mset_values <- numeric(0)
for (ct in c("A", "B", "C")) {
  for (ax in 1:2) {
    v <- mset_wn[[ct]][, ax]
    for (k in seq_along(v)) {
      nm <- sprintf("mset_%s_cc%d_%d", ct, ax, k - 1L)
      mset_values[nm] <- v[k]
    }
  }
}

values <- c(values, mset_values)

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write.csv(
  data.frame(metric = names(values), value = as.numeric(values)),
  output, row.names = FALSE, quote = FALSE
)
