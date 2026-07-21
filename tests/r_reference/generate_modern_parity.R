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

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write.csv(
  data.frame(metric = names(values), value = as.numeric(values)),
  output, row.names = FALSE, quote = FALSE
)
