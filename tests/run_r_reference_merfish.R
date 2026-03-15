#!/usr/bin/env Rscript
# Run R CoPro pipeline on brain MERFISH data (two cell types, D1/D2 neurons).
# Matches tutorial_two_cell_types.ipynb: uses pre-normalized expression CSV.
# Usage: Rscript tests/run_r_reference_merfish.R
#        (run from repo root: CoPro Claude Code/)

suppressPackageStartupMessages({
  renv_cache <- path.expand(
    "~/Library/Caches/org.R-project.R/R/renv/cache/v5/macos/R-4.5/aarch64-apple-darwin20"
  )
  pkg_dirs <- list.dirs(renv_cache, recursive = TRUE, full.names = TRUE)
  lib_dirs <- unique(dirname(pkg_dirs[file.exists(file.path(pkg_dirs, "DESCRIPTION"))]))
  .libPaths(c(lib_dirs, .libPaths()))

  library(irlba)
  library(fields)
  library(matrixStats)
})

# Load CoPro from source
repo_root <- getwd()
copro_src <- file.path(repo_root, "CoPro package")
pkgload::load_all(copro_src, quiet = TRUE)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  <- file.path(
  path.expand("~"),
  "Library/CloudStorage/Dropbox/Zhuang-ABCA-1.054_1"
)
EXPR_FILE <- file.path(DATA_DIR, "Zhuang_ABCA_1.054_subset_expression_data.csv")
META_FILE <- file.path(DATA_DIR, "Zhuang_ABCA_1.054_subset_metadata.csv")
OUT_DIR   <- file.path(repo_root, "copro-python", "tests", "r_reference", "merfish")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Parameters (matching Python tutorial) ────────────────────────────────────
N_PCA        <- 40
SIGMA_VALUES <- c(0.1, 0.14, 0.2, 0.5)
N_CC         <- 2
CELL_TYPES   <- c("061 STR D1 Gaba", "062 STR D2 Gaba")

# ── Load data ────────────────────────────────────────────────────────────────
cat("Loading MERFISH data...\n")
expr_df <- read.csv(EXPR_FILE, row.names = 1, check.names = FALSE)
meta     <- read.csv(META_FILE, row.names = 1)
# Align: expression uses cell_label as rownames, meta uses integer index
rownames(meta) <- as.character(meta$cell_label)

cat(sprintf("Full dataset — Cells: %d   Genes: %d\n", nrow(expr_df), ncol(expr_df)))

# ── Subset to D1 and D2 neurons ───────────────────────────────────────────────
keep      <- meta$subclass %in% CELL_TYPES
expr_sub  <- expr_df[keep, ]
meta_sub  <- meta[keep, ]

meta_sub$x <- as.numeric(meta_sub$x)
meta_sub$y <- as.numeric(meta_sub$y)

cat(sprintf("After subsetting — Cells: %d\n", nrow(expr_sub)))
print(table(meta_sub$subclass))

# Expression is already pre-normalized (no additional normalization needed)
expr_norm <- as.matrix(expr_sub)
class(expr_norm) <- "numeric"

# ── Spatial coordinates ───────────────────────────────────────────────────────
location <- data.frame(
  x = meta_sub$x,
  y = meta_sub$y,
  row.names = rownames(meta_sub)
)

cell_types <- as.character(meta_sub$subclass)
names(cell_types) <- rownames(meta_sub)

# ── Build CoPro object ───────────────────────────────────────────────────────
obj <- newCoProSingle(
  normalizedData = expr_norm,
  locationData   = location,
  metaData       = meta_sub,
  cellTypes      = cell_types
)

obj <- subsetData(obj, cellTypesOfInterest = CELL_TYPES)
obj <- computePCA(obj, nPCA = N_PCA, center = TRUE, scale. = TRUE)
obj <- computeDistance(obj, distType = "Euclidean2D", normalizeDistance = FALSE)
obj <- computeKernelMatrix(
  obj,
  sigmaValues      = SIGMA_VALUES,
  upperQuantile    = 0.85,
  normalizeKernel  = FALSE,
  lowerLimit       = 5e-7
)
obj <- runSkrCCA(obj, scalePCs = TRUE, nCC = N_CC, maxIter = 500, tol = 1e-5)
obj <- computeNormalizedCorrelation(obj, tol = 1e-3)
obj <- computeGeneAndCellScores(obj)

cat(sprintf("Pipeline complete. Sigma choice: %s\n", obj@sigmaValueChoice))

# ── Export results ───────────────────────────────────────────────────────────

# PCA for each cell type
for (ct in CELL_TYPES) {
  ct_safe <- gsub(" ", "_", ct)
  write.csv(obj@pcaGlobal[[ct]]$x,
            file.path(OUT_DIR, paste0("pca_scores_", ct_safe, ".csv")),
            quote = FALSE)
  write.csv(data.frame(sdev = obj@pcaGlobal[[ct]]$sdev),
            file.path(OUT_DIR, paste0("pca_sdev_", ct_safe, ".csv")),
            quote = FALSE)
  write.csv(obj@pcaGlobal[[ct]]$rotation,
            file.path(OUT_DIR, paste0("pca_rotation_", ct_safe, ".csv")),
            quote = FALSE)
}

# Sigma choice
writeLines(as.character(obj@sigmaValueChoice),
           file.path(OUT_DIR, "sigma_choice.txt"))

# Normalized correlation
nc_df <- do.call(rbind, obj@normalizedCorrelation)
write.csv(nc_df, file.path(OUT_DIR, "normalized_correlation.csv"),
          row.names = FALSE, quote = FALSE)

# CCA weights and cell/gene scores for each sigma
sigma_names <- paste("sigma", SIGMA_VALUES, sep = "_")
for (s_name in sigma_names) {
  if (is.null(obj@skrCCAOut[[s_name]])) next
  sigma_str <- gsub("\\.", "p", s_name)
  sigma_val <- as.numeric(gsub("sigma_", "", s_name))

  for (ct in CELL_TYPES) {
    ct_safe <- gsub(" ", "_", ct)
    write.csv(obj@skrCCAOut[[s_name]][[ct]],
              file.path(OUT_DIR, paste0("w_", sigma_str, "_", ct_safe, ".csv")),
              quote = FALSE)
    cell_flat <- paste0("cellScores|sigma", sigma_val, "|", ct)
    if (cell_flat %in% names(obj@cellScores)) {
      write.csv(obj@cellScores[[cell_flat]],
                file.path(OUT_DIR, paste0("cell_scores_", sigma_str, "_", ct_safe, ".csv")),
                quote = FALSE)
    }
    gene_flat <- paste0("geneScores|sigma", sigma_val, "|", ct)
    if (gene_flat %in% names(obj@geneScores)) {
      write.csv(obj@geneScores[[gene_flat]],
                file.path(OUT_DIR, paste0("gene_scores_", sigma_str, "_", ct_safe, ".csv")),
                quote = FALSE)
    }
  }
}

# Cross-type kernel matrix for sigma choice
sigma_choice_val <- obj@sigmaValueChoice
sigma_str_choice <- gsub("\\.", "p", paste0("sigma_", sigma_choice_val))
ct_A <- CELL_TYPES[1]; ct_B <- CELL_TYPES[2]
kernel_flat <- paste0("kernel|sigma", sigma_choice_val, "|", ct_A, "|", ct_B)
if (kernel_flat %in% names(obj@kernelMatrices)) {
  write.csv(obj@kernelMatrices[[kernel_flat]],
            file.path(OUT_DIR, paste0("kernel_", sigma_str_choice, "_D1_D2.csv")),
            row.names = FALSE, quote = FALSE)
  cat("Exported cross-type kernel for sigma =", sigma_choice_val, "\n")
}

cat("Reference results written to", OUT_DIR, "\n")
