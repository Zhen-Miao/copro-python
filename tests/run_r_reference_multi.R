#!/usr/bin/env Rscript
# Generate R reference results for multi-slide (pseudo-split) evaluation.
# Split alternative_round1 into two pseudo-slides by row-index order:
#   first half  → slide1
#   second half → slide2
#
# Usage (from repo root "CoPro Claude Code/"):
#   Rscript --no-site-file --no-init-file copro-python/tests/run_r_reference_multi.R

suppressPackageStartupMessages({
  # Add all renv-cached package directories to the library path
  renv_cache <- path.expand(
    "~/Library/Caches/org.R-project.R/R/renv/cache/v5/macos/R-4.5/aarch64-apple-darwin20"
  )
  # Walk pkg/version/hash/ and collect unique parent dirs (= pkg/version/hash)
  pkg_dirs <- list.dirs(renv_cache, recursive = TRUE, full.names = TRUE)
  # The actual library dirs are those that contain a DESCRIPTION file
  lib_dirs <- unique(dirname(pkg_dirs[file.exists(file.path(pkg_dirs, "DESCRIPTION"))]))
  .libPaths(c(lib_dirs, .libPaths()))

  library(arrow)
  library(irlba)
  library(fields)
  library(matrixStats)
})

# ── Load CoPro package ────────────────────────────────────────────────────────
repo_root  <- getwd()
copro_src  <- file.path(repo_root, "CoPro package")
if (!requireNamespace("CoPro", quietly = TRUE)) {
  pkgload::load_all(copro_src, quiet = TRUE)
}

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR <- file.path(repo_root, "CoPro scripts", "2026_spatial_simulation")
OUT_DIR  <- file.path(repo_root, "copro-python", "tests", "r_reference", "alt_round1_multi")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

SIGMA_VALUES <- c(0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5)
N_PCA        <- 25
N_CC         <- 2
CELL_TYPES   <- c("A", "B")

# ── Load data ─────────────────────────────────────────────────────────────────
cat("Loading data...\n")
expr_df <- read_parquet(file.path(DATA_DIR, "alternative_round1_expression.parquet"))
meta_df <- read_parquet(file.path(DATA_DIR, "alternative_round1_metadata.parquet"))

# Remove cell_id column from expression matrix
gene_cols <- setdiff(colnames(expr_df), "cell_id")
expr <- as.matrix(expr_df[, gene_cols])
cell_ids <- if ("cell_id" %in% colnames(expr_df)) expr_df$cell_id else as.character(seq_len(nrow(expr)))
rownames(expr) <- cell_ids

location <- data.frame(x = meta_df$x, y = meta_df$y, row.names = cell_ids)
meta     <- as.data.frame(meta_df)
rownames(meta) <- cell_ids
cell_types <- as.character(meta_df$cell_type)

# ── Split into two pseudo-slides by x-coordinate median ──────────────────────
# This ensures both cell types appear in both slides
x_median <- median(meta_df$x)
slide_id <- ifelse(meta_df$x < x_median, "slide1", "slide2")
cat(sprintf("Split by x < %.2f: %d cells → slide1=%d, slide2=%d\n",
            x_median, nrow(expr),
            sum(slide_id == "slide1"), sum(slide_id == "slide2")))
# Check cell type distribution per slide
for (s in c("slide1", "slide2")) {
  tab <- table(cell_types[slide_id == s])
  cat(sprintf("  %s: A=%d, B=%d\n", s, tab["A"], tab["B"]))
}

# ── Build CoProMulti object ───────────────────────────────────────────────────
obj <- newCoProMulti(
  normalizedData = expr,
  locationData   = location,
  metaData       = meta,
  cellTypes      = cell_types,
  slideID        = slide_id
)

# ── Run pipeline ──────────────────────────────────────────────────────────────
obj <- subsetData(obj, cellTypesOfInterest = CELL_TYPES)
obj <- computePCA(obj, nPCA = N_PCA, center = TRUE, scale. = TRUE, dataUse = "raw")
obj <- computeDistance(obj, normalizeDistance = FALSE, truncateLowDist = TRUE)
obj <- computeKernelMatrix(obj, sigmaValues = SIGMA_VALUES, upperQuantile = 0.85,
                           rowNormalizeKernel = TRUE)
obj <- runSkrCCA(obj, scalePCs = TRUE, nCC = N_CC, maxIter = 500, tol = 1e-5)
obj <- computeNormalizedCorrelation(obj)
obj <- computeGeneAndCellScores(obj)

# ── Export results ────────────────────────────────────────────────────────────
cat("Exporting results to", OUT_DIR, "\n")

# PCA global (rotation, sdev) — shared across slides
for (ct in CELL_TYPES) {
  write.csv(obj@pcaGlobal[[ct]]$x,
            file.path(OUT_DIR, paste0("pca_scores_global_", ct, ".csv")),
            quote = FALSE)
  write.csv(data.frame(sdev = obj@pcaGlobal[[ct]]$sdev),
            file.path(OUT_DIR, paste0("pca_sdev_", ct, ".csv")),
            quote = FALSE)
  write.csv(obj@pcaGlobal[[ct]]$rotation,
            file.path(OUT_DIR, paste0("pca_rotation_", ct, ".csv")),
            quote = FALSE)
}

# PCA per-slide scores
slides <- getSlideList(obj)
for (sID in slides) {
  for (ct in CELL_TYPES) {
    scores_slide <- obj@pcaResults[[sID]][[ct]]
    if (!is.null(scores_slide)) {
      write.csv(scores_slide,
                file.path(OUT_DIR, paste0("pca_scores_", sID, "_", ct, ".csv")),
                quote = FALSE)
    }
  }
}

# CCA weights and cell scores per sigma
sigma_names <- paste("sigma", SIGMA_VALUES, sep = "_")
for (t in sigma_names) {
  if (!t %in% names(obj@skrCCAOut) || is.null(obj@skrCCAOut[[t]])) next
  sigma_str <- gsub("\\.", "p", t)
  sigma_val <- as.numeric(gsub("sigma_", "", t))

  # Shared weights
  for (ct in CELL_TYPES) {
    write.csv(obj@skrCCAOut[[t]][[ct]],
              file.path(OUT_DIR, paste0("w_", sigma_str, "_", ct, ".csv")),
              quote = FALSE)
  }

  # Per-slide cell scores
  for (sID in slides) {
    for (ct in CELL_TYPES) {
      cell_key <- paste0("cellScores|sigma", sigma_val, "|", sID, "|", ct)
      if (cell_key %in% names(obj@cellScores)) {
        write.csv(obj@cellScores[[cell_key]],
                  file.path(OUT_DIR, paste0("cell_scores_", sigma_str, "_", sID, "_", ct, ".csv")),
                  quote = FALSE)
      }
    }
  }
}

# Normalized correlation
norm_corr_df <- do.call(rbind, obj@normalizedCorrelation)
write.csv(norm_corr_df,
          file.path(OUT_DIR, "normalized_correlation.csv"),
          row.names = FALSE, quote = FALSE)

# Sigma choice
writeLines(as.character(obj@sigmaValueChoice),
           file.path(OUT_DIR, "sigma_choice.txt"))

# Export kernel matrices for sigma_choice (per slide, A-B pair)
sigma_choice_val <- obj@sigmaValueChoice
sigma_str <- gsub("\\.", "p", paste0("sigma_", sigma_choice_val))
for (sID in slides) {
  kernel_flat_name <- paste0("kernel|sigma", sigma_choice_val, "|", sID, "|A|B")
  if (kernel_flat_name %in% names(obj@kernelMatrices)) {
    write.csv(obj@kernelMatrices[[kernel_flat_name]],
              file.path(OUT_DIR, paste0("kernel_", sigma_str, "_", sID, "_A_B.csv")),
              row.names = FALSE, quote = FALSE)
    cat("Exported kernel matrix for sigma =", sigma_choice_val, "slide =", sID, "\n")
  }
}

cat("Done. Results written to", OUT_DIR, "\n")
