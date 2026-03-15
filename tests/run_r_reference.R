#!/usr/bin/env Rscript
# Run R CoPro pipeline on simulation datasets and export reference results.
# Usage: Rscript --no-site-file --no-init-file tests/run_r_reference.R
#        (run from the repo root: CoPro Claude Code/)

suppressPackageStartupMessages({
  # Add all renv-cached package directories to the library path
  renv_cache <- path.expand(
    "~/Library/Caches/org.R-project.R/R/renv/cache/v5/macos/R-4.5/aarch64-apple-darwin20"
  )
  pkg_dirs <- list.dirs(renv_cache, recursive = TRUE, full.names = TRUE)
  lib_dirs <- unique(dirname(pkg_dirs[file.exists(file.path(pkg_dirs, "DESCRIPTION"))]))
  .libPaths(c(lib_dirs, .libPaths()))

  library(arrow)
  library(irlba)
  library(fields)
  library(matrixStats)
})

# Load CoPro package from source
repo_root <- getwd()
copro_src <- file.path(repo_root, "CoPro package")
if (!requireNamespace("CoPro", quietly = TRUE)) {
  pkgload::load_all(copro_src, quiet = TRUE)
}

# Paths (relative to repo root)
repo_root <- getwd()
DATA_DIR <- file.path(repo_root, "CoPro scripts", "2026_spatial_simulation")
OUT_BASE <- file.path(repo_root, "copro-python", "tests", "r_reference")

# Parameters matching the actual R analysis (02_run_copro_analysis_alternative.R)
SIGMA_VALUES <- c(0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5)
N_PCA <- 25
N_CC <- 2
CELL_TYPES <- c("A", "B")


run_pipeline <- function(expr_file, meta_file, out_dir) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  cat("Loading data from", basename(expr_file), "\n")
  expr_df <- read_parquet(expr_file)
  cell_ids <- if ("cell_id" %in% colnames(expr_df)) as.character(expr_df$cell_id) else as.character(seq_len(nrow(expr_df)))
  gene_cols <- setdiff(colnames(expr_df), "cell_id")
  expr <- as.matrix(expr_df[, gene_cols])
  class(expr) <- "numeric"
  rownames(expr) <- cell_ids
  meta <- as.data.frame(read_parquet(meta_file))
  rownames(meta) <- cell_ids

  cell_types <- as.character(meta$cell_type)
  location <- data.frame(x = meta$x, y = meta$y, row.names = cell_ids)

  obj <- newCoProSingle(
    normalizedData = expr,
    locationData = location,
    metaData = meta,
    cellTypes = cell_types
  )

  obj <- subsetData(obj, cellTypesOfInterest = CELL_TYPES)
  obj <- computePCA(obj, nPCA = N_PCA, center = TRUE, scale. = TRUE)
  obj <- computeDistance(obj, normalizeDistance = FALSE, truncateLowDist = TRUE)
  obj <- computeKernelMatrix(obj, sigmaValues = SIGMA_VALUES, upperQuantile = 0.85,
                              rowNormalizeKernel = TRUE)
  obj <- runSkrCCA(obj, scalePCs = TRUE, nCC = N_CC, maxIter = 500, tol = 1e-5)
  obj <- computeNormalizedCorrelation(obj)
  obj <- computeGeneAndCellScores(obj)

  # Export PCA scores, sdev, rotation
  for (ct in CELL_TYPES) {
    write.csv(obj@pcaGlobal[[ct]]$x,
              file.path(out_dir, paste0("pca_scores_", ct, ".csv")),
              quote = FALSE)
    write.csv(data.frame(sdev = obj@pcaGlobal[[ct]]$sdev),
              file.path(out_dir, paste0("pca_sdev_", ct, ".csv")),
              quote = FALSE)
    write.csv(obj@pcaGlobal[[ct]]$rotation,
              file.path(out_dir, paste0("pca_rotation_", ct, ".csv")),
              quote = FALSE)
  }

  # Export CCA weights and cell scores for each sigma
  sigma_names <- paste("sigma", SIGMA_VALUES, sep = "_")
  for (t in sigma_names) {
    if (!t %in% names(obj@skrCCAOut) || is.null(obj@skrCCAOut[[t]])) next
    sigma_str <- gsub("\\.", "p", t)
    for (ct in CELL_TYPES) {
      write.csv(obj@skrCCAOut[[t]][[ct]],
                file.path(out_dir, paste0("w_", sigma_str, "_", ct, ".csv")),
                quote = FALSE)
    }
    sigma_val <- as.numeric(gsub("sigma_", "", t))
    for (ct in CELL_TYPES) {
      cell_flat <- paste0("cellScores|sigma", sigma_val, "|", ct)
      if (cell_flat %in% names(obj@cellScores)) {
        write.csv(obj@cellScores[[cell_flat]],
                  file.path(out_dir, paste0("cell_scores_", sigma_str, "_", ct, ".csv")),
                  quote = FALSE)
      }
    }
  }

  # Export normalized correlation
  norm_corr_df <- do.call(rbind, obj@normalizedCorrelation)
  write.csv(norm_corr_df,
            file.path(out_dir, "normalized_correlation.csv"),
            row.names = FALSE, quote = FALSE)

  # Export sigma choice
  writeLines(as.character(obj@sigmaValueChoice),
             file.path(out_dir, "sigma_choice.txt"))

  # Export kernel matrix for sigma_choice (A-B pair)
  sigma_choice_val <- obj@sigmaValueChoice
  sigma_str <- gsub("\\.", "p", paste0("sigma_", sigma_choice_val))
  kernel_flat_name <- paste0("kernel|sigma", sigma_choice_val, "|A|B")
  if (kernel_flat_name %in% names(obj@kernelMatrices)) {
    write.csv(obj@kernelMatrices[[kernel_flat_name]],
              file.path(out_dir, paste0("kernel_", sigma_str, "_A_B.csv")),
              row.names = FALSE, quote = FALSE)
    cat("Exported kernel matrix for sigma =", sigma_choice_val, "\n")
  }

  cat("Reference results written to", out_dir, "\n")
}

run_pipeline(
  expr_file = file.path(DATA_DIR, "alternative_round1_expression.parquet"),
  meta_file  = file.path(DATA_DIR, "alternative_round1_metadata.parquet"),
  out_dir    = file.path(OUT_BASE, "alt_round1")
)

run_pipeline(
  expr_file = file.path(DATA_DIR, "null_round1_expression.parquet"),
  meta_file  = file.path(DATA_DIR, "null_round1_metadata.parquet"),
  out_dir    = file.path(OUT_BASE, "null_round1")
)

cat("Done.\n")
