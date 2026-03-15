#!/usr/bin/env Rscript
# Run R CoPro pipeline on organoid seqFISH data (one cell type, Epithelial).
# Matches tutorial_one_cell_type.ipynb preprocessing: cap 95th pct + log1p.
# Usage: Rscript tests/run_r_reference_organoid.R
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
DATA_DIR <- file.path(
  path.expand("~"),
  "Library/CloudStorage/Dropbox/DIALOGUE_plus project/Raj_lab_data/72hr/roi1/output"
)
EXPR_FILE <- file.path(DATA_DIR, "cell_by_gene/cell_by_gene.csv")
META_FILE <- file.path(DATA_DIR, "attributes/cell_attributes.csv")
OUT_DIR   <- file.path(repo_root, "copro-python", "tests", "r_reference", "organoid")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Parameters (matching Python tutorial) ────────────────────────────────────
N_PCA        <- 30
SIGMA_VALUES <- c(0.01, 0.02, 0.05, 0.1, 0.15, 0.2)
N_CC         <- 4
CELL_TYPE    <- "Epithelial"

# ── Load data ────────────────────────────────────────────────────────────────
cat("Loading organoid data...\n")
expr_raw <- read.csv(EXPR_FILE, row.names = 1)   # cells × genes
meta     <- read.csv(META_FILE)

n_cells <- nrow(meta)
cell_ids <- paste0("cell_", meta$label)
rownames(expr_raw) <- cell_ids
rownames(meta)     <- cell_ids

cat(sprintf("Cells: %d   Genes: %d\n", nrow(expr_raw), ncol(expr_raw)))

# ── Preprocess: cap at 95th percentile ───────────────────────────────────────
filter_out <- function(x) {
  upper <- quantile(x, 0.95)
  x[x > upper] <- upper
  x
}
expr_capped <- apply(expr_raw, MARGIN = 2, FUN = filter_out)

# ── Normalize: log1p ─────────────────────────────────────────────────────────
expr_norm <- log1p(expr_capped)
cat("Preprocessing complete (cap 95th pct + log1p).\n")

# ── Save normalized expression for Python comparison ─────────────────────────
write.csv(expr_norm, file.path(OUT_DIR, "expr_norm.csv"), quote = FALSE)

# ── Spatial coordinates (scaled by 5000) ─────────────────────────────────────
location <- data.frame(
  x = meta$center_x / 5000,
  y = meta$center_y / 5000,
  row.names = cell_ids
)

cell_types <- rep(CELL_TYPE, n_cells)
names(cell_types) <- cell_ids

# ── Build CoPro object ───────────────────────────────────────────────────────
obj <- newCoProSingle(
  normalizedData = expr_norm,
  locationData   = location,
  metaData       = meta,
  cellTypes      = cell_types
)

obj <- subsetData(obj, cellTypesOfInterest = CELL_TYPE)
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
ct <- CELL_TYPE

# PCA
write.csv(obj@pcaGlobal[[ct]]$x,
          file.path(OUT_DIR, "pca_scores.csv"), quote = FALSE)
write.csv(data.frame(sdev = obj@pcaGlobal[[ct]]$sdev),
          file.path(OUT_DIR, "pca_sdev.csv"), quote = FALSE)
write.csv(obj@pcaGlobal[[ct]]$rotation,
          file.path(OUT_DIR, "pca_rotation.csv"), quote = FALSE)

# Sigma choice
writeLines(as.character(obj@sigmaValueChoice),
           file.path(OUT_DIR, "sigma_choice.txt"))

# Normalized correlation
nc_df <- do.call(rbind, obj@normalizedCorrelation)
write.csv(nc_df, file.path(OUT_DIR, "normalized_correlation.csv"),
          row.names = FALSE, quote = FALSE)

# CCA weights and cell scores for each sigma
sigma_names <- paste("sigma", SIGMA_VALUES, sep = "_")
for (s_name in sigma_names) {
  if (is.null(obj@skrCCAOut[[s_name]])) next
  sigma_str <- gsub("\\.", "p", s_name)
  write.csv(obj@skrCCAOut[[s_name]][[ct]],
            file.path(OUT_DIR, paste0("w_", sigma_str, ".csv")),
            quote = FALSE)
  sigma_val <- as.numeric(gsub("sigma_", "", s_name))
  cell_flat <- paste0("cellScores|sigma", sigma_val, "|", ct)
  if (cell_flat %in% names(obj@cellScores)) {
    write.csv(obj@cellScores[[cell_flat]],
              file.path(OUT_DIR, paste0("cell_scores_", sigma_str, ".csv")),
              quote = FALSE)
  }
  gene_flat <- paste0("geneScores|sigma", sigma_val, "|", ct)
  if (gene_flat %in% names(obj@geneScores)) {
    write.csv(obj@geneScores[[gene_flat]],
              file.path(OUT_DIR, paste0("gene_scores_", sigma_str, ".csv")),
              quote = FALSE)
  }
}

# Kernel matrix for sigma choice
sigma_choice_val <- obj@sigmaValueChoice
kernel_flat <- paste0("kernel|sigma", sigma_choice_val, "|", ct, "|", ct)
if (kernel_flat %in% names(obj@kernelMatrices)) {
  sigma_str_choice <- gsub("\\.", "p", paste0("sigma_", sigma_choice_val))
  write.csv(obj@kernelMatrices[[kernel_flat]],
            file.path(OUT_DIR, paste0("kernel_", sigma_str_choice, ".csv")),
            row.names = FALSE, quote = FALSE)
  cat("Exported kernel matrix for sigma =", sigma_choice_val, "\n")
}

cat("Reference results written to", OUT_DIR, "\n")
