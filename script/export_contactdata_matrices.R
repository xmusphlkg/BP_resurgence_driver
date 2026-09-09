args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript script/export_contactdata_matrices.R <contactdata.tar.gz> <output_dir>")
}

tarball <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

tmp <- tempfile("contactdata_extract_")
dir.create(tmp)
on.exit(unlink(tmp, recursive = TRUE), add = TRUE)
untar(tarball, exdir = tmp)

extdata <- file.path(tmp, "contactdata", "inst", "extdata")
locations <- c("all", "home", "school", "work", "other")
sources <- c("2020", "2017")

age_to_bin <- function(x) {
  lower <- suppressWarnings(as.integer(sub("_.*$", "", x)))
  if (is.na(lower)) {
    return(as.character(x))
  }
  sprintf("[%02d,%02d)", lower, lower + 5L)
}

matrix_rows <- list()
row_idx <- 1L
for (source in sources) {
  for (location in locations) {
    rds_path <- file.path(extdata, sprintf("contact_%s_%s_all.rds", source, location))
    if (!file.exists(rds_path)) {
      next
    }
    matrices <- readRDS(rds_path)
    for (country in names(matrices)) {
      mat <- matrices[[country]]
      cells <- as.data.frame(as.table(mat), stringsAsFactors = FALSE)
      names(cells) <- c("source_age_raw", "target_age_raw", "contacts_per_day")
      matrix_rows[[row_idx]] <- data.frame(
        contactdata_country = country,
        data_source = source,
        location = location,
        source_age_bin = vapply(cells$source_age_raw, age_to_bin, character(1)),
        target_age_bin = vapply(cells$target_age_raw, age_to_bin, character(1)),
        contacts_per_day = as.numeric(cells$contacts_per_day),
        stringsAsFactors = FALSE
      )
      row_idx <- row_idx + 1L
    }
  }
}

matrices <- do.call(rbind, matrix_rows)
write.csv(
  matrices,
  file.path(output_dir, "contactdata_prem_contact_matrices_by_location_16.csv"),
  row.names = FALSE
)

population_path <- file.path(extdata, "population_byage.rds")
if (file.exists(population_path)) {
  population <- readRDS(population_path)
  write.csv(
    population,
    file.path(output_dir, "contactdata_population_byage.csv"),
    row.names = FALSE
  )
}

message(
  "Exported ", nrow(matrices), " contact matrix cells from contactdata package to ",
  normalizePath(output_dir, mustWork = FALSE)
)
