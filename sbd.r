library(dtwclust)


my_data <- read.csv("data_for_R/dataframe_returns.csv", header = TRUE)

data_without_date <- my_data[, -1]

my_data_numeric <- apply(data_without_date, 2, function(x) as.numeric(as.character(x)))

ts_matrix <- t(my_data_numeric) #transpose


ts_list <- lapply(seq_len(nrow(ts_matrix)), function(i) as.numeric(ts_matrix[i,]))# Convert to a list of time series

# Verify that all elements are numeric vectors without NAs
all_valid <- all(sapply(ts_list, function(x) is.numeric(x) && !any(is.na(x))))
if (!all_valid) {
  stop("Some series still contain NA values or non-numeric elements")
}

# Calculate SBD distance matrix
dist_matrix <- proxy::dist(ts_list, method = "SBD")

# Convert to standard matrix format
dist_matrix_mat <- as.matrix(dist_matrix)

write.csv(dist_matrix_mat, "data_for_R/sbd_matrix.csv", row.names = FALSE)