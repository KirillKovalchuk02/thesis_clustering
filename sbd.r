# install.packages("TSclust")
# library(TSclust)

# my_data <- read.csv("stocks_data_filled.csv", header = TRUE)

# print(sum(is.na(my_data)))

# print(which(apply(is.na(my_data), 1, any)))
# dist_matrix <- diss(my_data, METHOD = "SBD")

# # View the distance matrix
# #print(as.matrix(dist_matrix))


# write.csv(dmatrix, "sbd_matrix.csv", row.names = FALSE)



#install.packages("dtwclust")
library(dtwclust)


my_data <- read.csv("stocks_data_filled.csv", header = TRUE)




# for (col_name in colnames(my_data)) {
#   orig_values <- my_data[[col_name]]
#   num_values <- as.numeric(as.character(orig_values))
#   na_count <- sum(is.na(num_values))
  
#   if (na_count > 0) {
#     cat("Column:", col_name, "- NAs after conversion:", na_count, "\n")
#     prob_indices <- which(is.na(num_values))
#     cat("First few problematic values:", head(orig_values[prob_indices]), "\n\n")
#   }
# }



print(colnames(my_data))

data_without_date <- my_data[, -1]

# Check data structure and convert to numeric if needed
my_data_numeric <- apply(data_without_date, 2, function(x) as.numeric(as.character(x)))

ts_matrix <- t(my_data_numeric)

# Convert to a list of time series
ts_list <- lapply(seq_len(nrow(ts_matrix)), function(i) as.numeric(ts_matrix[i,]))

# Verify that all elements are numeric vectors without NAs
all_valid <- all(sapply(ts_list, function(x) is.numeric(x) && !any(is.na(x))))
if (!all_valid) {
  stop("Some series still contain NA values or non-numeric elements")
}

# Calculate SBD distance matrix
dist_matrix <- proxy::dist(ts_list, method = "SBD")

# Convert to standard matrix format
dist_matrix_mat <- as.matrix(dist_matrix)

# Save
write.csv(dist_matrix_mat, "sbd_matrix.csv", row.names = FALSE)