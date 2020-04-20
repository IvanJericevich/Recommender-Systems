# Title: Recommender Systems
# Authors: Ivan Jericevich & Yovna Junglee
# 1. Preliminaries
# 2. Data Preparation
# 3. KNN
# 3.1 Item-Based
# 3.2 User-Based
# 4. KNN with PCA
# 5. Save Workspace
####################################################################################################

### 1. Preliminaries
cat("\014") # Clear console
rm(list = ls()) # Clear environment
if(!is.null(dev.list())) dev.off() # Clear plots
gc() # Garbage collection to get extra ram for large matrix
setwd("")
# load("KNN Workspace.RData") # Load data from previous workspace
list_of_packages = c("reshape2", "doParallel", "Rcpp") # This automatically installs and loads packages not already installed on the users computer
new_packages = list_of_packages[!(list_of_packages %in% installed.packages()[, "Package"])] # reshape2 = acast; doParallel = %dopar%; Rcpp = sourceCpp
if(length(new_packages) > 0) {install.packages(new_packages)}
lapply(list_of_packages, require, character.only = TRUE)
sourceCpp(paste0(getwd(), "/Functions/Pearson Correlation.cpp"))
sourceCpp(paste0(getwd(), "/Functions/KNN.cpp"))
source(paste0(getwd(), "/Functions/Supplementary.R"))
####################################################################################################

### 2. Data Preparation
ratings = read.csv(file = paste0(getwd(), "/Data/ratings.csv"), header = TRUE)[, -4] # Sorted according to userId
movies = merge(read.csv(file = paste0(getwd(), "/Data/movies.csv"), header = TRUE), aggregate(tag ~ movieId, data = read.csv(file = paste0(getwd(), "/Data/tags.csv"), header = TRUE)[, -c(1, 4)], paste, collapse = "|"), all = TRUE) # Merge movies file with the tags file; sorted according to movieId
movies$year = sapply(as.character(movies$title), substrRight, n = 6) # Extract year from movie title
movies$title = sapply(as.character(movies$title), substrLeft, n = 7)
## Training and test split
single_rated_movies_indeces = which(sapply(X = unique(ratings$movieId), FUN = function(j) { sum(ratings$movieId == j) }) == 1)
adjusted_ratings = ratings[-c(which(ratings$movieId %in% unique(ratings$movieId)[single_rated_movies_indeces])),]
indeces = unlist(sapply(X = unique(adjusted_ratings$movieId), FUN = function(j) { set.seed(12345); sample(which(adjusted_ratings$movieId == j), size = floor(sum(adjusted_ratings$movieId == j) * 0.8)) }))
train = rbind(adjusted_ratings[indeces,], ratings[which(ratings$movieId %in% unique(ratings$movieId)[single_rated_movies_indeces]),])
test = ratings[-indeces,]
R = acast(train, userId ~ movieId, value.var = "rating")
R_temp = acast(test, userId ~ movieId, value.var = "rating")
R_test = matrix(NA, ncol = ncol(R), nrow = nrow(R), dimnames = list(rownames(R), colnames(R)))
R_test[, colnames(R_temp)] = R_temp
####################################################################################################

### 3. KNN
## 3.1 Item-Based
W_items = Pearson_Items(R) # Get similarity between items
W_items[is.na(W_items)] = 0 # Create symmetric matrix from lower triangle
W_items = W_items + t(W_items) # Matrix of similarities between corated items. Also known as pearson correlation (similarity measure)
diag(W_items) = diag(W_items) - 1 # We dont want to sum up the diagonal since it contains elements
rownames(W_items) = colnames(R); colnames(W_items) = colnames(R)
Get_Recommendations = function(userId, W) { # Only predict ratings on items which have more than 1 neighbour (predicting on these items would cause the predcition to simply be the rating provided by the neighbour)
  rated = colnames(R)[!is.na(R[userId,])]
  R_hat = numeric(ncol(R))
  for (j in 1:ncol(R)) {
    similarities = W[which(rownames(W) == colnames(R)[j]),]
    similarities = ifelse(similarities < 0, 0, similarities)
    similarities = sort(similarities, decreasing = TRUE)[-1] # Ignore the similarity of item j with itself
    if (sum(similarities > 0) <= 1) {
      R_hat[j] = NA
      break
    } else {
      similarities_sum = sum(similarities[rated], na.rm = TRUE)
      numerator = similarities * R[userId, names(similarities)]
      R_hat[j] = sum(numerator, na.rm = TRUE)/similarities_sum
    }
  }
  names(R_hat) = colnames(R)
  return(R_hat)
}
# Implementation
R_hat_item = matrix(ncol = ncol(R), nrow = nrow(R))
rownames(R_hat_item) = rownames(R); colnames(R_hat_item) = colnames(R)
cl = makeCluster(detectCores() - 1); registerDoParallel(cl)
R_hat_item = foreach (i = 1:nrow(R), .combine = rbind, .export = "Item_Based_Recommendation") %dopar% { # Loop over users
  Item_Based_Recommendation(userId = i, W = W_items, R = R[i,])
}
stopCluster(cl) # Or R_hat_item = mclapply(1:nrow(R), Item_Based_Recommendation, mc.cores = detectCores() - 1, W = W_items, R = R[i,]); for (i in 1:nrow(R)) { R_hat_item[i,] = Item_Based_Recommendation(userId = i, W = W_items, R = R[i,]); print(i) }
mean_absolute_error_item = MAE(R_test, R_hat_item)
root_mean_squared_error_item = RMSE(R_test, R_hat_item)
item_based_recommendations = as.numeric(names(sort(Get_Recommendations(userId = 10, W = W_items)[is.na(R[10,])], decreasing = TRUE))) # Get movie recommendations for user userId = 10
rbind(movies[which(movies$movieId == item_based_recommendations[1]),],
      movies[which(movies$movieId == item_based_recommendations[2]),],
      movies[which(movies$movieId == item_based_recommendations[3]),],
      movies[which(movies$movieId == item_based_recommendations[4]),],
      movies[which(movies$movieId == item_based_recommendations[5]),],
      movies[which(movies$movieId == item_based_recommendations[6]),],
      movies[which(movies$movieId == item_based_recommendations[7]),],
      movies[which(movies$movieId == item_based_recommendations[8]),],
      movies[which(movies$movieId == item_based_recommendations[9]),],
      movies[which(movies$movieId == item_based_recommendations[10]),]) # Print the nonrated movies for userId which had the highest predicted ratings based on the KNN algorithm
movies$title[movies$movieId %in% as.numeric(names(R[10, !is.na(R[10,])]))] # Movies watched by userId = 10

## 3.2 User-Based
W_users = Pearson_Users(R) # Get similarity between users
W_users[is.na(W_users)] = 0 # Create symmetric matrix from lower triangle
W_users = W_users + t(W_users) # Matrix of similarities between users with corated items. Also known as pearson correlation (similarity measure)
diag(W_users) = diag(W_users) - 1 # We dont want to sum up the diagonal since it contains elements
rownames(W_users) = rownames(R); colnames(W_users) = rownames(R)
# Implementation
R_hat_user = matrix(ncol = ncol(R), nrow = nrow(R))
rownames(R_hat_user) = rownames(R); colnames(R_hat_user) = colnames(R)
cl = makeCluster(detectCores() - 1); registerDoParallel(cl)
R_hat_user = foreach (i = 1:nrow(R), .combine = rbind, .export = "User_Based_Recommendation") %dopar% { # Loop over users
  User_Based_Recommendation(userId = i, W = W_users, k = 50, R = R)
}
stopCluster(cl) # Or R_hat_user = mclapply(1:nrow(R), User_Based_Recommendation, mc.cores = detectCores() - 1, W = W_users, k = 50, R = R); for (i in 1:nrow(R)) { R_hat_user[i,] = User_Based_Recommendation(userId = i, W = W_users, k = 50, R = R); print(i) }
mean_absolute_error_user = MAE(R_test, R_hat_user)
root_mean_squared_error_user = RMSE(R_test, R_hat_user)
user_based_recommendations = as.numeric(names(sort(User_Based_Recommendation(userId = 10, W = W_users, k = 50, R = R)[is.na(R[10,])], decreasing = TRUE))) # Get recommendations for user userId = 10
rbind(movies[which(movies$movieId == user_based_recommendations[1]),],
      movies[which(movies$movieId == user_based_recommendations[2]),],
      movies[which(movies$movieId == user_based_recommendations[3]),],
      movies[which(movies$movieId == user_based_recommendations[4]),],
      movies[which(movies$movieId == user_based_recommendations[5]),],
      movies[which(movies$movieId == user_based_recommendations[6]),],
      movies[which(movies$movieId == user_based_recommendations[7]),],
      movies[which(movies$movieId == user_based_recommendations[8]),],
      movies[which(movies$movieId == user_based_recommendations[9]),],
      movies[which(movies$movieId == user_based_recommendations[10]),]) # Print the nonrated movies for userId which had the highest predicted ratings based on the KNN algorithm
movies$title[movies$movieId %in% as.numeric(names(R[10, !is.na(R[10,])]))] # Movies watched by user userId = 10
####################################################################################################

### 4. KNN with PCA
R_pca_users = R
R_pca_users[is.na(R_pca_users)] = 0
R_pca_users = prcomp(R_pca_users) # We want to keep all the users but reduce the item dimension
R_pca_users = R_pca_users$x[, 1:300]
W_users_pca = cor(t(R_pca_users)) # Get similarity between items
rownames(W_users_pca) = rownames(R); colnames(W_users_pca) = rownames(R)
R_pca_items = R
R_pca_items[is.na(R_pca_items)] = 0
R_pca_items = prcomp(t(R_pca_items)) # We want to keep all the items but reduce the user dimension
R_pca_items = R_pca_items$x[, 1:300]
W_items_pca = cor(t(R_pca_items)) # Get similarity between users
rownames(W_items_pca) = colnames(R); colnames(W_items_pca) = colnames(R)
# Implementation
R_hat_item_pca = matrix(ncol = ncol(R), nrow = nrow(R))
rownames(R_hat_item_pca) = rownames(R); colnames(R_hat_item_pca) = colnames(R)
R_hat_user_pca = matrix(ncol = ncol(R), nrow = nrow(R))
rownames(R_hat_user_pca) = rownames(R); colnames(R_hat_user_pca) = colnames(R)
for (i in 1:nrow(R)) {
  R_hat_item_pca[i,] = Item_Based_Recommendation(userId = i, W = W_items_pca, R = R[i,])
  R_hat_user_pca[i,] = User_Based_Recommendation(userId = i, W = W_users_pca, k = nrow(R))
  print(i) # Counter
}
mean_absolute_error_item_pca = MAE(R_test, R_hat_item_pca)
root_mean_squared_error_item_pca = RMSE(R_test, R_hat_item_pca)
mean_absolute_error_user_pca = MAE(R_test, R_hat_user_pca)
root_mean_squared_error_user_pca = RMSE(R_test, R_hat_user_pca)
####################################################################################################

### 5. Save Workspace
rm()
save(list = c(""), file = paste0(getwd(), "/Data/KNN Workspace.RData"))
save.image(paste0(getwd(), "/Data/KNN Workspace.RData"))
####################################################################################################
