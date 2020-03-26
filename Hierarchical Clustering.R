# Title: Recommender Systems
# Authors: Ivan Jericevich
# 1. Preliminaries
# 2. Data Preparation
# 3. Hierarchical Clustering
#   - PCA
#   - SVD
# 4. Save Workspace
####################################################################################################

### 1. Preliminaries
cat("\014") # Clear console
gc() # Garbage collection to get extra ram for large matrix
rm(list = ls()) # Clear environment
if(!is.null(dev.list())) dev.off() # Clear plots
setwd("")
# load("Hierarchical Workspace.RData") # Load data from previous workspace
list_of_packages = c("reshape2", "dendextend") # This automatically installs and loads packages not already installed on the users computer
new_packages = list_of_packages[!(list_of_packages %in% installed.packages()[, "Package"])] # reshape2 = acast; dendextend = color_branches
if(length(new_packages) > 0) {install.packages(new_packages)}
lapply(list_of_packages, require, character.only = TRUE)
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

### 3. Hierarchical Clustering
## PCA
pca = R; pca[is.na(pca)] = 0
pca = prcomp(pca) # We want to keep all the users but reduce the item dimension
plot(cumsum(pca$sdev^2 / sum(pca$sdev^2)), xlab = "PC Number/Singular Values", ylab = "Amount of explained variance", main = "Cumulative variance plot") # Same as SVD scree plot
abline(v = 166, col = "blue", lty = 5)
abline(h = 0.8007579, col = "blue", lty = 5)
abline(v = 258, col = "red", lty = 5)
abline(h = 0.9000746, col = "red", lty = 5)
legend("bottomright", legend = c("Cut-off @ PC166", "Cut-off @ PC258"), col = c("blue", "red"), lty = 5, cex = 1)
R_pca = pca$x[, 1:300]
W_pca = dist(R_pca)
clusters_pca = hclust(W_pca)
clustered_users_pca = cutree(clusters_pca, h = 75)
plot(color_branches(clusters_pca, h = 75), leaflab = "none")
W_pca = as.matrix(W_pca)
rownames(W_pca) = colnames(W_pca) = rownames(R)
## SVD
s = R; s[is.na(s)] = 0
s = svd(s)
R_svd = s$u %*% diag(s$d) %*% diag(c(rep(1, 300), rep(0, 310))) %*% t(s$v) # R = U D V'
rownames(R_svd) = rownames(R); colnames(R_svd) = colnames(R)
W_svd = dist(R_svd)
clusters_svd = hclust(W_svd)
plot(color_branches(clusters_svd, h = 75), leaflab = "none")
clustered_users_svd = cutree(clusters_svd, h = 100) # Increasing h reduces accuracy
W_svd = as.matrix(W_svd)
rownames(W_svd) = colnames(W_svd) = rownames(R)
## Recommendation Function
User_Based_Recommendation = function(userId, clustered_users, W) { # Problems: some clusters only contain one user, some clusters dont have users which have also rated the target user's movie j
  userId_cluster = clustered_users[which(clustered_users == clustered_users[userId])]
  userId_cluster = userId_cluster[-which(names(userId_cluster) == userId)]
  R_hat = numeric(ncol(R)) # Initialise predicted ratings on movies by user UserId
  names(R_hat) = colnames(R)
  for (j in 1:ncol(R)) { # Loop through movies
    if (length(userId_cluster) == 0) {
      R_hat = rep(NA, ncol(R))
      break
    }
    ratings_sum = 0
    n = 0
    for (i in 1:length(userId_cluster)) {
      if (!is.na(R[names(userId_cluster)[i], j])) {
        ratings_sum = ratings_sum + W[names(userId_cluster)[i], userId] * R[names(userId_cluster)[i], j]
        n = n + W[names(userId_cluster)[i], userId]
      }
    }
    R_hat[j] = ifelse(n == 0, NA, ratings_sum/n)
  }
  return(R_hat)
}
## Implementation
R_hat_svd = matrix(ncol = ncol(R), nrow = nrow(R))
R_hat_pca = matrix(ncol = ncol(R), nrow = nrow(R))
for (i in 1:nrow(R)) {
  R_hat_pca[i,] = User_Based_Recommendation(i, clustered_users_pca, W_pca)
  R_hat_svd[i,] = User_Based_Recommendation(i, clustered_users_svd, W_svd)
  print(i)
}
# Performance Metrics
mean_absolute_error_svd = MAE(R_test, R_hat_svd)
root_mean_squared_error_svd = RMSE(R_test, R_hat_svd)
mean_absolute_error_pca = MAE(R_test, R_hat_pca)
root_mean_squared_error_pca = RMSE(R_test, R_hat_pca)
####################################################################################################

### 4. Save Workspace
rm()
save(list = c(""), file = paste0(getwd(), "/Data/Hierarchical Workspace.RData"))
save.image(paste0(getwd(), "/Data/Hierarchical Workspace.RData"))
####################################################################################################