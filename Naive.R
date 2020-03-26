# Title: Recommender Systems
# Authors: Ivan Jericevich
# 1. Preliminaries
# 2. Data Preparation
# 3. Naive
# 4. Save Workspace
####################################################################################################

### 1. Preliminaries
cat("\014") # Clear console
gc() # Garbage collection to get extra ram for large matrix
rm(list = ls()) # Clear environment
if(!is.null(dev.list())) dev.off() # Clear plots
setwd("")
# load("Naive Workspace.RData") # Load data from previous workspace
list_of_packages = c("reshape2") # This automatically installs and loads packages not already installed on the users computer
new_packages = list_of_packages[!(list_of_packages %in% installed.packages()[, "Package"])] # reshape2 = acast
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

### 3. Naive Method
naive_predictions = matrix(apply(R, 1, mean, na.rm = TRUE), nrow = nrow(R_test), ncol = ncol(R_test), byrow = FALSE)
colnames(naive_predictions) = colnames(R_test); rownames(naive_predictions) = rownames(R_test)
MAE(R_test, naive_predictions)
RMSE(R_test, naive_predictions)
####################################################################################################

### 4. Save Workspace
rm()
save(list = c(""), file = paste0(getwd(), "/Data/Naive Workspace.RData"))
save.image(paste0(getwd(), "/Data/Naive Workspace.RData"))
####################################################################################################