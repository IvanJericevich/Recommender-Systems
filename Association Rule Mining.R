# Title: Recommender Systems
# Authors: Ivan Jericevich & Yovna Junglee
# 1. Preliminaries
# 2. Data Preparation
# 3. Association Rule Mining
# 4. Save Workspace
####################################################################################################

### 1. Preliminaries
cat("\014") # Clear console
gc() # Garbage collection to get extra ram for large matrix
rm(list = ls()) # Clear environment
if(!is.null(dev.list())) dev.off() # Clear plots
setwd("")
# load("ARM Workspace.RData") # Load data from previous workspace
list_of_packages = c("reshape2", "arules") # This automatically installs and loads packages not already installed on the users computer
new_packages = list_of_packages[!(list_of_packages %in% installed.packages()[, "Package"])] # reshape2 = acast; arules = apriori
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
####################################################################################################

### 3. Association Rule Mining
R = acast(adjusted_ratings, userId ~ movieId, value.var = "rating"); R = ifelse(is.na(R), 0, 1)
trans = as(R, "transactions")
rules = apriori(trans, parameter = list(supp = 0.01, target = "rules", minlen = 2, maxlen = 2, maxtime = 0))
write(rules, file = paste0(getwd(), "/Data/Association Rules.csv"), sep = ",", quote = TRUE, row.names = FALSE)
rules = as(rules, "data.frame"); rules$rules = as.character(rules$rules)
lhs_rhs = sapply(rules$rules, strsplit, split = " => ")
lhs = character(nrow(rules)); rhs = character(nrow(rules))
for (i in 1:nrow(rules)) {
  lhs[i] = substr(lhs_rhs[[i]][1], start = 2, stop = nchar(lhs_rhs[[i]][1]) - 1)
  rhs[i] = substr(lhs_rhs[[i]][2], start = 2, stop = nchar(lhs_rhs[[i]][2]) - 1)
}
rules = cbind(lhs = lhs, rhs = rhs, rules[, -1])
rules$lhs = as.character(rules$lhs); rules$rhs = as.character(rules$rhs)
Recommend = function(userId) { # Recommend movies for user userId based on association rules
  rated_lhs = rules[which(rules$lhs %in% colnames(R)[which(R[userId,] != 0)]),]
  rated_lhs_nonrated_rhs = rated_lhs[which(rated_lhs$rhs %in% colnames(R)[which(R[userId,] == 0)]),]
  rated_lhs_nonrated_rhs[order(rated_lhs_nonrated_rhs$confidence, decreasing = TRUE),]
}
rules_conf = Recommend(10)
movies$title[movies$movieId %in% rules_conf$rhs]
for (i in 1:20) { # Print the top 20 movies with the highest confidence measure
  print(movies$title[which(movies$movieId == rules_conf$rhs[i])])
}
####################################################################################################

### 4. Save Workspace
rm()
save(list = c(""), file = paste0(getwd(), "/Data/ARM Workspace.RData"))
save.image(paste0(getwd(), "/Data/ARM Workspace.RData"))
####################################################################################################
