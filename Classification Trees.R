# Title: Recommender Systems
# Authors: Ivan Jericevich & Yovna Junglee
# 1. Preliminaries
# 2. Data Preparation
# 3. Classification Trees
# 4. Save Workspace
####################################################################################################

### 1. Preliminaries
cat("\014") # Clear console
gc() # Garbage collection to get extra ram for large matrix
rm(list = ls()) # Clear environment
if(!is.null(dev.list())) dev.off() # Clear plots
setwd("")
# load("Tree Workspace.RData") # Load data from previous workspace
list_of_packages = c("reshape2", "tree") # This automatically installs and loads packages not already installed on the users computer
new_packages = list_of_packages[!(list_of_packages %in% installed.packages()[, "Package"])] # reshape2 = acast; tree = tree
if(length(new_packages) > 0) {install.packages(new_packages)}
lapply(list_of_packages, require, character.only = TRUE)
source(paste0(getwd(), "/Functions/Supplementary.R"))
####################################################################################################

### 2. Data Preparation
ratings = read.csv(file = paste0(getwd(), "/Data/ratings.csv"), header = TRUE)[, -4] # Sorted according to userId
movies = merge(read.csv(file = paste0(getwd(), "/Data/movies.csv"), header = TRUE), aggregate(tag ~ movieId, data = read.csv(file = paste0(getwd(), "/Data/tags.csv"), header = TRUE)[, -c(1, 4)], paste, collapse = "|"), all = TRUE) # Merge movies file with the tags file; sorted according to movieId
movies$title = sapply(movies$title, trimws)
indeces = which(is.na(as.numeric(sapply(as.character(movies$title), substrRight, n = 4))))
movies$title[indeces] = paste(movies$title[indeces], c("(1993)", "(2018)", "(2015)", "(1979)", "(2016)", "(2016)", "(2016)", "(2016)", "(2014)", "(2017)", "(2017)", "(2011)"), sep = " ")
movies$year = as.numeric(sapply(as.character(movies$title), substrRight, n = 4)) # Extract year from movie title
movies$title = sapply(as.character(movies$title), substrLeft, n = 7)
movies$views = sapply(movies$movieId, function(i) {sum(ratings$movieId == i, na.rm = TRUE)})
genre = character()
year = numeric()
for (i in 1:nrow(ratings)) {
  genre[i] = as.character(movies$genres[which(movies$movieId == ratings$movieId[i])])
  year[i] = movies$year[which(movies$movieId == ratings$movieId[i])]
}
genre_names = unique(unlist(strsplit(as.character(genre), split = "|", fixed = TRUE)))[-20]
dummy_years = data.frame(y1900s = factor(as.numeric(year >= 1900 & year < 1910)),
                         y1910s = factor(as.numeric(year >= 1910 & year < 1920)),
                         y1920s = factor(as.numeric(year >= 1920 & year < 1930)),
                         y1930s = factor(as.numeric(year >= 1930 & year < 1940)),
                         y1940s = factor(as.numeric(year >= 1940 & year < 1950)),
                         y1950s = factor(as.numeric(year >= 1950 & year < 1960)),
                         y1960s = factor(as.numeric(year >= 1960 & year < 1970)),
                         y1970s = factor(as.numeric(year >= 1970 & year < 1980)),
                         y1980s = factor(as.numeric(year >= 1980 & year < 1990)),
                         y1990s = factor(as.numeric(year >= 1990 & year < 2000)),
                         y2000s = factor(as.numeric(year >= 2000 & year < 2010)),
                         y2010s = factor(as.numeric(year >= 2010 & year < 2020)))
ratings = cbind(ratings, dummy_years, setNames(data.frame(lapply(genre_names, function(i) factor(as.integer(grepl(i, genre))))), genre_names))
colnames(ratings)[30] = "Sci_Fi"
colnames(ratings)[32] = "Film_Noir"
ratings$views = sapply(ratings$movieId, function(i) {movies$views[which(movies$movieId == i)]})
ratings$rating = factor(ratings$rating, levels = c(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5), ordered = TRUE)
movies = merge(read.csv(file = "movies.csv", header = TRUE), aggregate(tag ~ movieId, data = read.csv(file = "tags.csv", header = TRUE)[, -c(1, 4)], paste, collapse = "|"), all = TRUE) # Merge movies file with the tags file; sorted according to movieId
movies$year = sapply(as.character(movies$title), substrRight, n = 6) # Extract year from movie title
movies$title = sapply(as.character(movies$title), substrLeft, n = 7)
## Training and test split
single_rated_movies_indeces = which(sapply(X = unique(ratings$movieId), FUN = function(j) { sum(ratings$movieId == j) }) == 1)
adjusted_ratings = ratings[-c(which(ratings$movieId %in% unique(ratings$movieId)[single_rated_movies_indeces])),]
indeces = unlist(sapply(X = unique(adjusted_ratings$movieId), FUN = function(j) { set.seed(12345); sample(which(adjusted_ratings$movieId == j), size = floor(sum(adjusted_ratings$movieId == j) * 0.8)) }))
train = rbind(adjusted_ratings[indeces,], ratings[which(ratings$movieId %in% unique(ratings$movieId)[single_rated_movies_indeces]),])
test = ratings[-indeces,]
####################################################################################################

### 3. Classification Trees
predicted = numeric(nrow(test))
for (i in unique(train$userId)) {
  full_tree = tree(rating ~ . - movieId, data = train[which(train$userId == i),]) # Build a full tree for each user
  y = predict(full_tree, newdata = test[which(test$userId == i),], type = "class")
  predicted[which(test$userId == i)] = as.numeric(levels(y))[y]
  print(i) # Counter
}
MAE(as.numeric(levels(test$rating))[test$rating], predicted)
RMSE(as.numeric(levels(test$rating))[test$rating], predicted)
####################################################################################################

### 4. Save Workspace
rm()
save(list = c(""), file = paste0(getwd(), "/Data/Tree Workspace.RData"))
save.image(paste0(getwd(), "/Data/Tree Workspace.RData"))
####################################################################################################
