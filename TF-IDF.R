# Title: Recommender Systems
# Authors: Ivan Jericevich
# 1. Preliminaries
# 2. Data Preparation
# 3. Exploratory Data Analysis
# 4. TF-IDF
# 5. Save Workspace
####################################################################################################

### 1. Preliminaries
cat("\014") # Clear console
rm(list = ls()) # Clear environment
gc() # Garbage collection to get extra ram for large matrix
if(!is.null(dev.list())) dev.off() # Clear plots
setwd("")
# load("TF-IDF Workspace.RData") # Load data from previous workspace
list_of_packages = c("tidytext", "dplyr", "ggplot2", "reshape2", "Rcpp") # This automatically installs and loads packages not already installed on the users computer
new_packages = list_of_packages[!(list_of_packages %in% installed.packages()[, "Package"])] # reshape2 = acast; tidytext = unnest_tokens; Rcpp = sourceCpp; dplyr = %>%; ggplot2 = ggplot
if(length(new_packages) > 0) {install.packages(new_packages)}
lapply(list_of_packages, require, character.only = TRUE)
sourceCpp(paste0(getwd(), "/Functions/Pearson Correlation.cpp"))
sourceCpp(paste0(getwd(), "/Functions/KNN.cpp"))
source(paste0(getwd(), "/Functions/Supplementary.R"))
####################################################################################################

### 2. Data Preparation
ratings = read.csv(file = "ratings.csv", header = TRUE)[, -4] # Sorted according to userId
metadata = read.csv("metadata.csv", header = TRUE)[, c(6, 10)]
metadata = metadata[metadata$id %in% ratings$movieId,] # Only keep metadata rows of movies which have been rated by atleast one user
metadata$overview = as.character(metadata$overview)
metadata = metadata[-which(metadata$overview == "No overview found."),] # Remove movies which have no overview
ratings = ratings[ratings$movieId %in% metadata$id,]
R = acast(ratings, userId ~ movieId, value.var = "rating")
data("stop_words")
metadata_tokenized = mutate(metadata, overview = gsub(x = overview, pattern = "[0-9]+|[[:punct:]]|\\(.*\\)", replacement = "")) %>% # Remove numbers and punctuation
  unnest_tokens(output = word, input = overview) %>% # Tokenize movie overview and remove stop words
  anti_join(stop_words)
####################################################################################################

### 3. Exploratory data analysis
## Word frequencies
word_frequency = metadata_tokenized %>%
  count(word, sort = TRUE)
word_frequency_per_id = metadata_tokenized %>% # Frequency of each word per movie
  count(id, word, sort = TRUE)
total_words_per_id = word_frequency_per_id %>% # Total number of words for each movie
  group_by(id) %>% 
  summarize(total = sum(n))
movie_words = left_join(word_frequency_per_id, total_words_per_id) # One row for each word-id combination
frequency_by_rank = movie_words %>% # Frequency of each word per movie ranked per movie
  group_by(id) %>% 
  mutate(rank = row_number(), `term frequency` = n/total) %>%
  group_by(id)
## Graphics
# Total word counts
metadata_tokenized %>% # Plot word counts
  count(word, sort = TRUE) %>%
  filter(n > 100) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col() +
  xlab(NULL) +
  coord_flip()
# Distribution of frequencies per movie
ggplot(movie_words[which(movie_words$id %in% movie_words$id[1:4]),], aes(n/total, fill = id)) + # Plot distribution of frequency of words per movie (i.e. how words were counted once, how mmany words were counted twice ect. for each movie)
  geom_histogram(show.legend = FALSE) +
  facet_wrap(~id, ncol = 2, scales = "free_y")
# Zipf's law: frequency inversely proportional to rank
frequency_by_rank %>%
  ggplot(aes(rank, `term frequency`, color = id)) + 
  geom_line(size = 1.1, alpha = 0.8, show.legend = FALSE) + 
  scale_x_log10() +
  scale_y_log10()
####################################################################################################

### 4. TF-IDF
movie_words = movie_words %>%
  bind_tf_idf(word, id, n)
TFIDF = acast(movie_words, word ~ id, value.var = "tf_idf")
W = Pearson_Items(TFIDF)
W[is.na(W)] = 0 # Create symmetric matrix from lower triangle
W = W + t(W) # Matrix of similarities between corated items. Also known as pearson correlation (similarity measure)
diag(W) = diag(W) - 1 # We dont want to sum up the diagonal since it contains elements
rownames(W) = colnames(W) = colnames(TFIDF)
# Implementation
R_hat = matrix(ncol = ncol(R), nrow = nrow(R))
rownames(R_hat) = rownames(R); colnames(R_hat) = colnames(R)
cl = makeCluster(detectCores() - 1); registerDoParallel(cl)
R_hat = foreach (i = 1:nrow(R), .combine = rbind, .export = "Item_Based_Recommendation") %dopar% { # Loop over users
  Item_Based_Recommendation(userId = i, W = W_items, R = R[i,])
}
stopCluster(cl) # Or R_hat_item = mclapply(1:nrow(R), Item_Based_Recommendation, mc.cores = detectCores() - 1, W = W_items, R = R[i,]); for (i in 1:nrow(R)) { R_hat_item[i,] = Item_Based_Recommendation(userId = i, W = W_items, R = R[i,]); print(i) }
mean_absolute_error_item = MAE(R, R_hat)
root_mean_squared_error_item = RMSE(R, R_hat)
## Exploration
movie_words %>% # High TF-IDF terms
  select(-total) %>%
  arrange(desc(tf_idf))
# Graphics
movie_words %>% # Visualise high TF-IDF words
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>% 
  group_by(id) %>% 
  top_n(15) %>% 
  ungroup() %>%
  ggplot(aes(word, tf_idf, fill = id)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf") +
  facet_wrap(~id, ncol = 2, scales = "free") +
  coord_flip()
####################################################################################################

### 5. Save Workspace
rm()
save(list = c(""), file = paste0(getwd(), "/Data/TF-IDF Workspace.RData"))
save.image(paste0(getwd(), "/Data/TF-IDF Workspace.RData"))
####################################################################################################