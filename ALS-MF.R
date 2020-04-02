# Title: Recommender Systems
# Authors: Ivan Jericevich & Yovna Junglee
# 1. Preliminaries
# 2. Functions
# 3. Testing
# 4. Data Preparation
# 5. Alternating Least Squares Matrix Factorisation
# 6. Save Workspace
####################################################################################################

### 1. Preliminaries
cat("\014") # Clear console
rm(list = ls()) # Clear environment
if(!is.null(dev.list())) dev.off() # Clear plots
gc() # Garbage collection to get extra ram for large matrix
setwd("")
# load("ALS-MF Workspace.RData") # Load data from previous workspace
list_of_packages = c("reshape2") # This automatically installs and loads packages not already installed on the users computer
new_packages = list_of_packages[!(list_of_packages %in% installed.packages()[, "Package"])] # reshape2 = acast
if(length(new_packages) > 0) {install.packages(new_packages)}
lapply(list_of_packages, require, character.only = TRUE)
source(paste0(getwd(), "/Functions/Supplementary.R"))
set.seed(123)
####################################################################################################

### 2. Functions
## Alternating least squares (fix p)  
# Find optimal q by fixing p
ALS.q = function(qvals, pmat, p = 2, lambda, user.item, kfeatures) {
  nitems = ncol(user.item)
  qmat = matrix(qvals, nrow = kfeatures, ncol = nitems, byrow = T)
  p_norm = sqrt(apply(pmat^p, 1, sum))
  q_norm = sqrt(apply(qmat^p, 2, sum))
  # Get predicted
  predicted = pmat %*% qmat
  zero_entries = which(user.item == 0, arr.ind = T)
  predicted[zero_entries] = 0
  # Calculate squared error loss
  error = 0.5 * (sum((user.item - predicted)^2) + lambda * (sum(p_norm)) + lambda * (sum(q_norm)))
  return(error)
}
## Alternating least squares (fix q)
# Find optimal p by fixing q
ALS.p = function(pvals, qmat, p = 2, lambda, user.item, kfeatures) {
  nusers = nrow(user.item)
  pmat = matrix(pvals, nrow = nusers, ncol = kfeatures, byrow = T)
  p_norm = sqrt(apply(pmat^p, 1, sum))
  q_norm = sqrt(apply(qmat^p, 2, sum))
  # Get predicted
  predicted = pmat %*% qmat
  zero_entries = which(user.item == 0, arr.ind = T)
  predicted[zero_entries] = 0
  # Calculate squared error loss
  error = 0.5 * (sum((user.item - predicted)^2) + lambda * (sum(p_norm)) + lambda * (sum(q_norm)))
  return(error)
}
## ALS-MF
matrix_factorization_ALS = function(user.item, lambda, kfeatures, p = 2, tolerance = 0.001, max_iter = 10000000, method) {
  # Get number of users and items
  nitems = ncol(user.item)
  nusers = nrow(user.item)
  kfeatures = kfeatures
  # Initialise values
  init_mat = init(nitems, nusers, kfeatures)
  pmat = init_mat$pvec_init
  qmat = init_mat$qvec_init
  predicted = pmat %*% qmat
  zero_entries = which(user.item == 0, arr.ind = T)
  predicted[zero_entries] = 0
  current_loss = 0
  # Number of iterations
  i = 0
  eps = 10000
  iter = 0
  pen.error = 0
  trace = NULL
  while (eps > tolerance && i < max_iter) {
    # Optimise loss wrt to q (fixing p)
    random.starts = seq(0.05, 2, by = .1)
    random.ends = seq(100, 105, by = .1)
    print(paste(i, " : Optimising loss wrt to q fixing p"))
    try = optim(f = ALS.q, p = sample(random.starts, nitems * kfeatures, replace = T),
                 # upper = rep(random.ends, nusers * nitems, replace = T),
                 pmat = pmat, lambda = lambda, user.item = user.item, kfeatures = kfeatures)
    i = i + na.omit(as.vector(try$counts))[1]
    iter = iter + 1
    # Update values
    qmat = matrix(try$par, nrow = kfeatures, ncol = nitems, byrow = T)
    # Record current MSE
    predicted = pmat %*% qmat
    zero_entries = which(user.item == 0, arr.ind = T)
    predicted[zero_entries] = 0
    trace[iter] = sum((predicted - user.item)^2) / (nitems * nusers)
    pen.error = try$value
    eps.p = abs(current_loss - try$value)
    current_loss = try$value
    print(paste("q ", eps.p))
    # Only Optimise loss wrt to p (fixing q) if current eps not good enough
    if (eps.p > tolerance) {
      print(paste(i, " : Optimising loss wrt to p fixing q"))
      random.starts = seq(0.05, 2, by = .1)
      random.ends = seq(100, 105, by = .1)
      try.p = optim(f = ALS.p, p = sample(random.starts, nusers * kfeatures, replace = T),
                     # upper = rep(random.ends, nusers*nitems, replace = T),
                     qmat = qmat, lambda = lambda, user.item = user.item, kfeatures = kfeatures)
      # Update values
      pmat = matrix(try.p$par, nrow = nusers, ncol = kfeatures, byrow = T)
      eps = abs(current_loss - try.p$value)
      current_loss = try.p$value
      pen.error = try.p$value
      i = i + na.omit(as.vector(try.p$counts))[1]
      iter = iter + 1
      # Record current MSE
      predicted = pmat %*% qmat
      zero_entries = which(user.item == 0, arr.ind = T)
      predicted[zero_entries] = 0
      trace[iter] = sum((predicted - user.item)^2) / (nitems * nusers)
      print(paste("p ", eps))
    } else {
      eps = eps.p
    }
  }
  predicted = pmat %*% qmat
  zero_entries = which(user.item == 0, arr.ind = T)
  predicted[zero_entries] = 0
  MSE = sum((predicted - user.item)^2) / (nitems * nusers)
  return(list(user.feature = pmat, item.feature = qmat, Predicted = predicted, MSE = MSE, niter_optim = i, niter = iter, penalised_error = pen.error, trace = trace))
}
####################################################################################################

### 3. Testing
nusers_test = 2
nitems_test = 3
kfeatures_test = 2
lambda_test = 0.0025
gamma_test = 0.005
# Generate data
user.item_test = matrix(sample(0:5, size = nitems_test * nusers_test), nrow = nusers_test, ncol = nitems_test)
final.test = matrix_factorization_ALS(user.item_test, lambda = lambda_test, kfeatures = 2)
plot(final.test$item.feature[1,], final.test$item.feature[2,])
####################################################################################################


### 4. Data Preparation
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

### 5. Alternating Least Squares Matrix Factorisation
t1 = Sys.time()
lambda = 0.0020
mod1.als = matrix_factorization_ALS(R, lambda = lambda, kfeatures = 2, max_iter = 20000)
t2 = Sys.time()
## Performance measures
rated.test = which(R_test > 0, arr.ind = T) # Obtain indices for rated items in test set
MAE.mf = NULL; SSE.mf = NULL
n_movies = nrow(rated.test)
for (obs in 1:nrow(rated.test)) {
  ind = rated.test[obs,]
  actual = R_test[ind[1], ind[2]]
  pred = mod1$user.feature[ind[1],] %*% mod1$item.feature[, ind[2]]
  MAE.mf[obs] = mean(abs(actual - pred))
  SSE.mf[obs] = (actual - pred)^2
}
mae.val = mean(MAE.mf, na.rm = T); mae.val
RMSE.val = sqrt(sum(SSE.mf) / n_movies); RMSE.val
movies$title = as.character(movies$title)
predict_userID(mod1, 10, 10, R)
####################################################################################################


### 6. Save Workspace
rm()
save(list = c(""), file = paste0(getwd(), "/Data/ALS-MF Workspace.RData"))
save.image(paste0(getwd(), "/Data/ALS-MF Workspace.RData"))
####################################################################################################