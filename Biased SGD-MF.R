# Title: Recommender Systems
# Authors: Ivan Jericevich & Yovna Junglee
# 1. Preliminaries
# 2. Functions
# 3. Testing
# 4. Data Preparation
# 5. Biased Stochastic Gradient Descent Matrix Factorisation
# 6. Save Workspace
####################################################################################################

### 1. Preliminaries
cat("\014") # Clear console
rm(list = ls()) # Clear environment
if(!is.null(dev.list())) dev.off() # Clear plots
gc() # Garbage collection to get extra ram for large matrix
setwd("")
# load("Biased SGD-MF Workspace.RData") # Load data from previous workspace
list_of_packages = c("reshape2") # This automatically installs and loads packages not already installed on the users computer
new_packages = list_of_packages[!(list_of_packages %in% installed.packages()[, "Package"])] # reshape2 = acast
if(length(new_packages) > 0) {install.packages(new_packages)}
lapply(list_of_packages, require, character.only = TRUE)
source(paste0(getwd(), "/Functions/Supplementary.R"))
set.seed(123)
####################################################################################################

### 2. Functions
gradient_function = function(r_ui, pvec, qvec, gamma, lambda, bi, bu, mu) {
  err = r_ui - (mu + bi + bu + sum(pvec * qvec))
  # Update vectors p and q
  new_pvec = NULL
  new_qvec = NULL
  new_pvec = pvec + gamma * (err * qvec - lambda * pvec)
  new_qvec = qvec + gamma * (err * pvec - lambda * qvec)
  new_bi = bi + gamma * (err - lambda * bi)
  new_bu = bu + gamma * (err - lambda * bu)
  return(list(new_pvec = new_pvec, new_qvec = new_qvec, new_bi = new_bi, new_bu = new_bu))
}
# Evaluates the regularized cost functions
loss = function(user.item, predicted, pmat, qmat, p = 2, lambda, bitem, buser) {
  # Regularization for pmat
  p_norm = sqrt(apply(pmat^p, 1, sum))
  q_norm = sqrt(apply(qmat^p, 2, sum))
  # Calculate squared error loss
  error = 0.5 * (sum((user.item - predicted)^2) + lambda * (sum(p_norm)) + lambda * (sum(q_norm)) + lambda * (sum(bitem^2)) + lambda * (sum(buser^2)))
  return(error)
}

# Updates the matrix for one epoch
matrix_update = function(user.item, nusers, nitems, gamma, lambda, pmat, qmat, p, mu, bitem, buser, mu_mat) {
  for (i in 1:nusers) {
    for (j in 1:nitems) {
      # Only update for non-zero ratings zero entries remain zero
      if(user.item[i, j] != 0) {
        # Update the values of entries in the user-feature and item-feature matrix
        updated_values = gradient_function(user.item[i, j], pvec = pmat[i,], qvec = qmat[, j], gamma = gamma, lambda = lambda, mu = mu, bi = bitem[j], bu = buser[i])
        pmat[i,] = updated_values$new_pvec
        qmat[,j] = updated_values$new_qvec
        bitem[j] = updated_values$new_bi
        buser[i] = updated_values$new_bu
      }
    }
  }
  predicted = pmat %*% qmat + mu_mat + matrix(rep(bitem, nusers), ncol = nitems, nrow = nusers, byrow = T) + matrix(rep(buser, nitems), ncol = nitems, nrow = nusers, byrow = F)
  zero_entries = which(user.item == 0, arr.ind = T)
  predicted[zero_entries] = 0
  loss_value = loss(user.item, predicted, pmat = pmat, qmat = qmat, p = p, lambda = lambda, bitem = bitem, buser = buser)
  return(list(predicted = predicted, loss_value = loss_value, pmat = pmat, qmat = qmat, bitem = bitem, buser = buser))
}
## Biased SGD-MF
matrix_factorization_bias = function(user.item, gamma, lambda, kfeatures, p = 2, tolerance = 0.001, max_iter = 20000) {
  # Get number of users and items
  nitems = ncol(user.item)
  nusers = nrow(user.item)
  # Initialise values
  init_mat = init(nitems, nusers, kfeatures)
  pmat = init_mat$pvec_init
  qmat = init_mat$qvec_init
  bitem = init_mat$bitem_init
  buser = init_mat$buser_init
  # Get overall mean
  rated = which(user.item > 0, arr.ind = T)
  mu = mean(user.item[rated])
  mu_mat = matrix(0, ncol = nitems, nrow = nusers)
  mu_mat[rated] = mu
  # Get predictions based on actual values
  predicted = pmat %*% qmat + mu_mat + matrix(rep(bitem, nusers), ncol = nitems, nrow = nusers, byrow = T) + matrix(rep(buser, nitems), ncol = nitems, nrow = nusers, byrow = F)
  zero_entries = which(user.item == 0, arr.ind = T)
  predicted[zero_entries] = 0
  current_loss = 0
  mse_record = NULL
  # Number of iterations
  i = 0
  final = list()
  eps = 10000
  while(eps > tolerance && i < max_iter) {
    new_mat = matrix_update(user.item,nusers, nitems, gamma, lambda, pmat, qmat, p, bitem = bitem, buser = buser, mu = mu, mu_mat = mu_mat)
    predicted = new_mat$predicted
    pmat = new_mat$pmat
    qmat = new_mat$qmat
    bitem =  new_mat$bitem
    buser = new_mat$buser
    loss_value = new_mat$loss_value
    final = new_mat
    eps = abs(loss_value - current_loss)
    current_loss = loss_value
    MSE = sum((predicted - user.item)^2) / (nitems * nusers)
    mse_record[i] = MSE
    print(paste(i, eps, current_loss, MSE))
    i = i + 1
  }
  MSE = sum((final$predicted - user.item)^2) / (nitems * nusers)
  return(list(Predicted = final$predicted, user.feature = final$pmat, item.feature = final$qmat, bitem = final$bitem, buser = final$buser, mu = mu, penalised_error = final$loss_value, MSE = MSE, niter = i, trace = mse_record))
}
####################################################################################################

### 3. Testing
nusers_test = 10
nitems_test = 10
kfeatures_test = 4
lambda_test = 0.0025
gamma_test = 0.005
user.item_test = matrix(sample(0:5, size = nitems_test * nusers_test, replace = T), nrow = nitems_test, ncol = nusers_test)
final.test = matrix_factorization_bias(user.item_test, gamma_test, lambda_test, kfeatures = 2)
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

### 5. Biased Stochastic Gradient Descent Matrix Factorisation
t1 = Sys.time()
lambda = 0.002
gamma = 0.005
mod1 = matrix_factorization_SGD(R, gamma, lambda, kfeatures = 2, tolerance = 0.001, max_iter = 2500)
t2 = Sys.time()
## Visualisation
# Plot MSE trace
ggplot(data.frame(cbind(epoch = 1:(mod1b$niter - 1), trace = mod1b$trace)), aes(x = epoch, y = trace)) +
  geom_line(col = "blue", alpha = .3) +
  geom_point(col = "blue", alpha = .3) +
  ylab("Mean square error") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"), axis.text = element_text(size = 18), axis.title = element_text(size = 18))
# Plot features 
item.feature = data.frame(t(mod1$item.feature))
colnames(item.feature) = c("F1", "F2")
user.feature = data.frame(mod1$user.feature)
colnames(user.feature) = c("F1", "F2")
# Outliers in the data
find_outliers = c(which(mod1$bitem > 3.5), which(mod1$bitem < (-5)))
ggplot(item.feature, aes(x = F1, y = F2)) +
  geom_point(col="orange") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"), axis.text = element_text(size = 20), axis.title = element_text(size = 20)) +
  geom_text_repel(data = item.feature[find_outliers,], label = movies$title[find_outliers], col = "black", size = 6)
ggplot(user.feature, aes(x = F1, y = F2)) +
  geom_point(col = "darkgreen") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"), axis.text = element_text(size = 20), axis.title = element_text(size = 20))
ggplot(as.data.frame(cbind(id = 1:9742, b = mod1$bitem))[which(mod1$bitem > -5),], aes(x = id, y = b, col = b)) +
  geom_point() +
  scale_colour_gradient2(low = "red", high = "red", mid = "grey69", midpoint = 0) +
  geom_point(aes(x = which(mod1$bitem == min(mod1$bitem)), y = min(mod1$bitem)), col = "red") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"), axis.text = element_text(size = 20), axis.title = element_text(size = 20), legend.position = "none") +
  geom_abline(intercept = 0,slope = 0, col = "black", linetype = "dashed") +
  geom_text_repel(data = as.data.frame(cbind(id = 1:9742, b = mod1$bitem))[find_outliers,], label = movies$title[find_outliers], col = "black", size = 6) +
  labs(x = "movieID", y = "Item effect")
ggplot(as.data.frame(cbind(id = 1:610, b = mod1$buser)), aes(x = id, y = b, col = b)) +
  geom_point() +
  scale_colour_gradient2(low = "red", high = "red", mid = "grey69", midpoint = 0) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = "black"), axis.text = element_text(size = 20), axis.title = element_text(size = 20), legend.position = "none") +
  geom_abline(intercept = 0, slope = 0, col = "black", linetype = "dashed") +
  labs(x = "userID", y = "User effect")
## Performance metrics
rated.test = which(R_test > 0, arr.ind = T) # Obtain indices for rated items in test set
MAE.mf = NULL; SSE.mf = NULL
n_movies = nrow(rated.test)
for (obs in 1:nrow(rated.test)) {
  ind = rated.test[obs,]
  actual = R_test[ind[1], ind[2]]
  pred = mod1$user.feature[ind[1],] %*% mod1$item.feature[, ind[2]] + mod1$mu + mod1$bitem[ind[2]] + mod1$buser[ind[1]]
  MAE.mf[obs] = mean(abs(actual - pred))
  SSE.mf[obs] = (actual - pred)^2
}
mae.val = mean(MAE.mf, na.rm = T); mae.val
RMSE.val = sqrt(sum(SSE.mf) / n_movies); RMSE.val
movies$title. = as.character(movies$title)
predict_userID(mod1, 10, 10, R)
####################################################################################################

### 6. Save Workspace
rm()
save(list = c(""), file = paste0(getwd(), "/Data/Biased SGD-MF Workspace.RData"))
save.image(paste0(getwd(), "/Data/Biased SGD-MF Workspace.RData"))
####################################################################################################