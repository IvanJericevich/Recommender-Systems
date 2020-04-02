substrRight = function(x, n){ # Function to get the last n characters in a string
  substr(x, nchar(x) - n + 1, nchar(x))
}
substrLeft = function(x, n){ # Function to exclude the last n characters in a string
  substr(x, 1, nchar(x) - n)
}
MAE = function(actual, predicted) {
  mean(abs(actual - predicted), na.rm = TRUE)
}
RMSE = function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}
## Matrix Factorisation
# Initialise vectors
init = function(nitems, nusers, kfeatures) {
  ratings = seq(0, 1, by = 0.1)
  # Initialise user-feature matrix
  pvec_init = sample(ratings, size = nusers * kfeatures, replace = T)
  pvec_init = matrix(pvec_init, ncol = kfeatures)
  # Initialise item-feature matrix
  qvec_init = sample(ratings, size = nitems * kfeatures, replace = T)
  qvec_init = matrix(qvec_init, nrow = kfeatures)
  return(list(pvec_init = pvec_init, qvec_init = qvec_init))
}
# Provide recommendations for userID
predict_userID = function(model, userID, n, user.item) {
  rm_not_rated = which(colSums(user.item) == 0)
  not_watched = which(user.item[userID,] == 0)
  preds = model$user.feature %*% model$item.feature
  pred_for_id = data.frame(cbind(id.ord = 1:ncol(user.item), preds[userID,]))
  pred_for_id = pred_for_id[-rm_not_rated,]
  ordered_ratings = pred_for_id[order(pred_for_id$V2, decreasing = T),]
  top_movies = movies[as.numeric(ordered_ratings$id.ord),]$title[1:n]
  return(top_movies)
}