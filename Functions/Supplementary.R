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