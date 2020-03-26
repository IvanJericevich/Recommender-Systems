#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::plugins("cpp11")]]

// [[Rcpp::export]]
NumericVector Item_Based_Recommendation(int userId, NumericMatrix W, NumericVector R) { // Predict ratings on any movies by userId which are similar to the movies they rated based on the item similarity matrix
    NumericVector R_hat(W.ncol()); // Initialise predicted ratings on all movies by user UserId
    R_hat.names() = R.names();
    for(int j = 0; j < R_hat.size(); j++) { // Loop through all movies
        NumericVector W_j = W(j, _); // Extract similarities between movie j and all other movies (excluding itself)
        W_j[j] = 0; // Dont use similarity of target item with itself in prediction
        NumericVector W_rated = W_j[!is_na(R)]; // Vector of movieId's rated by user userId
        for(int k = 0; k < W_rated.size(); k++) { // Ignore negative correlations
            if(W_rated[k] < 0) W_rated[k] = 0;
        }
        NumericVector similarities = W_rated; // Similarities of movie j will all movies rated by user userId
        NumericVector R_index = R[!is_na(R)]; // Rated movies by user userId
        NumericVector numerator = similarities * R_index; // Multiply similarities to movie j with ratings on all movies made by user userId. Similarity score between each movie rated by user userId and movie j multiplied with all userId's ratings and summed up. Divided by the total sum of similarities between movies rated by userId and movie j - this is then the predicted rating for nonrated movie j
        R_hat[j] = sum(numerator) / sum(similarities); // Vector of predicted ratings on all movies for user userId
        Rcout << j; // Counter
    }
    return R_hat;
}

NumericVector Sort(NumericVector x) { // Sort named vector
    IntegerVector index = seq_along(x) - 1;
    std::sort(index.begin(), index.end(), [&](int i, int j){return x[i] > x[j];});
    return x[index];
}

// [[Rcpp::export]]
NumericVector User_Based_Recommendation(int userId, NumericMatrix W, int k, NumericMatrix R) { // Predict ratings on any movies by userId based on the user similarity matrix
    NumericVector R_hat(R.ncol()); // Initialise predicted ratings on all movies by user UserId
    R_hat.names() = colnames(R);
    NumericVector similarities1 = W(userId - 1, _);
    similarities1.names() = rownames(R);
    NumericVector similarities = Sort(similarities1); // Only use top K similarities between user userId and all other users (excluding themself) (note that many of the ratings from each similar neighbour will be NA)
    similarities[0] = 0;
    NumericVector userId_mean(similarities.size());
    for(int i = 0; i < similarities.size(); i++) { // Loop trhough all users
        userId_mean[i] = mean(na_omit(R(i, _))); // Mean rating for each user
        if(i > k) similarities[i] = 0; // Only select the first k similarities
    }
    NumericVector R_similarities(similarities.size());
    for(int j = 0; j < R_hat.size(); j++) { // Loop through all movies
        R_similarities = R(_, j) - userId_mean; // Standardize ratings
        R_similarities.names() = rownames(R);
        R_similarities = R_similarities[as<CharacterVector>(similarities.names())]; // Order userId's ratings according to named vector similarities
        NumericVector numerator = similarities * R_similarities;
        R_hat[j] = userId_mean[userId - 1] + sum(na_omit(numerator))/sum(similarities);
        Rcout << j; // Counter
    }
    return R_hat;
}