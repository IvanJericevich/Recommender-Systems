#include <Rcpp.h>
using namespace Rcpp;
// [[Rcpp::plugins("cpp11")]]

// [[Rcpp::export]]
NumericMatrix Pearson_Items(NumericMatrix X) { // Similarity measure. Also known as Pearson correlation
    NumericMatrix W(X.ncol(), X.ncol());
    for(int i = 0; i < X.ncol(); i++) {
        for(int j = 0; j <= i; j++) {
            if(i == j) {
                W(i, j) = 1;
            } else {
                double x_bar = mean( na_omit(X(_, i)) );
                double y_bar = mean( na_omit(X(_, j)) );
                W(i, j) = sum(na_omit((X(_, i) - x_bar) * (X(_, j) - y_bar))) / (sqrt(sum(na_omit((X(_, i) - x_bar) * (X(_, i) - x_bar)))) * sqrt(sum(na_omit((X(_, j) - y_bar) * (X(_, j) - y_bar)))));
            }
        }
    }
    return W;
}

// [[Rcpp::export]]
NumericMatrix Pearson_Users(NumericMatrix X) { // Similarity measure. Also known as Pearson correlation
    NumericMatrix W(X.nrow(), X.nrow());
    for(int i = 0; i < X.nrow(); i++) {
        for(int j = 0; j <= i; j++) {
            if(i == j) {
                W(i, j) = 1;
            } else {
                double x_bar = mean(na_omit(X(i, _)));
                double y_bar = mean(na_omit(X(j, _)));
                W(i, j) = sum(na_omit((X(i, _) - x_bar) * (X(j, _) - y_bar))) / (sqrt(sum(na_omit((X(i, _) - x_bar) * (X(i, _) - x_bar)))) * sqrt(sum(na_omit((X(j, _) - y_bar) * (X(j, _) - y_bar))))); // Sum over co-rated items between user i and j
            }
        }
    }
    return W;
}