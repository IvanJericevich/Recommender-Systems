# Recommender-Systems
Recommender systems are software tools and techniques that provide suggestions for items to be of use to a user. The collaborative filtering approach evaluates items using the opinions or ratings of other users. Alternatively, the content-based approach works by learning the items’ features to match the user’s preferences and interests. The code found in this repository implements several collaborative filtering and content-based methods: K-nearest neighbours, hierarchical clustering, association rule mining, ordinal logistic regression, classification trees, TF-IDF, and matrix factorisation.

## Data
This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.

Users were selected at random for inclusion. All selected users had rated at least 20 movies. No demographic information is included. Each user is represented by an id, and no other information is provided.

The data are contained in the files `links.csv`, `movies.csv`, `ratings.csv` and `tags.csv`. More details about the contents and use of all these files follows.

This is a *development* dataset. As such, it may change over time and is not an appropriate dataset for shared research results. See available *benchmark* datasets if that is your intent.

This and other GroupLens data sets are publicly available for download at <http://grouplens.org/datasets/>.
