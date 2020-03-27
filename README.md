# Recommender Systems
## Introduction
Recommender systems are software tools and techniques that provide suggestions for items to be of use to a user. The collaborative filtering approach evaluates items using the opinions or ratings of other users. Alternatively, the content-based approach works by learning the items’ features to match the user’s preferences and interests. The code found in this repository implements several collaborative filtering and content-based methods: K-nearest neighbours, hierarchical clustering, association rule mining, ordinal logistic regression, classification trees, TF-IDF, and matrix factorisation.

## Data
This dataset describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. The data are contained in the files `links.csv`, `movies.csv`, `ratings.csv` and `tags.csv`. More details about the contents and use of all these files follows. This and other GroupLens data sets are publicly available for download at <http://grouplens.org/datasets/>.

MovieLens users were selected at random for inclusion. User ids have been anonymized. User ids are consistent between `ratings.csv` and `tags.csv` (i.e., the same id refers to the same user across the two files).

Only movies with at least one rating or tag are included in the dataset. These movie ids are consistent with those used on the MovieLens web site (e.g., id `1` corresponds to the URL <https://movielens.org/movies/1>). Movie ids are consistent between `ratings.csv`, `tags.csv`, `movies.csv`, and `links.csv` (i.e., the same id refers to the same movie across these four data files).


1. ratings.csv


...All ratings are contained in the file `ratings.csv`. Each line of this file after the header row represents one rating of one movie by one user, and has the following format: `userId,movieId,rating,timestamp`. The lines within this file are ordered first by userId, then, within user, by movieId. Ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars). Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

2. tags.csv

...All tags are contained in the file `tags.csv`. Each line of this file after the header row represents one tag applied to one movie by one user, and has the following format: `userId,movieId,tag,timestamp`. The lines within this file are ordered first by userId, then, within user, by movieId. Tags are user-generated metadata about movies. Each tag is typically a single word or short phrase. The meaning, value, and purpose of a particular tag is determined by each user. Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

3. movies.csv

... Movie information is contained in the file `movies.csv`. Each line of this file after the header row represents one movie, and has the following format: `movieId,title,genres`. Movie titles are entered manually or imported from <https://www.themoviedb.org/>, and include the year of release in parentheses. Errors and inconsistencies may exist in these titles.

```R
# Get movie recommendations for user userId = 10
item_based_recommendations = as.numeric(names(sort(Get_Recommendations(userId = 10, W = W_items)[is.na(R[10,])], decreasing = TRUE)))
```
