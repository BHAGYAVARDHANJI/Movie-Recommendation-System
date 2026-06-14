import pandas as pd

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

print(movies.shape)
print(credits.shape)
movies = movies.merge(credits, on="title")
movies = movies[
    [
        "movie_id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew"
    ]
]
movies["overview"] = movies["overview"].fillna("")

movies["tags"] = (
    movies["overview"].astype(str)
    + " "
    + movies["genres"].astype(str)
    + " "
    + movies["keywords"].astype(str)
)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
cv = CountVectorizer(max_features=5000, stop_words="english")

vectors = cv.fit_transform(movies["tags"]).toarray()

similarity = cosine_similarity(vectors)
def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    for i in movie_list:
        print(movies.iloc[i[0]].title)
print(similarity.shape)
recommend("Avatar")

import pickle

