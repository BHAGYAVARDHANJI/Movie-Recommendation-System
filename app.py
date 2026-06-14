from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

import pandas as pd
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

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

    recommendations = []

    for i in movie_list:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []

    if request.method == "POST":
        movie = request.form["movie"]

        try:
            recommendations = recommend(movie)
        except:
            recommendations = ["Movie not found"]

    return render_template(
        "index.html",
        recommendations=recommendations
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)