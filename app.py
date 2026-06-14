from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load and prepare data
movies = pd.read_csv("movies_data.csv")
movies['combined'] = movies['Genre'] + ' ' + movies['Cast'] + ' ' + movies['Title']

# Compute similarity matrix at startup (no pickle file needed)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = vectorizer.fit_transform(movies['combined'])
similarity = cosine_similarity(vectors)

def recommend(movie_title):
    # Case-insensitive search
    match = movies[movies['Title'].str.lower() == movie_title.lower()]
    if match.empty:
        # Try partial match
        match = movies[movies['Title'].str.lower().str.contains(movie_title.lower())]
    if match.empty:
        return None

    movie_index = match.index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in movie_list:
        row = movies.iloc[i[0]]
        recommendations.append({
            'title': row['Title'],
            'genre': row['Genre'],
            'rating': row['Rating'],
            'year': row['Year']
        })
    return recommendations

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    error = None
    searched_movie = ""

    if request.method == "POST":
        movie = request.form.get("movie", "").strip()
        searched_movie = movie
        if movie:
            result = recommend(movie)
            if result is None:
                error = f'Movie "{movie}" not found. Try: Inception, The Matrix, Titanic, Parasite'
            else:
                recommendations = result

    all_movies = sorted(movies['Title'].tolist())
    return render_template("index.html",
                           recommendations=recommendations,
                           error=error,
                           searched_movie=searched_movie,
                           all_movies=all_movies)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
