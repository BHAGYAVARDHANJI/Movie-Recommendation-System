"""
Movie Recommender ML - Training Script
Uses movies_data.csv to build and test the recommendation model.
Run this script locally to test the ML logic.
The app.py runs this logic automatically on startup.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies_data.csv")
movies['combined'] = movies['Genre'] + ' ' + movies['Cast'] + ' ' + movies['Title']

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = vectorizer.fit_transform(movies['combined'])
similarity = cosine_similarity(vectors)

def recommend(movie_title, n=5):
    match = movies[movies['Title'].str.lower() == movie_title.lower()]
    if match.empty:
        match = movies[movies['Title'].str.lower().str.contains(movie_title.lower())]
    if match.empty:
        print(f"Movie '{movie_title}' not found.")
        return
    movie_index = match.index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:n+1]
    print(f"\nBecause you liked '{movies.iloc[movie_index]['Title']}':")
    for rank, (i, score) in enumerate(movie_list, 1):
        row = movies.iloc[i]
        print(f"  {rank}. {row['Title']} ({row['Year']}) - {row['Genre']} | ⭐{row['Rating']} | Match: {score*100:.1f}%")

if __name__ == "__main__":
    print(f"Loaded {len(movies)} movies\n")
    recommend("Inception")
    recommend("The Godfather")
    recommend("Toy Story")
