"""Movie Recommendation System"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


print("="*70)
print("MOVIE RECOMMENDATION SYSTEM")
print("="*70)

# Load data
df = pd.read_csv('movies_data.csv')
print(f"\nMovies loaded: {len(df)}")
print(df.head())

# Preprocessing
df['combined'] = df['Genre'] + ' ' + df['Cast']

# Feature extraction
vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = vectorizer.fit_transform(df['combined'])
print(f"\nTF-IDF matrix shape: {tfidf_matrix.shape}")

# Compute similarity
similarity_matrix = cosine_similarity(tfidf_matrix)
print(f"Similarity matrix shape: {similarity_matrix.shape}")

# Recommendation function
def get_recommendations(movie_title, n=5):
    try:
        idx = df[df['Title'].str.lower() == movie_title.lower()].index[0]
    except:
        return None
    
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    recs = []
    for i, score in scores:
        recs.append((df['Title'].iloc[i], df['Genre'].iloc[i], df['Rating'].iloc[i], score))
    return recs

# Test recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

test_movies = ['Inception', 'The Matrix', 'Titanic']

for movie in test_movies:
    print(f"\nMovie: {movie}")
    recs = get_recommendations(movie, 3)
    if recs:
        for i, (title, genre, rating, score) in enumerate(recs, 1):
            print(f"  {i}. {title} - {genre} ({score*100:.1f}%)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
ax1.hist(df['Rating'], bins=10, color='#3498db', edgecolor='black')
ax1.set_xlabel('Rating')
ax1.set_title('Movie Rating Distribution')

ax2 = axes[0, 1]
ax2.bar(df['Title'], df['Rating'], color='#2ecc71')
ax2.set_ylabel('Rating')
ax2.set_title('Movie Ratings')
ax2.tick_params(axis='x', rotation=45)

ax3 = axes[1, 0]
similarity_scores = similarity_matrix[0]
ax3.hist(similarity_scores, bins=20, color='#e74c3c')
ax3.set_xlabel('Similarity')
ax3.set_title('Similarity Distribution')

ax4 = axes[1, 1]
years = df['Year'].value_counts().sort_index()
ax4.bar(years.index, years.values, color='#f39c12')
ax4.set_xlabel('Year')
ax4.set_title('Movies by Year')

plt.tight_layout()
plt.savefig('movie_recommendation_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved: movie_recommendation_analysis.png")
plt.show()

print("\n" + "="*70)
print("PROJECT COMPLETED!")
print("="*70)