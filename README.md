# 🎬 Movie Recommendation System

A content-based movie recommendation system built with Python and Machine Learning. Type any movie name and get 5 similar movie recommendations instantly.

## Live Demo
Deployed on Render — check the Deployments section on the right.

## How It Works
1. Each movie is described by its **Genre**, **Cast**, and **Title**
2. **TF-IDF Vectorization** converts text into numerical vectors
3. **Cosine Similarity** finds movies closest to your selection
4. Top 5 most similar movies are returned

## Features
- 🎯 140+ movies in the dataset
- 🔍 Smart search with autocomplete
- 🎨 Clean dark UI with responsive design
- ⚡ No pre-computed files needed — similarity computed at startup

## Tech Stack
- **Backend:** Python, Flask, Gunicorn
- **ML:** Scikit-learn (TF-IDF + Cosine Similarity)
- **Data:** Pandas, NumPy
- **Frontend:** HTML, CSS

## Run Locally
```bash
git clone https://github.com/BHAGYAVARDHANJI/Movie-Recommendation-System.git
cd Movie-Recommendation-System
pip install -r requirements.txt
python app.py
```
Then open http://localhost:10000

## Files
| File | Purpose |
|------|---------|
| `app.py` | Main Flask app — loads data, computes similarity, serves web |
| `movies_data.csv` | Dataset with 140+ movies |
| `movie_recommender_ml.py` | Standalone ML script for local testing |
| `recommendation_system.py` | Data analysis and visualization script |
| `templates/index.html` | Frontend HTML |
| `static/style.css` | Styling |

## Dataset
Custom dataset with 140+ popular movies including genre, cast, rating, and year.
