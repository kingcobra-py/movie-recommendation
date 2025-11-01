# 🎬 Movie Recommendation System

A simple **content-based movie recommender** built using the TMDB 5000 Movies dataset.  
It suggests 5 similar movies based on **genres** and **keywords** using TF-IDF and cosine similarity.

## 📖 How It Works
1. Loads movie data from `tmdb_5000_movies.csv`.
2. Extracts and combines each movie’s genres and keywords.
3. Converts text into numerical vectors using **TF-IDF**.
4. Measures similarity between movies with **cosine similarity**.
5. Asks for a movie title and recommends the top 5 similar ones.

## 🛠️ Tech Stack
- Python  
- pandas, numpy  
- scikit-learn  

## 🚀 Usage
```bash
python movie_recommender.py
