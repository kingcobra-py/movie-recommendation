"""
🎬 Movie Recommendation System (Content-Based)
Uses TMDB 5000 dataset to suggest 5 similar movies based on genres and keywords.
"""

import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# load dataset
dataset = pd.read_csv("tmdb_5000_movies.csv")

# combine genres and keywords
def genres_and_keyword(row):
    keyname, genname = [], []
    genres = json.loads(row['genres'])
    for i in genres:
        genname.append(i['name'])
    keyword = json.loads(row['keywords'])
    for i in keyword:
        keyname.append(i['name'])
    return "%s %s" % (genname, keyname)

# create new column
dataset['docs'] = dataset.apply(genres_and_keyword, axis=1)

# build TF-IDF model
tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(dataset['docs'])

# index movie titles (lowercase)
movieidx = pd.Series(dataset.index, index=dataset['title'].str.lower())

# recommend top 5 similar movies
def recommand(title):
    idx = movieidx[title]
    if type(idx) == pd.Series:
        idx = idx.iloc[0]
    query = X[idx]
    score = cosine_similarity(query, X).flatten()
    recommand_index = (-score).argsort()[1:6]
    return dataset['title'].iloc[recommand_index]

# get user input
nameOfmovie = str(input('Type Name Of Movie: ')).strip().lower()

# check if movie exists (case-insensitive) and show recommendations
if nameOfmovie in dataset['title'].str.lower().values:
    print(f"Our Recommanded Movies:\n{recommand(nameOfmovie)}")
else:
    print('No Movie Found')
