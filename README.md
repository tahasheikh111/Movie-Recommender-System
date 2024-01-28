# Movie Recommendation System

This project involves building a simple movie recommendation system using Python and the scikit-learn library. The system uses the TMDB 5000 Movies dataset, consisting of information about movies, such as genres, cast, crew, and keywords.

## Requirements
- Python 3.x
- Pandas
- Numpy
- scikit-learn
- NLTK

## Installation
```bash
pip install pandas numpy scikit-learn nltk
```

## Usage

1. Import necessary libraries and load dataset:

```python
import pandas as pd
import numpy as np

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
```

2. Merge datasets and preprocess the data:

```python
# Merge on the 'title' column
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['id', 'title', 'overview', 'keywords', 'genres', 'cast', 'crew']]

# Handle null values and duplicates
movies = movies.dropna().drop_duplicates()

# Apply custom functions to preprocess genres, keywords, cast, and crew columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(convert_crew)
```

3. Create a 'tag' column for each movie:

```python
movies['tag'] = movies['overview'] + movies['keywords'] + movies['genres'] + movies['cast'] + movies['crew']
```

4. Perform text preprocessing and create feature vectors:

```python
new_df = movies[['id', 'title', 'tag']]

# Apply stemming and convert to lowercase
new_df['tag'] = new_df['tag'].apply(lambda x: " ".join(x))
new_df['tag'] = new_df['tag'].apply(ownstem)
new_df['tag'] = new_df['tag'].apply(lambda x: x.lower())

# Use CountVectorizer to create feature vectors
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tag']).toarray()
```

5. Calculate cosine similarity between movies:

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
```

6. Create a recommendation function:

```python
def recommend(movie):
    # Fetch index of the movie
    movie_index = new_df[new_df['title'] == movie].index[0]

    # Sort the similarity indexes
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:10]

    # Print recommended movies
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

# Example recommendation
recommend('Avatar')
```

## Credits
- The dataset used in this project is from TMDB (The Movie Database): [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata)
