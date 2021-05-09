import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def movie_predict(name):
    df1 = pd.read_csv('dataset/TMDB5000/tmdb_5000_credits.csv')
    df2 = pd.read_csv('dataset/TMDB5000/tmdb_5000_movies.csv')

    df = df2.merge(df1)
    df['overview'] = df['overview'].fillna('')

    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(df['overview'])
    cos = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()   # map for indices to movies


    try:
        idx = indices[name]
    except KeyError:
        return 'Movie not in dataset'

    similarity = list(enumerate(cos[idx]))

    similarity_scores = sorted(similarity, key = lambda x: x[1], reverse = True)

    similarity_scores = similarity_scores[1:21]    # ignore first because it is the same movie

    movie_indices = [i[0] for i in similarity_scores]
    
    movies = [x for x in df['title'].iloc[movie_indices]]
    
    return movies




