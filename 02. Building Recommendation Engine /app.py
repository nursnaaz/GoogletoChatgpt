import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import utility  # Ensure utility contains the preprocessing function

# Load and preprocess data
@st.cache_data
def load_data():
    movie_data = pd.read_csv('imdb_top_1000.csv')
    data = movie_data[['Overview', 'Director', 'Genre', 'Star1', 'Poster_Link']]
    data['combined_info'] = data.apply(lambda x: ' '.join(x.astype(str)), axis=1)
    data['combined_info'] = data['combined_info'].apply(utility.preprocess_text)
    tf = TfidfVectorizer(ngram_range=(1,2))
    tf_idf_matrix = tf.fit_transform(data['combined_info'])
    doc_sim_df = pd.DataFrame(cosine_similarity(tf_idf_matrix))
    return movie_data, doc_sim_df

movie_data, doc_sim_df = load_data()

# Streamlit UI
st.title('Movie Recommendation Engine')

selected_movie = st.selectbox('Choose a movie', movie_data['Series_Title'].values)

def movie_recommender(movie_title, movie_data, doc_sims, num_movies=10):
    movies = movie_data['Series_Title'].values
    movie_idx = np.where(movies == movie_title)[0][0]
    movie_similarities = doc_sims.iloc[movie_idx].values
    similar_movie_idxs = np.argsort(-movie_similarities)[1:num_movies+1]  # Fetching 10 movies
    similar_movies = movie_data.iloc[similar_movie_idxs]
    return similar_movies

if st.button('Recommend'):
    recommended_movies = movie_recommender(selected_movie, movie_data, doc_sims=doc_sim_df, num_movies=10)
    
    # Display recommended movies in two rows of 5
    for i in range(0, 10, 5):
        row = recommended_movies.iloc[i:i+5]
        cols = st.columns(5)  # Create 5 columns for each row
        for idx, (col, movie) in enumerate(zip(cols, row.iterrows())):
            movie = movie[1]  # Extract movie details
            with col:
                st.image(movie['Poster_Link'], caption=movie['Series_Title'], width=150)
