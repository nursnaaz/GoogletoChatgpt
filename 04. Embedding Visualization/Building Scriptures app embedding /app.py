import streamlit as st
from gensim.models import Word2Vec, FastText
import gensim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)
# Use Streamlit's caching mechanism to load models only once
def load_model(model_path):
    return gensim.models.KeyedVectors.load(model_path, mmap='r')

# UI for selecting text and model type
st.title('Explore Word Embeddings')

# Select religious text
option = st.selectbox('Choose a religious text:', ('HolyBible', 'NobelQuran', 'TheGita'))

# Select embedding model
model_type = st.radio('Choose embedding model:', ('Word2Vec', 'FastText'))

# Construct the model filename based on selections
model_filename = f"{option}_{model_type.lower()}.model"

# Load the model
model = load_model(model_filename)

# Text input for word exploration
word = st.text_input("Enter a word to find similar words:", "heaven")

# Display similar words when a word is entered
if word:
    try:
        similar_words = model.wv.most_similar(word)
        st.write("Similar words to", word, ":")
        for similar_word, similarity in similar_words:
            st.write(f"{similar_word}: {similarity:.2f}")
    except KeyError:
        st.write("Word not in vocabulary.")


# New section for user input words and visualization
st.write("## Visualization of Word Embeddings")
user_input = st.text_input("Enter words separated by commas (e.g., heaven, hell, world):", "heaven, hell")

def visualize_embeddings(model, words):
    # Filter words present in the model's vocabulary
    filtered_words = [word for word in words if word in model.wv.key_to_index]
    if not filtered_words:
        st.write("None of the entered words were found in the model's vocabulary.")
        return

    word_vectors = np.array([model.wv[word] for word in filtered_words])

    # Use PCA for dimensionality reduction
    pca = PCA(n_components=2, random_state=42)
    T = pca.fit_transform(word_vectors)

    # Create a matplotlib figure for plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(T[:, 0], T[:, 1], c='green', edgecolors='r')

    for i, word in enumerate(filtered_words):
        ax.annotate(word, xy=(T[i, 0], T[i, 1]), xytext=(5, 2), textcoords='offset points')




if st.button('Visualize Embeddings'):
    words = [word.strip() for word in user_input.split(',')]
    plt = visualize_embeddings(model, words)
    st.pyplot(plt)
