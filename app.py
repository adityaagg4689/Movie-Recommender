import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

TMDB_API_KEY = "931f04421b20e464985b7b4eaeddc5f8"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500/"

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_movies():
    movies = pd.DataFrame(pickle.load(open("movies_dic.pkl", "rb")))
    movies = movies.head(1500)  # 🔥 VERY IMPORTANT
    return movies


@st.cache_data
def build_vectorizer(tags):
    cv = CountVectorizer(max_features=3000, stop_words="english")
    vectors = cv.fit_transform(tags)   # KEEP SPARSE
    return cv, vectors


@st.cache_data
def fetch_poster(movie_id):
    try:
        res = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=5
        ).json()
        poster = res.get("poster_path")
        return POSTER_BASE_URL + poster if poster else None
    except:
        return None


# ---------------- RECOMMEND LOGIC ----------------
def recommend(movie_title, n=5):
    idx = movies[movies["title"] == movie_title].index[0]

    # compute similarity ONLY for selected movie
    sim_scores = cosine_similarity(vectors[idx], vectors).flatten()

    top_indices = sim_scores.argsort()[::-1][1:n+1]

    names, posters = [], []
    for i in top_indices:
        names.append(movies.iloc[i].title)
        posters.append(fetch_poster(movies.iloc[i].movie_id))

    return names, posters


# ---------------- APP ----------------
movies = load_movies()
cv, vectors = build_vectorizer(movies["tags"])

st.title("🎬 Movie Recommendation System")
st.markdown("Get **5 similar movies** based on your selection 🍿")

selected_movie = st.selectbox("Choose a movie", movies["title"])

if st.button("✨ Recommend"):
    names, posters = recommend(selected_movie, n=5)

    cols = st.columns(5)
    for col, name, poster in zip(cols, names, posters):
        with col:
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.write("No Image")
            st.markdown(
                f"<p style='text-align:center; font-weight:600'>{name}</p>",
                unsafe_allow_html=True
            )
