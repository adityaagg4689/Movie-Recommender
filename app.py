import streamlit as st
import pandas as pd
import pickle
import requests

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
def load_data():
    movies = pd.DataFrame(pickle.load(open("movies_dic.pkl", "rb")))
    similarity = pickle.load(open("sim.pkl", "rb"))
    return movies, similarity


@st.cache_data
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}

    try:
        res = requests.get(url, params=params, timeout=5).json()
        poster = res.get("poster_path")
        return POSTER_BASE_URL + poster if poster else None
    except:
        return None


def recommend(movie_title, n=10):
    # get index of selected movie
    idx = movies.index[movies["title"] == movie_title][0]
    scores = list(enumerate(similarity[idx]))
    top_movies = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]

    names, posters = [], []

    for i, _ in top_movies:
        names.append(movies.iloc[i].title)
        posters.append(fetch_poster(movies.iloc[i].movie_id))

    return names, posters


# ---------------- APP ----------------
movies, similarity = load_data()

st.title("🎬 Movie Recommendation System")
st.markdown("Get **10 similar movies** based on your selection 🍿")

selected_movie = st.selectbox("Choose a movie", movies["title"])

if st.button("✨ Recommend"):
    names, posters = recommend(selected_movie)

    cols = st.columns(5)
    for i in range(len(names)):
        with cols[i % 5]:
            if posters[i]:
                st.image(posters[i], use_container_width=True)
            else:
                st.write("No Image Available")
            st.markdown(
                f"<p style='text-align:center; font-weight:600'>{names[i]}</p>",
                unsafe_allow_html=True
            )
