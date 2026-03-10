import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

GLASS_CSS = """
<style>
.stApp {
    background: radial-gradient(1200px 600px at 10% 10%, rgba(56, 189, 248, 0.07), transparent 40%),
                radial-gradient(900px 500px at 90% 20%, rgba(34, 197, 94, 0.06), transparent 35%),
                linear-gradient(135deg, #0b1220, #0e1726 60%, #0b1220);
    color: #e2e8f0;
}
header, #MainMenu, footer { visibility: hidden; }
h1, .stTitle {
    color: #e5f4ff !important;
    letter-spacing: 0.3px;
    text-shadow: 0 1px 0 rgba(255,255,255,0.02);
}
.stSelectbox label {
    font-weight: 600;
    color: #b6c3d6 !important;
}
div.stButton > button:first-child {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #07111f;
    border: 0;
    border-radius: 12px;
    padding: 0.6rem 1.1rem;
    font-weight: 700;
    box-shadow: 0 6px 18px rgba(34, 197, 94, 0.25);
}
div.stButton > button:first-child:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 24px rgba(34, 197, 94, 0.35);
}
.reco-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-top: 10px;
}
@media (max-width: 1400px) {.reco-grid { grid-template-columns: repeat(4, 1fr); }}
@media (max-width: 1100px) {.reco-grid { grid-template-columns: repeat(3, 1fr); }}
@media (max-width: 800px) {.reco-grid { grid-template-columns: repeat(2, 1fr); }}
@media (max-width: 520px) {.reco-grid { grid-template-columns: 1fr; }}
.card {
    position: relative;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 16px;
    padding: 10px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
    transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.45);
    border-color: rgba(255,255,255,0.28);
}
.poster {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    border-radius: 12px;
    display: block;
    box-shadow: inset 0 0 1px rgba(255,255,255,0.15);
}
.title {
    margin-top: 10px;
    font-size: 0.98rem;
    line-height: 1.25rem;
    font-weight: 700;
    color: #e9f2ff;
    letter-spacing: 0.2px;
}
.card::before {
    content: "";
    position: absolute;
    inset: -2px;
    background: radial-gradient(400px 120px at -10% -10%, rgba(56,189,248,0.18), transparent 60%),
                radial-gradient(260px 100px at 120% -10%, rgba(99,102,241,0.18), transparent 60%);
    filter: blur(30px);
    z-index: -1;
}
section[data-testid="stSidebar"] {
    background: rgba(17, 24, 39, 0.45) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-right: 1px solid rgba(255,255,255,0.08);
}
.stSelectbox div[data-baseweb="select"] {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(6px);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.16);
}
</style>
"""
st.markdown(GLASS_CSS, unsafe_allow_html=True)

movies = pd.read_csv("movies_5000_with_posters.csv")
movies["title"] = movies["title"].astype(str).str.strip()
movies["genre"] = movies["genre"].fillna("").astype(str)
movies["overview"] = movies["overview"].fillna("").astype(str)
movies["poster_url"] = movies["poster_url"].fillna("").astype(str)
movies["tags"] = movies["genre"] + " " + movies["overview"]

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie: str, top_k: int = 5):
    matches = movies.index[movies["title"].str.casefold() == movie.casefold()]
    if len(matches) == 0:
        candidates = movies[movies["title"].str.contains(movie, case=False, na=False)]
        if candidates.empty:
            return [], []
        movie_index = candidates.index[0]
    else:
        movie_index = matches[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1 : top_k + 1]
    titles, posters = [], []
    for i in movies_list:
        titles.append(movies.iloc[i[0]].title)
        posters.append(movies.iloc[i[0]].poster_url)
    return titles, posters

st.title("🎬 Movie Recommendation System")

col_left, col_right = st.columns([2,1])
with col_left:
    selected_movie = st.selectbox("Select a movie", movies["title"].values, index=0)
with col_right:
    st.write("")
    go = st.button("Recommend")

if go:
    names, posters = recommend(selected_movie)
    if not names:
        st.warning("No similar movies found. Try another title.")
    else:
        cards = ['<div class="reco-grid">']
        for title, poster in zip(names, posters):
            t = str(title).replace("<", "&lt;").replace(">", "&gt;")
            src = poster if poster else "https://via.placeholder.com/300x450?text=No+Image"
            cards.append(f'''
                <div class="card">
                    <img class="poster" src="{src}" alt="{t}" />
                    <div class="title">{t}</div>
                </div>
            ''')
        cards.append("</div>")
        components.html("".join(cards), height=620, scrolling=True)