from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

movies = pd.read_csv("movies_5000_with_posters.csv")
movies["tags"] = (movies["genre"] + " " + movies["overview"]).str.lower()

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append({
            "title": movies.iloc[i[0]].title,
            "poster": movies.iloc[i[0]].poster_url
        })
    return recommended_movies

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    selected_movie = None

    if request.method == "POST":
        selected_movie = request.form["movie"]
        recommendations = recommend(selected_movie)

    movie_titles = movies["title"].tolist()
    return render_template("index.html", movie_titles=movie_titles,
                           recommendations=recommendations,
                           selected_movie=selected_movie)

if __name__ == "__main__":
    app.run(debug=True)