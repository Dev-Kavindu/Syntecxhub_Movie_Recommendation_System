from __future__ import annotations

import os
import pickle
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
MOVIE_LIST_PATH = MODELS_DIR / "movie_list.pkl"
SIMILARITY_PATH = MODELS_DIR / "similarity.pkl"

# Load .env so TMDB_API_KEY works even when running `python app.py`
# (Flask only auto-loads .env in some configurations like `flask run` with python-dotenv.)
try:
    from dotenv import load_dotenv

    load_dotenv(APP_DIR / ".env")
except Exception:
    pass

# NOTE: Do NOT hardcode your real TMDB key in source control.
API_KEY = os.environ.get("TMDB_API_KEY", "")

TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

app = Flask(__name__)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def get_movies() -> pd.DataFrame:
    """Load the movie list from disk.

    The notebook exports this as: pickle.dump(new_df.to_dict(), ...)
    which we reconstruct into a DataFrame here.
    """

    if not MOVIE_LIST_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MOVIE_LIST_PATH.name} in {MODELS_DIR}. "
            "Run the notebook export step to generate it."
        )

    movie_dict = _load_pickle(MOVIE_LIST_PATH)
    movies_df = pd.DataFrame(movie_dict)

    if "title" not in movies_df.columns:
        raise ValueError("movie_list.pkl must contain a 'title' column.")

    return movies_df


@lru_cache(maxsize=1)
def get_similarity() -> np.ndarray | None:
    """Try to load the full similarity matrix.

    Note: This file can be large. If loading fails due to memory constraints,
    we return None and fall back to on-the-fly similarity computation.
    """

    if not SIMILARITY_PATH.exists():
        return None

    try:
        sim = _load_pickle(SIMILARITY_PATH)
        return np.array(sim)
    except Exception as e:
        # Some environments can't load a full NxN float64 similarity matrix.
        # If we hit any memory-allocation failure, fall back to on-the-fly similarity.
        name = e.__class__.__name__
        if isinstance(e, MemoryError) or "MemoryError" in name:
            return None
        raise


@lru_cache(maxsize=1)
def get_count_matrix():
    """Build the sparse tag vector matrix (same setup as the notebook).

    This is used as a fallback if the full NxN similarity matrix can't be loaded.
    """

    movies = get_movies()
    if "tags" not in movies.columns:
        raise ValueError(
            "movie_list.pkl must include a 'tags' column to compute similarities on-the-fly."
        )

    cv = CountVectorizer(max_features=5000, stop_words="english")
    matrix = cv.fit_transform(movies["tags"].fillna("").astype(str))
    return matrix


@lru_cache(maxsize=4096)
def fetch_movie_details(movie_id: int) -> dict[str, Any]:
    """Fetch TMDB movie details: poster URL and vote average."""

    if not API_KEY:
        return {"poster_url": None, "vote_average": None}

    try:
        movie_id_int = int(movie_id)
    except Exception:
        return {"poster_url": None, "vote_average": None}

    url = f"{TMDB_API_BASE}/movie/{movie_id_int}"

    try:
        resp = requests.get(
            url,
            params={"api_key": API_KEY, "language": "en-US"},
            timeout=5,
        )
        if not resp.ok:
            return {"poster_url": None, "vote_average": None}
        data = resp.json()
    except Exception:
        return {"poster_url": None, "vote_average": None}

    poster_path = data.get("poster_path")
    poster_url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None
    vote_average = data.get("vote_average")

    return {
        "poster_url": poster_url,
        "vote_average": vote_average,
    }


@lru_cache(maxsize=4096)
def fetch_poster(movie_id: int) -> str | None:
    """Fetch TMDB poster URL for a TMDB movie_id using requests (deprecated, kept for compatibility)."""
    details = fetch_movie_details(movie_id)
    return details.get("poster_url")


def recommend(movie_title: str, top_n: int = 5) -> tuple[str, list[dict[str, Any]]]:
    """Return (matched_title, recommendations) using the notebook's similarity logic."""

    movies = get_movies()

    title_clean = (movie_title or "").strip()
    if not title_clean:
        raise ValueError("Please enter a movie title.")

    # Case-insensitive exact match (same logic, more user-friendly)
    titles_lower = movies["title"].astype(str).str.lower()
    matches = movies.index[titles_lower == title_clean.lower()].tolist()

    if not matches:
        candidates = movies["title"].astype(str).tolist()
        suggestions = get_close_matches(title_clean, candidates, n=5, cutoff=0.6)
        msg = "Movie not found."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        raise LookupError(msg)

    movie_index = int(matches[0])

    sim = get_similarity()
    if sim is not None:
        distances = sim[movie_index]
    else:
        # Fallback: compute similarity vector against all movies from sparse tag vectors
        matrix = get_count_matrix()
        distances = cosine_similarity(matrix[movie_index], matrix).ravel()

    ranked = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in ranked[1 : top_n + 1]]

    matched_title = str(movies.iloc[movie_index]["title"])

    # Build objects for the frontend (title + poster + rating)
    recs: list[dict[str, Any]] = []
    rec_rows = movies.iloc[top_indices]

    for _, row in rec_rows.iterrows():
        title = str(row.get("title", ""))
        movie_id = row.get("movie_id")
        movie_id_int = int(movie_id) if movie_id is not None else None

        movie_details = (
            fetch_movie_details(movie_id_int)
            if movie_id_int is not None
            else {"poster_url": None, "vote_average": None}
        )

        recs.append(
            {
                "title": title,
                "movie_id": movie_id_int,
                "poster_url": movie_details.get("poster_url"),
                "vote_average": movie_details.get("vote_average"),
            }
        )

    return matched_title, recs


@lru_cache(maxsize=1)
def get_titles_lookup() -> tuple[list[str], list[str]]:
    """Cached title list for fast suggestions."""

    titles = get_movies()["title"].astype(str).tolist()
    titles_lower = [t.lower() for t in titles]
    return titles, titles_lower


@app.get("/get_movies")
def get_movies_route():
    """Return the full list of movie titles."""
    try:
        movies = get_movies()
        titles = movies["title"].astype(str).tolist()
        return jsonify({"movies": titles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/suggest")
def suggest_route():
    q = (request.args.get("q") or "").strip().lower()
    try:
        limit = int(request.args.get("limit") or 10)
    except ValueError:
        limit = 10

    limit = max(1, min(limit, 20))
    if not q:
        return jsonify({"suggestions": []})

    titles, titles_lower = get_titles_lookup()

    suggestions: list[str] = []

    # Prefer prefix matches (fast + feels right for autocomplete)
    for t, tl in zip(titles, titles_lower):
        if tl.startswith(q):
            suggestions.append(t)
            if len(suggestions) >= limit:
                return jsonify({"suggestions": suggestions})

    # Fall back to substring matches if we still have room
    for t, tl in zip(titles, titles_lower):
        if q in tl and t not in suggestions:
            suggestions.append(t)
            if len(suggestions) >= limit:
                break

    return jsonify({"suggestions": suggestions})


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/recommend")
def recommend_route():
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        title = data.get("title")
    else:
        title = request.form.get("title")

    try:
        matched_title, recs = recommend(title, top_n=5)
        return jsonify({"matched_title": matched_title, "recommendations": recs})
    except (ValueError, LookupError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Unexpected server error. Please try again."}), 500


if __name__ == "__main__":
    # Run: python app.py
    # Set FLASK_DEBUG=1 to enable debug mode.
    debug = os.environ.get("FLASK_DEBUG") == "1"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)
