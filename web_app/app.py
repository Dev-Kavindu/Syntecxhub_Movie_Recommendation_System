from __future__ import annotations

import ast
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
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR.parent / "data"
MOVIES_CSV_PATH = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV_PATH = DATA_DIR / "tmdb_5000_credits.csv"
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

# Initialize stemmer for text preprocessing
stemmer = PorterStemmer()


def _convert(obj: str) -> list[str]:
    """Convert a JSON-like string column into a list of names."""
    try:
        items = ast.literal_eval(obj)
    except (ValueError, SyntaxError):
        return []
    
    if not isinstance(items, list):
        return []
    
    return [item.get('name', '') for item in items if isinstance(item, dict) and 'name' in item]


def _convert3(obj: str) -> list[str]:
    """Extract top 3 cast member names from the cast field."""
    if isinstance(obj, str):
        try:
            items = ast.literal_eval(obj)
        except (ValueError, SyntaxError):
            return []
    elif isinstance(obj, list):
        items = obj
    else:
        return []

    result = []
    for i in items[:3]:
        if isinstance(i, dict) and 'name' in i:
            result.append(i['name'])
        elif isinstance(i, str):
            result.append(i)
    return result


def _fetch_director(obj: str) -> list[str]:
    """Extract the director name from the crew field."""
    try:
        items = ast.literal_eval(obj)
    except (ValueError, SyntaxError):
        return []
    
    if not isinstance(items, list):
        return []
    
    for i in items:
        if isinstance(i, dict) and i.get('job') == 'Director':
            return [i.get('name', '')]
    return []


def _stem(text: str) -> str:
    """Apply stemming to text."""
    return ' '.join([stemmer.stem(word) for word in text.split()])


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def get_movies() -> pd.DataFrame:
    """Load and process movie data from CSV files.

    This replicates the notebook's preprocessing pipeline:
    1. Load movies and credits CSV files
    2. Merge on title
    3. Parse JSON-like columns (genres, keywords, cast, crew)
    4. Create combined tags column
    5. Apply stemming
    """

    if not MOVIES_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MOVIES_CSV_PATH.name} in {DATA_DIR}. "
            "Ensure the data directory contains the TMDB CSV files."
        )

    if not CREDITS_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CREDITS_CSV_PATH.name} in {DATA_DIR}. "
            "Ensure the data directory contains the TMDB CSV files."
        )

    # Load CSV files
    movies = pd.read_csv(MOVIES_CSV_PATH)
    credits = pd.read_csv(CREDITS_CSV_PATH)

    # Merge credits into movies on title
    movies = movies.merge(credits, on='title')

    # Select relevant columns
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

    # Drop rows with missing values
    movies.dropna(inplace=True)

    # Parse JSON-like columns
    movies['genres'] = movies['genres'].apply(_convert)
    movies['keywords'] = movies['keywords'].apply(_convert)
    movies['cast'] = movies['cast'].apply(_convert3)
    movies['crew'] = movies['crew'].apply(_fetch_director)

    # Create tags by combining all text features
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

    # Select only the columns needed for recommendations
    new_df = movies[['movie_id', 'title', 'tags']]

    # Join list elements into space-separated strings
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))

    # Convert to lowercase
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

    # Apply stemming
    new_df['tags'] = new_df['tags'].apply(_stem)

    if "title" not in new_df.columns:
        raise ValueError("Processed data must contain a 'title' column.")

    return new_df


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
            "Processed movie data must include a 'tags' column to compute similarities on-the-fly."
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
