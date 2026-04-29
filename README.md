<div align="center">

# MovieMind AI

### *Discover Your Next Favorite Movie with Content-Based Filtering*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Live Deployment](https://img.shields.io/badge/Deployment-Railway-9900ff.svg)](https://syntecxhubmovierecommendationsystem-production.up.railway.app/)

A sleek, ML-powered web application that suggests movies based on user interest using **Content-Based Filtering**. Built with a modern glassmorphism dashboard UI and integrated with the TMDB API for real-time movie posters and details.

</div>

---

## 📸 Screenshots / Demo

<!-- Add your screenshots here -->
<!-- 
![Dashboard Screenshot](path/to/screenshot1.png)
![Recommendation Results](path/to/screenshot2.png)
-->

> *Professional dark-mode dashboard with neon AI core sidebar animation*

---

## 🧠 Core Concept: The Machine Learning Brain

### Content-Based Filtering

MovieMind AI analyzes movie metadata to find similar titles. Unlike collaborative filtering that relies on user behavior, content-based filtering uses the intrinsic attributes of each movie:

- **Genres** — Action, Drama, Sci-Fi, etc.
- **Keywords** — Plot-relevant terms extracted from overviews
- **Cast & Crew** — Actors, directors, and key production members
- **Overview Text** — Natural language summaries

### Vectorization: From Text to Math

To compute similarities, we first convert textual metadata into numerical vectors. This is done using **CountVectorizer** from scikit-learn:

1. **Text Preprocessing** — All metadata fields are concatenated into a single `"tags"` string per movie
2. **Stop Word Removal** — Common English words (the, and, is) are stripped
3. **Tokenization** — Remaining terms are split into individual tokens
4. **Vocabulary Building** — The top 5,000 most frequent tokens form the feature space
5. **Sparse Matrix Generation** — Each movie becomes a 5,000-dimensional vector where each dimension represents the frequency of a specific term

### Similarity Engine: Cosine Similarity

The similarity between two movies is computed using **Cosine Similarity**, which measures the cosine of the angle between two vectors in a high-dimensional space:

$$
\cos(\theta) = \frac{A \cdot B}{|A| |B|}
$$

**Why Cosine Similarity?**

- **Scale-Invariant** — Focuses on the direction (content pattern) rather than magnitude (document length)
- **High-Dimensional Efficiency** — Works well with sparse text vectors where most dimensions are zero
- **Interpretability** — Scores range from `0` (unrelated) to `1` (identical), making ranking intuitive

The result is an `N × N` similarity matrix where `N` is the number of movies in the dataset. Each entry `sim[i][j]` represents the similarity score between movie `i` and movie `j`.

### Recommendation Pipeline

```mermaid
graph LR
    A[User Input] --> B[Fuzzy Title Match]
    B --> C[Retrieve Movie Vector]
    C --> D[Compute Cosine Similarity]
    D --> E[Rank by Score]
    E --> F[Top-5 Selection]
    F --> G[TMDB API Enrichment]
    G --> H[Display Results]
```

1. **Preprocessing** — Movie metadata is cleaned, stemmed, and concatenated into a unified `tags` column
2. **Vectorization** — `CountVectorizer` transforms tags into a numerical matrix
3. **Similarity Computation** — `cosine_similarity` generates the full pairwise matrix
4. **Query Matching** — When a user inputs a movie title, the engine finds the closest match using `difflib.get_close_matches` with a fuzzy threshold
5. **Ranking** — The top-5 most similar movies (excluding the input itself) are returned
6. **Enrichment** — TMDB API fetches live poster images and vote averages for each result

### Fallback Strategy

The application uses a **hybrid data loading approach** to avoid Git LFS and unpickling issues in production:

- **Primary**: CSV-based data loading from `data/tmdb_5000_movies.csv` and `data/tmdb_5000_credits.csv`
- **Fallback**: On-the-fly similarity computation using CountVectorizer when precomputed `.pkl` files are unavailable
- **Benefits**: Eliminates large pickle file dependencies, reduces repository size, and improves deployment reliability

---

## 🏗️ System Architecture

```
Movie_Recommendation_System/
├── web_app/
│   ├── app.py                  # Flask backend + ML pipeline
│   ├── models/                 # Pickled movie data & similarity matrix
│   ├── static/
│   │   ├── style.css           # Glassmorphism dashboard styles
│   │   └── script.js           # Autocomplete, fetch, theme toggle
│   ├── templates/
│   │   └── index.html          # Dashboard layout
│   └── .env.example            # TMDB_API_KEY template
├── notebooks/                  # Data preprocessing & model export
├── data/                       # Raw datasets
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes envs, caches, large pickles
├── LICENSE                     # MIT License
└── README.md                   # You are here
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Dynamic UI** | Professional glassmorphism dashboard with fixed left sidebar and responsive movie card grids |
| **Mobile-Responsive** | Fully responsive design that adapts seamlessly to mobile, tablet, and desktop screens |
| **Real-Time Search** | Autocomplete suggestions powered by fuzzy matching against the full movie database |
| **TMDB Integration** | Live poster images, ratings, and metadata fetched from The Movie Database API |
| **Neon Sidebar** | CSS-only animated AI core with rotating rings and breathing glow effect |
| **Theme Toggle** | Dark/Light mode switch with smooth transitions and persistent user preference |
| **ML Pipeline** | Content-based filtering using CountVectorizer and Cosine Similarity |
| **Production Ready** | Gunicorn-configured WSGI server with environment variable support |

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- A TMDB API key ([get one here](https://www.themoviedb.org/settings/api))

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/moviemind-ai.git
cd moviemind-ai
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add Your TMDB API Key

Create a `.env` file in the `web_app/` directory:

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

### Step 5: Prepare Model Files

Ensure `web_app/models/movie_list.pkl` and `web_app/models/similarity.pkl` are present. These are generated by running the Jupyter notebook in the `notebooks/` directory. If the files are too large for Git, store them via Git LFS or an external storage service.

### Step 6: Run Locally

```bash
cd web_app
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## 🌐 Deployment

### Using Gunicorn

```bash
cd web_app
gunicorn -w 4 -b 0.0.0.0:$PORT app:app
```

The `app.py` is already configured for production with `host='0.0.0.0'` and `port` read from the `PORT` environment variable (defaults to `5000`).

### Platform Deployments

#### Railway (Primary Deployment)

The application is currently deployed on Railway at:
**[https://syntecxhubmovierecommendationsystem-production.up.railway.app/](https://syntecxhubmovierecommendationsystem-production.up.railway.app/)**

**Deployment Steps:**
1. Create a new project from your GitHub repo
2. Add `TMDB_API_KEY` to environment variables
3. Deploy with automatic builds
4. Railway automatically handles the CSV data loading and ML pipeline initialization

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, Flask |
| ML / Data | Scikit-Learn, Pandas, NumPy |
| API Integration | Requests (TMDB API) |
| Frontend | HTML5, CSS3 (Glassmorphism), JavaScript (ES6+) |
| Environment | python-dotenv |
| Production WSGI | Gunicorn |

---

## 📊 Data Source

- **TMDB API** — Real-time movie posters, ratings, and metadata
- **Local Dataset** — CSV files (`tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`) containing movie metadata, credits, and processed tags
- **Processed Models** — Precomputed similarity matrix (`.pkl`) for faster recommendations, with CSV fallback for production reliability

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Developed by Kavindu Chamod

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kavindu-chamod-ranaweera/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Dev-Kavindu)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

</div>
