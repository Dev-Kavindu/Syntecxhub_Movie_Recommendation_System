const THEME_KEY = "moviemind_theme";

function setTheme(theme) {
  const isLight = theme === "light";
  document.body.classList.toggle("light-mode", isLight);
  localStorage.setItem(THEME_KEY, theme);

  const toggle = document.getElementById("themeToggle");
  if (toggle) toggle.checked = isLight;
}

function getTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved === "dark" || saved === "light") return saved;

  // Default to dark mode
  return "dark";
}

function el(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function setStatus(message, type = "") {
  const status = document.getElementById("status");
  status.textContent = message || "";
  status.className = "status" + (type === "error" ? " status--error" : "");
}

function posterIconSvg() {
  return `
    <svg viewBox="0 0 24 24" width="26" height="26" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path d="M7 3h10a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z" stroke="currentColor" stroke-width="1.5" opacity="0.9"/>
      <path d="M9 7h6M9 11h6M9 15h4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" opacity="0.9"/>
      <path d="M15.5 16.5l1.6-.9a.5.5 0 0 1 .75.43v1.92a.5.5 0 0 1-.75.43l-1.6-.92a.5.5 0 0 1 0-.85Z" fill="currentColor" opacity="0.85"/>
    </svg>`;
}

function setPoster(elPoster, posterUrl) {
  if (!posterUrl) {
    elPoster.classList.add("card__poster--placeholder");
    elPoster.innerHTML = posterIconSvg();
    return;
  }

  elPoster.classList.remove("card__poster--placeholder");
  elPoster.innerHTML = "";

  const img = document.createElement("img");
  img.src = posterUrl;
  img.alt = "Movie poster";
  img.loading = "lazy";

  img.addEventListener("error", () => {
    elPoster.classList.add("card__poster--placeholder");
    elPoster.innerHTML = posterIconSvg();
  });

  elPoster.appendChild(img);
}

function renderStars(voteAverage) {
  if (voteAverage === null || voteAverage === undefined) {
    return `<div class="card__rating"><span class="rating-text">No rating</span></div>`;
  }

  const ratingValue = voteAverage / 2; // Convert from 0-10 to 0-5
  const fullStars = Math.floor(ratingValue);
  const hasHalfStar = ratingValue % 1 >= 0.5;
  let starsHtml = "";

  for (let i = 0; i < 5; i++) {
    if (i < fullStars) {
      starsHtml += `<span class="star star--full">★</span>`;
    } else if (i === fullStars && hasHalfStar) {
      starsHtml += `<span class="star star--half">★</span>`;
    } else {
      starsHtml += `<span class="star star--empty">☆</span>`;
    }
  }

  return `<div class="card__rating">${starsHtml}<span class="rating-value">${voteAverage.toFixed(1)}/10</span></div>`;
}

function renderCards(items) {
  const container = document.getElementById("recommendations-grid");
  container.innerHTML = "";

  (items || []).forEach((item, idx) => {
    const normalized = typeof item === "string" ? { title: item, poster_url: null, movie_id: null, vote_average: null } : item;

    const card = el("div", "card");

    const poster = el("div", "card__poster");
    setPoster(poster, normalized.poster_url);

    const badge = el("div", "card__badge");
    const dot = el("span", "badge-dot");
    const label = el("span");
    label.textContent = `Recommendation #${idx + 1}`;
    badge.appendChild(dot);
    badge.appendChild(label);

    const h3 = el("h3", "card__title");
    h3.textContent = normalized.title || "";

    const ratingHtml = renderStars(normalized.vote_average);
    const ratingContainer = el("div", "card__rating-container");
    ratingContainer.innerHTML = ratingHtml;

    const meta = el("div", "card__meta");
    meta.textContent = "Similar content based on your model.";

    card.appendChild(poster);
    card.appendChild(badge);
    card.appendChild(h3);
    card.appendChild(ratingContainer);
    card.appendChild(meta);

    // Wrap card in a link if movie_id exists, otherwise just append the card
    if (normalized.movie_id) {
      const link = document.createElement("a");
      link.href = `https://www.themoviedb.org/movie/${normalized.movie_id}`;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.className = "card-link";
      link.appendChild(card);
      container.appendChild(link);
    } else {
      container.appendChild(card);
    }
  });
}

async function recommend(title) {
  const res = await fetch("/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "Request failed");
  }

  return data;
}

function debounce(fn, waitMs) {
  let t = null;
  return (...args) => {
    if (t) window.clearTimeout(t);
    t = window.setTimeout(() => fn(...args), waitMs);
  };
}

async function fetchSuggestions(query, limit = 10) {
  const url = `/suggest?q=${encodeURIComponent(query)}&limit=${limit}`;
  const res = await fetch(url);
  const data = await res.json();
  return Array.isArray(data.suggestions) ? data.suggestions : [];
}

function wireAutocomplete(input, box) {
  let items = [];
  let active = -1;
  let lastQuery = "";

  function setExpanded(v) {
    input.setAttribute("aria-expanded", v ? "true" : "false");
  }

  function clear() {
    items = [];
    active = -1;
    box.innerHTML = "";
    box.classList.remove("suggestions--open");
    setExpanded(false);
  }

  function highlight() {
    const options = Array.from(box.querySelectorAll(".suggestion"));
    options.forEach((opt, i) => {
      opt.classList.toggle("suggestion--active", i === active);
    });
  }

  function choose(value) {
    input.value = value;
    input.dataset.selected = "1";
    input.focus();
    clear();
  }

  function render(nextItems) {
    items = nextItems;
    active = -1;
    box.innerHTML = "";

    if (!items.length) {
      clear();
      return;
    }

    items.forEach((title) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "suggestion";
      btn.setAttribute("role", "option");
      btn.textContent = title;

      // Use mousedown so selection wins over input blur
      btn.addEventListener("mousedown", (e) => {
        e.preventDefault();
        choose(title);
      });

      box.appendChild(btn);
    });

    box.classList.add("suggestions--open");
    setExpanded(true);
  }

  const update = debounce(async () => {
    const q = input.value.trim();
    if (!q) {
      clear();
      return;
    }

    // Avoid duplicate fetches
    if (q.toLowerCase() === lastQuery.toLowerCase() && box.classList.contains("suggestions--open")) {
      return;
    }

    lastQuery = q;

    try {
      const next = await fetchSuggestions(q, 10);
      // Only render if input hasn't changed while awaiting
      if (input.value.trim() === q) {
        render(next);
      }
    } catch {
      // Autocomplete should never block the main flow
      clear();
    }
  }, 150);

  input.addEventListener("input", () => {
    input.dataset.selected = "0";
    update();
  });

  input.addEventListener("keydown", (e) => {
    if (!items.length) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      active = Math.min(active + 1, items.length - 1);
      highlight();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      active = Math.max(active - 1, 0);
      highlight();
    } else if (e.key === "Enter") {
      if (active >= 0 && active < items.length) {
        e.preventDefault();
        choose(items[active]);
      }
    } else if (e.key === "Escape") {
      clear();
    }
  });

  input.addEventListener("blur", () => {
    window.setTimeout(clear, 120);
  });

  return { clear };
}

function wireUp() {
  console.log("[MovieMind] Initializing...");

  // Theme
  setTheme(getTheme());
  const themeToggle = document.getElementById("themeToggle");
  if (themeToggle) {
    themeToggle.addEventListener("change", () => {
      setTheme(themeToggle.checked ? "light" : "dark");
    });
  }

  // Elements
  const input = document.getElementById("movie-input");
  const box = document.getElementById("suggestions");
  const form = document.getElementById("recommendForm");
  const btn = document.getElementById("recommend-btn");

  if (!input || !box || !form || !btn) {
    console.error("[MovieMind] Missing elements:", { input, box, form, btn });
    return;
  }

  console.log("[MovieMind] Elements found successfully");

  // Autocomplete
  input.dataset.selected = "0";
  const ac = wireAutocomplete(input, box);

  // Recommend handler
  async function handleRecommend(e) {
    if (e) e.preventDefault();

    const title = input.value.trim();
    console.log("[MovieMind] Recommending for:", title);

    if (!title) {
      setStatus("Please pick a movie title.", "error");
      return;
    }

    if (input.dataset.selected !== "1") {
      setStatus("Please choose a movie from the suggestions.", "error");
      return;
    }

    ac.clear();
    btn.disabled = true;
    setStatus("Generating recommendations...");

    try {
      const data = await recommend(title);
      console.log("[MovieMind] Received recommendations:", data);
      renderCards(data.recommendations || []);
      setStatus(`Showing top ${data.recommendations.length} recommendations.`);
    } catch (err) {
      console.error("[MovieMind] Recommend error:", err);
      renderCards([]);
      setStatus(err.message, "error");
    } finally {
      btn.disabled = false;
    }
  }

  form.addEventListener("submit", handleRecommend);
  btn.addEventListener("click", handleRecommend);
}

document.addEventListener("DOMContentLoaded", wireUp);
