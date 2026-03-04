"use client";

import { FormEvent, useState } from "react";

type Recipe = {
  _id: string;
  name: string;
  description?: string;
  ingredients?: string;
  url?: string;
  image?: string;
  score?: number;
};

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setHasSearched(true);

    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `Request failed with ${res.status}`);
      }

      const data = await res.json();
      setRecipes(data.results ?? []);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Unexpected error while searching.");
      setRecipes([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <section className="card">
        <h1 className="title">Vector recipe search</h1>
        <p className="subtitle">
          Search your MongoDB Atlas recipe index by ingredients or natural language.
        </p>

        <form onSubmit={handleSubmit} className="search-row">
          <input
            className="search-input"
            placeholder='Example: "chicken thighs and lemon" or "quick vegetarian pasta"'
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button
            type="submit"
            className="search-button"
            disabled={loading || !query.trim()}
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </form>

        <div className="hint">
          Powered by MongoDB Atlas Vector Search over precomputed recipe embeddings.
        </div>

        {error && <div className="error">{error}</div>}

        {recipes.length > 0 && (
          <div className="results">
            {recipes.map((recipe) => (
              <article key={recipe._id} className="recipe">
                <header className="recipe-header">
                  <div className="recipe-name">
                    {recipe.url ? (
                      <a
                        href={recipe.url}
                        target="_blank"
                        rel="noreferrer"
                        style={{ color: "#38bdf8", textDecoration: "none" }}
                      >
                        {recipe.name}
                      </a>
                    ) : (
                      recipe.name
                    )}
                  </div>
                  {typeof recipe.score === "number" && (
                    <span className="recipe-score">
                      score: {recipe.score.toFixed(3)}
                    </span>
                  )}
                </header>
                {recipe.description && (
                  <div className="recipe-description">{recipe.description}</div>
                )}
                {recipe.ingredients && (
                  <div className="recipe-meta">
                    {recipe.ingredients}
                  </div>
                )}
              </article>
            ))}
          </div>
        )}

        {hasSearched && !loading && !error && recipes.length === 0 && (
          <div className="empty-state">
            No recipes found yet. Try a more general description or fewer ingredients.
          </div>
        )}
      </section>
    </main>
  );
}
