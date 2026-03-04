# Recipe Vector Search (Next.js + MongoDB Atlas)

This app is a minimal Next.js front end that queries a **MongoDB Atlas Vector Search** index over recipe documents.

Type ingredients or a natural-language description of what you want to cook, and it will return the closest recipes based on the precomputed `embedding` field in your collection.

## Prerequisites

- Node.js 18+ installed
- A MongoDB Atlas cluster with:
  - Database: `MONGODB_DATABASE`
  - Collection: `MONGODB_COLLECTION`
  - Documents look like this:
    ```json
    {
      "_id": "...",
      "name": "Drop Biscuits and Sausage Gravy",
      "ingredients": "Biscuits\\n3 cups All-purpose Flour\\n...",
      "description": "Late Saturday afternoon...",
      "url": "https://example.com/recipe",
      "image": "https://example.com/image.jpg",
      "embedding": [0.01, 0.02, ...] // vector
    }
    ```
  - A **Vector Search** index on the `embedding` field (see below).

- A Hugging Face Inference API key to embed search queries.

## Environment variables

Defined in `.env`:

- `MONGODB_URI` – your Atlas connection string
- `MONGODB_DATABASE` – database name (e.g. `OMAI-arbetsprov`)
- `MONGODB_COLLECTION` – collection name (e.g. `recipes`)
- `MONGODB_VECTOR_INDEX` – (optional) Atlas vector index name, defaults to `vector-index`
- `HUGGINGFACE_API_KEY` – token for Hugging Face Inference API
- `EMBEDDING_MODEL` – e.g. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

## Vector index definition (Atlas)

In Atlas, create a **Vector Search** index on your recipes collection similar to:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 768,
      "similarity": "cosine"
    }
  ]
}
```

Make sure `numDimensions` matches the size of the `embedding` arrays in your documents.

## Install & run

```bash
npm install
npm run dev
```

Then open `http://localhost:3000` and start prompting for recipes

## How it works

- `app/page.tsx` – simple client page with a search box and results list.
- `app/api/search/route.ts` – API route that:
  - embeds the query text using the Hugging Face model from `EMBEDDING_MODEL`
  - runs a `$vectorSearch` aggregation against the Atlas index on `embedding`
  - returns the top matches with their vector similarity scores
- `lib/mongodb.ts` – reuses a single `MongoClient` instance across requests.
- `lib/embedding.ts` – small helper for calling the Hugging Face Inference API for embeddings.

## Programming language & tech stack

This solution is implemented in **TypeScript** using **Next.js** 

- **Runtime**: Node.js 18+, Next.js 15
- **Language**: TypeScript / JavaScript
- **Core dependencies**:
  - `next`, `react`, `react-dom` – front end framework and UI
  - `mongodb` – MongoDB Node.js driver for talking to MongoDB Atlas
  - `@huggingface/inference` – client for the Hugging Face Inference API

Supporting systems the app relies on:

- **MongoDB Atlas** – stores recipe documents and hosts the **Vector Search** index.
- **Hugging Face Inference API** – hosts the embedding / similarity model used by the API.

To start the program in development:

1. Create and fill in `.env` as described above.
2. Run `npm install` to install dependencies.
3. Run `npm run dev` to start the Next.js dev server on `http://localhost:3000`.

For a simple production-style run:

```bash
npm install
npm run build
npm start
```

## Language handling (multi-language input, English recipes)

The API is designed so that you can **search using multiple natural languages**, while the **recipes themselves are returned in English**:

- The query string is embedded with a **multilingual sentence-transformer model** (see `EMBEDDING_MODEL`, e.g. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
- Because the model is multilingual, you can type queries in languages like Swedish, English, etc., and still retrieve the most relevant English recipes.
- The recipes stored in MongoDB are in English, so all recipe fields in the response (`name`, `description`, `ingredients`) are returned in English only.

## AI usage and motivation

This project uses a few focused AI components rather than a general-purpose chat model:

- **Text embeddings for semantic search**:
  - `lib/embedding.ts` calls the Hugging Face Inference API's `featureExtraction` endpoint to turn the free-text query into a dense vector.
  - These vectors capture semantic meaning instead of just exact words, so the search can match on intent (e.g. "easy vegetarian dinner" vs. "quick meat-free meal").
  - The embeddings are compared against precomputed recipe embeddings in MongoDB Atlas' **Vector Search** index to efficiently find the closest recipes.

- **Multilingual support**:
  - By using a **multilingual sentence-transformer** model, the same embedding space is shared across several languages.
  - This lets users search in different languages while still retrieving English recipes, satisfying the "multi-language input, English output" requirement.

- **Reranking for better relevance**:
  - In `app/api/search/route.ts`, the initial vector search results can optionally be **reranked** with the Hugging Face `sentenceSimilarity` API.
  - The reranker model looks at the full text of each candidate recipe relative to the query and assigns a similarity score.
  - Results are then sorted by this rerank score to surface the most relevant recipes at the top.
