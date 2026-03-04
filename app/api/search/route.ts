import { NextRequest, NextResponse } from "next/server";
import { InferenceClient } from "@huggingface/inference";
import { getMongoClient } from "../../../lib/mongodb";
import { embedQuery } from "../../../lib/embedding";

const HF_API_KEY = process.env.HUGGINGFACE_API_KEY;
const DB_NAME = process.env.MONGODB_DATABASE;
const COLLECTION_NAME = process.env.MONGODB_COLLECTION;
const VECTOR_INDEX = (process.env.MONGODB_VECTOR_INDEX || "recipes-index").replace(/['";]/g, "").trim();

let hfClient: InferenceClient | null = null;
function getHfClient(): InferenceClient {
  if (!hfClient) {
    if (!HF_API_KEY) {
      throw new Error("HUGGINGFACE_API_KEY is not set for reranking.");
    }
    hfClient = new InferenceClient(HF_API_KEY);
  }
  return hfClient;
}

async function rerankResults(query: string, documents: any[]) {
  if (!HF_API_KEY || !documents.length) return null;

  try {
    const client = getHfClient();

    const scores = await client.sentenceSimilarity({
      model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
      provider: "hf-inference",
      inputs: {
        source_sentence: query,
        sentences: documents.map((d) => 
          `${d.name ?? ""} ${d.description ?? ""} ${d.ingredients ?? ""}`.trim()
        ),
      },
    });

    if (!Array.isArray(scores)) return null;
    return scores.map((score, index) => ({ index, score }));
  } catch (error) {
    console.error("Rerank error", error);
    return null;
  }
}

export async function POST(req: NextRequest) {
  try {
    const { query } = await req.json();
    const cleanQuery: string = typeof query === "string" ? query.trim() : "";

    if (!cleanQuery) {
      return NextResponse.json({ error: "Empty query" }, { status: 400 });
    }

    const [client, embedding] = await Promise.all([
      getMongoClient(),
      embedQuery(cleanQuery)
    ]);

    const db = client.db(DB_NAME as string);
    const collection = db.collection(COLLECTION_NAME as string);

    const pipeline: any[] = [
      {
        $vectorSearch: {
          index: VECTOR_INDEX,
          path: "embedding",
          queryVector: embedding,
          numCandidates: 200,
          limit: 30
        }
      },
      {
        $project: {
          name: 1,
          description: 1,
          ingredients: 1,
          url: 1,
          image: 1,
          vectorScore: { $meta: "vectorSearchScore" }
        }
      }
    ];

    const initialResults = await collection.aggregate(pipeline).toArray();

    let finalResults = initialResults;
    const rerankedScores = await rerankResults(cleanQuery, initialResults);

    if (rerankedScores && Array.isArray(rerankedScores)) {
      finalResults = rerankedScores
        .map((item: any) => ({
          ...initialResults[item.index],
          rerankScore: item.score
        }))
        .sort((a, b) => b.rerankScore - a.rerankScore)
        .slice(0, 10);
    }

    const shaped = finalResults.map((doc: any) => ({
      _id: String(doc._id),
      name: doc.name,
      description: doc.description,
      ingredients: doc.ingredients,
      url: doc.url,
      image: doc.image,
      score: doc.rerankScore || doc.vectorScore
    }));

    return NextResponse.json({
      results: shaped,
      meta: {
        reranked: !!rerankedScores,
        count: shaped.length,
        index: VECTOR_INDEX
      }
    });
  } catch (err: any) {
    console.error(err);
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}