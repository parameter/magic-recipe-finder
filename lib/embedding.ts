import { InferenceClient } from "@huggingface/inference";

const HF_API_KEY = process.env.HUGGINGFACE_API_KEY;
const HF_MODEL = process.env.EMBEDDING_MODEL;

if (!HF_API_KEY) {
  console.warn("HUGGINGFACE_API_KEY is not set. Embedding calls will fail.");
}

if (!HF_MODEL) {
  console.warn("EMBEDDING_MODEL is not set. Embedding calls will fail.");
}

let hfClient: InferenceClient | null = null;

function getClient(): InferenceClient {
  if (!hfClient) {
    if (!HF_API_KEY) {
      throw new Error("HUGGINGFACE_API_KEY is not set.");
    }
    hfClient = new InferenceClient(HF_API_KEY);
  }
  return hfClient;
}

export async function embedQuery(text: string): Promise<number[]> {
  if (!HF_API_KEY || !HF_MODEL) {
    throw new Error("Embedding configuration missing (HUGGINGFACE_API_KEY or EMBEDDING_MODEL).");
  }

  const client = getClient();

  // Use feature-extraction to obtain a single embedding vector for the query text.
  const result = await client.featureExtraction({
    model: HF_MODEL,
    inputs: text
  });

  // result is number[][] or number[]
  if (Array.isArray(result) && Array.isArray(result[0])) {
    return (result[0] as unknown[]).map((v) => Number(v));
  }

  if (Array.isArray(result)) {
    return (result as unknown[]).map((v) => Number(v));
  }

  throw new Error("Unexpected embedding response format from Hugging Face Inference client.");
}

