import { MongoClient } from "mongodb";

const uri = process.env.MONGODB_URI;

if (!uri) {
  throw new Error("MONGODB_URI is not set in the environment");
}

// At this point `uri` is guaranteed to be a string, so we
// assert the type to satisfy TypeScript.
const mongoUri: string = uri;

let client: MongoClient | null = null;
let clientPromise: Promise<MongoClient> | null = null;

export function getMongoClient(): Promise<MongoClient> {
  if (!clientPromise) {
    client = new MongoClient(mongoUri);
    clientPromise = client.connect();
  }

  return clientPromise;
}

