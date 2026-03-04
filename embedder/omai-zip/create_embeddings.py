"""
Application to create embeddings from recipe JSON data and upload to MongoDB
"""
import json
import sys
import time
import argparse
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from pymongo import MongoClient
from bson import ObjectId
from huggingface_hub import InferenceClient
import numpy as np
from config import (
    MONGODB_URI,
    MONGODB_DATABASE,
    MONGODB_COLLECTION,
    EMBEDDING_MODEL,
    BATCH_SIZE,
    JSON_FILE_PATH,
    HUGGINGFACE_API_KEY,
    INFERENCE_PROVIDER
)


def load_recipes(file_path: str, offset: int = 0, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load recipes from JSON file (one JSON object per line)
    
    Args:
        file_path: Path to the JSON file
        offset: Number of recipes to skip from the beginning (default: 0)
        limit: Maximum number of recipes to load (None = all remaining)
    
    Returns:
        List of recipe dictionaries
    """
    recipes = []
    skipped = 0
    loaded_count = 0
    
    print(f"Loading recipes from {file_path}...")
    if offset > 0:
        print(f"Skipping first {offset} recipes...")
    if limit:
        print(f"Loading up to {limit} recipes...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Skip until we reach the offset
                if skipped < offset:
                    skipped += 1
                    continue
                
                # Stop if we've reached the limit
                if limit and loaded_count >= limit:
                    break
                
                try:
                    recipe = json.loads(line)
                    recipes.append(recipe)
                    loaded_count += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(recipes)} recipes (skipped {skipped}, offset: {offset})")
        if limit:
            print(f"Processed {loaded_count} recipes (limit: {limit})")
        return recipes
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)


def create_text_for_embedding(recipe: Dict[str, Any]) -> str:
    """
    Combine recipe fields into a single text string for embedding
    """
    parts = []
    
    # Add name if available
    if 'name' in recipe and recipe['name']:
        parts.append(f"Recipe: {recipe['name']}")
    
    # Add description if available
    if 'description' in recipe and recipe['description']:
        parts.append(f"Description: {recipe['description']}")
    
    # Add ingredients if available
    if 'ingredients' in recipe and recipe['ingredients']:
        parts.append(f"Ingredients: {recipe['ingredients']}")
    
    # Combine all parts
    text = " ".join(parts)
    
    # If no text was created, use a default
    if not text.strip():
        text = "Recipe"
    
    return text


def create_single_embedding(client: InferenceClient, text: str, max_retries: int = 3) -> Optional[np.ndarray]:
    """
    Create embedding for a single text with retry logic
    
    Returns:
        numpy array of embedding or None if failed
    """
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            result = client.feature_extraction(
                text=text,
                model=EMBEDDING_MODEL
            )
            
            # Convert result to numpy array
            if isinstance(result, list):
                embedding = np.array(result, dtype=np.float32)
            elif hasattr(result, 'tolist'):
                embedding = np.array(result.tolist(), dtype=np.float32)
            else:
                embedding = np.array(list(result), dtype=np.float32)
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed
                return None
            else:
                # Wait and retry
                time.sleep(retry_delay * (2 ** attempt))
    
    return None


def create_embeddings(client: InferenceClient, texts: List[str], batch_size: int) -> np.ndarray:
    """
    Create embeddings for a list of texts using Hugging Face Inference API
    """
    print(f"Creating embeddings for {len(texts)} texts using Inference API...")
    embeddings = []
    
    # Process texts in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i + batch_size]
        
        # Retry logic for API calls
        max_retries = 3
        retry_delay = 2  # seconds
        success = False
        
        for attempt in range(max_retries):
            try:
                # Use feature_extraction task for embeddings
                # API uses 'text' as keyword argument (can be string or list)
                result = client.feature_extraction(
                    text=batch if len(batch) > 1 else batch[0],  # Pass list for batch, string for single
                    model=EMBEDDING_MODEL
                )
                
                # Handle result - could be single embedding or list of embeddings
                if isinstance(result, list):
                    # Check if it's a list of embeddings (batch) or single embedding as list
                    if len(result) > 0 and isinstance(result[0], list):
                        # List of embeddings (batch)
                        embeddings.extend(result)
                    else:
                        # Single embedding returned as list
                        embeddings.append(result)
                else:
                    # Single embedding (numpy array or similar)
                    if hasattr(result, 'tolist'):
                        embeddings.append(result.tolist())
                    else:
                        embeddings.append(list(result))
                
                success = True
                break  # Success, exit retry loop
                
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                is_last_attempt = (attempt == max_retries - 1)
                
                # Check for 404 (model not found) - non-retryable
                is_404 = "404" in error_msg or "not found" in error_msg.lower()
                
                # Check if it's a retryable error (network, timeout, rate limit, SSL)
                retryable_errors = [
                    "timeout", "connection", "network", "rate limit", 
                    "503", "502", "500", "429", "httpx", "ssl", "tls",
                    "connectionerror", "timeouterror", "readtimeout"
                ]
                retryable_types = [
                    "ConnectionError", "TimeoutError", "ReadTimeout",
                    "ConnectTimeout", "SSLError", "HTTPError"
                ]
                is_retryable = (
                    not is_404 and  # Don't retry 404 errors
                    (any(err in error_msg.lower() for err in retryable_errors) or
                     any(err_type in error_type for err_type in retryable_types))
                )
                
                if is_last_attempt or not is_retryable:
                    print(f"\nError creating embeddings for batch {i//batch_size + 1} (attempt {attempt + 1}/{max_retries})")
                    print(f"  Error type: {error_type}")
                    print(f"  Error message: {error_msg[:200]}...")  # Truncate long messages
                    
                    if is_404:
                        print(f"\n  ⚠️  Model '{EMBEDDING_MODEL}' not found (404)")
                        print("  This usually means:")
                        print("    - The model name is incorrect")
                        print("    - The model doesn't exist on Hugging Face")
                        print("    - The model path is wrong")
                        print("\n  Suggested models:")
                        print("    - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                        print("    - sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
                        print("    - intfloat/multilingual-e5-base")
                        print("    - BAAI/bge-small-en-v1.5")
                        print("\n  Update EMBEDDING_MODEL in your .env file")
                    
                    if not is_retryable and not is_404:
                        print("  (Non-retryable error - skipping retry)")
                    
                    # Fallback: create zero embeddings for failed batch
                    embedding_dim = len(embeddings[0]) if embeddings else 384
                    embeddings.extend([[0.0] * embedding_dim] * len(batch))
                    break
                else:
                    # Retryable error - wait and retry
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"\nWarning: Batch {i//batch_size + 1} failed (attempt {attempt + 1}/{max_retries})")
                    print(f"  Error: {error_type} - {error_msg[:150]}...")
                    print(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Normalize embeddings for better similarity search
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_array = embeddings_array / norms
    
    return embeddings_array


def upload_single_recipe(collection, recipe: Dict[str, Any], embedding: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Upload a single recipe with embedding to MongoDB
    
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    try:
        doc = recipe.copy()
        doc['embedding'] = embedding.tolist()
        
        # Handle MongoDB _id field - convert from {"$oid": "..."} format if needed
        recipe_id = None
        if '_id' in doc:
            if isinstance(doc['_id'], dict) and '$oid' in doc['_id']:
                # Convert MongoDB ObjectId format
                recipe_id = ObjectId(doc['_id']['$oid'])
                doc['_id'] = recipe_id
            elif isinstance(doc['_id'], str):
                # Try to convert string to ObjectId
                try:
                    recipe_id = ObjectId(doc['_id'])
                    doc['_id'] = recipe_id
                except Exception:
                    # Keep as string if conversion fails
                    recipe_id = doc['_id']
            else:
                recipe_id = doc['_id']
        
        # Use upsert to handle duplicates
        if recipe_id:
            filter_query = {'_id': recipe_id}
            result = collection.replace_one(filter_query, doc, upsert=True)
            if result.upserted_id or result.modified_count > 0 or result.matched_count > 0:
                return True, None
            else:
                return False, "No document was inserted or updated"
        else:
            # No _id field, just insert
            collection.insert_one(doc)
            return True, None
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return False, error_msg


def upload_to_mongodb(recipes: List[Dict[str, Any]], embeddings: np.ndarray, clear_collection: bool = False):
    """
    Upload recipes with their embeddings to MongoDB
    
    Args:
        recipes: List of recipe dictionaries
        embeddings: Numpy array of embeddings
        clear_collection: If True, clear the collection before uploading (default: False)
    """
    print(f"Connecting to MongoDB at {MONGODB_URI}...")
    
    try:
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DATABASE]
        collection = db[MONGODB_COLLECTION]
        
        # Clear existing collection only if requested (usually only for first batch)
        if clear_collection:
            print(f"Clearing existing collection '{MONGODB_COLLECTION}'...")
            collection.delete_many({})
        else:
            print(f"Uploading to existing collection '{MONGODB_COLLECTION}' (not clearing)")
        
        print(f"Uploading {len(recipes)} recipes with embeddings to MongoDB...")
        
        documents = []
        for recipe, embedding in zip(tqdm(recipes, desc="Preparing documents"), embeddings):
            doc = recipe.copy()
            doc['embedding'] = embedding.tolist()  # Convert numpy array to list for MongoDB
            documents.append(doc)
            
            # Insert in batches for better performance
            if len(documents) >= BATCH_SIZE:
                try:
                    collection.insert_many(documents, ordered=False)  # Continue on duplicate key errors
                except Exception as e:
                    # Handle duplicate key errors gracefully (useful for batch processing)
                    if "duplicate key" in str(e).lower() or "E11000" in str(e):
                        # Try inserting one by one to skip duplicates
                        for single_doc in documents:
                            try:
                                collection.insert_one(single_doc)
                            except Exception:
                                pass  # Skip duplicates
                    else:
                        raise
                documents = []
        
        # Insert remaining documents
        if documents:
            try:
                collection.insert_many(documents, ordered=False)
            except Exception as e:
                if "duplicate key" in str(e).lower() or "E11000" in str(e):
                    for single_doc in documents:
                        try:
                            collection.insert_one(single_doc)
                        except Exception:
                            pass
                else:
                    raise
        
        # Create index on embedding field for vector search (if MongoDB supports it)
        try:
            # Note: This requires MongoDB Atlas with vector search or MongoDB 6.0.11+ with vector search
            # For now, we'll just create a regular index
            collection.create_index("name")
            print("Created index on 'name' field")
        except Exception as e:
            print(f"Note: Could not create vector index: {e}")
            print("You may need MongoDB Atlas with vector search enabled for vector similarity search")
        
        print(f"Successfully uploaded {len(recipes)} recipes to MongoDB")
        print(f"Database: {MONGODB_DATABASE}")
        print(f"Collection: {MONGODB_COLLECTION}")
        
        client.close()
        
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        print("\nPlease ensure MongoDB is running and the connection string is correct.")
        print("You can set MONGODB_URI in a .env file or environment variable.")
        sys.exit(1)


def load_log_file(log_file_path: str) -> Dict[str, Any]:
    """
    Load existing log file or create new one
    """
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    
    # Create new log structure
    return {
        "start_time": datetime.now().isoformat(),
        "total_processed": 0,
        "successful": 0,
        "failed": 0,
        "recipes": []
    }


def save_log_file(log_file_path: str, log_data: Dict[str, Any]):
    """
    Save log data to JSON file
    """
    log_data["last_updated"] = datetime.now().isoformat()
    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)


def log_recipe_result(log_data: Dict[str, Any], recipe_index: int, recipe_id: str, 
                     recipe_name: str, status: str, error: Optional[str] = None):
    """
    Log a recipe processing result
    """
    log_entry = {
        "index": recipe_index,
        "recipe_id": recipe_id,
        "recipe_name": recipe_name,
        "status": status,  # "success" or "failed"
        "timestamp": datetime.now().isoformat()
    }
    
    if error:
        log_entry["error"] = str(error)
    
    log_data["recipes"].append(log_entry)
    log_data["total_processed"] += 1
    
    if status == "success":
        log_data["successful"] += 1
    else:
        log_data["failed"] += 1


def process_recipes_one_by_one(file_path: str, client: InferenceClient, 
                               mongo_collection, log_file_path: str,
                               offset: int = 0, limit: Optional[int] = None,
                               ignore_log: bool = False):
    """
    Process recipes one at a time: read -> create embedding -> upload -> log
    
    Args:
        ignore_log: If True, ignore existing log entries and process all recipes
    """
    # Load or create log file
    log_data = load_log_file(log_file_path)
    
    # Get already processed recipe IDs to skip (only if not ignoring log)
    if ignore_log or offset > 0:
        processed_ids = set()
        if offset > 0:
            print(f"Offset specified ({offset}), ignoring log file for skipping recipes")
        else:
            print("Ignore log enabled, processing all recipes")
    else:
        processed_ids = {entry.get("recipe_id") for entry in log_data.get("recipes", []) 
                        if entry.get("status") == "success"}
        print(f"Found {len(processed_ids)} already processed recipes in log")
    
    recipe_index = offset
    processed_count = 0
    skipped_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip to offset
            for _ in range(offset):
                next(f, None)
            
            # Process recipes one by one
            for line in tqdm(f, desc="Processing recipes", initial=offset):
                # Check limit
                if limit and processed_count >= limit:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    recipe = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                recipe_id = str(recipe.get('_id', {}).get('$oid', recipe_index))
                recipe_name = recipe.get('name', 'Unknown')
                
                # Skip if already processed successfully
                if recipe_id in processed_ids:
                    skipped_count += 1
                    recipe_index += 1
                    continue
                
                # Create text for embedding
                text = create_text_for_embedding(recipe)
                
                # Print the text that will be embedded
                print(f"\n[{recipe_index}] Recipe: {recipe_name}")
                print(f"Text to embed: {text[:200]}{'...' if len(text) > 200 else ''}")
                print("-" * 80)
                
                # Create embedding
                embedding = create_single_embedding(client, text)
                
                if embedding is None:
                    # Failed to create embedding
                    log_recipe_result(log_data, recipe_index, recipe_id, recipe_name, 
                                    "failed", "Failed to create embedding")
                    save_log_file(log_file_path, log_data)
                    recipe_index += 1
                    continue
                
                # Upload to MongoDB
                success, error_msg = upload_single_recipe(mongo_collection, recipe, embedding)
                
                if success:
                    log_recipe_result(log_data, recipe_index, recipe_id, recipe_name, "success")
                    print(f"✓ Successfully uploaded to MongoDB")
                else:
                    error_detail = error_msg or "Failed to upload to MongoDB"
                    print(f"✗ Failed to upload: {error_detail}")
                    log_recipe_result(log_data, recipe_index, recipe_id, recipe_name, 
                                    "failed", error_detail)
                
                # Save log after each recipe (for recovery)
                save_log_file(log_file_path, log_data)
                
                recipe_index += 1
                processed_count += 1
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")
        save_log_file(log_file_path, log_data)
        print(f"Progress saved. Processed {processed_count} recipes.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError processing recipes: {e}")
        save_log_file(log_file_path, log_data)
        raise
    
    print(f"\nProcessing complete!")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (already done): {skipped_count}")
    print(f"  Successful: {log_data['successful']}")
    print(f"  Failed: {log_data['failed']}")
    print(f"  Log file: {log_file_path}")


def main():
    """
    Main function to orchestrate the embedding creation and upload process
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate embeddings from recipe JSON data and upload to MongoDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all recipes
  python create_embeddings.py
  
  # Process first 1000 recipes
  python create_embeddings.py --limit 1000
  
  # Process recipes starting from index 5000, up to 1000 recipes
  python create_embeddings.py --offset 5000 --limit 1000
  
  # Resume from recipe 10000
  python create_embeddings.py --offset 10000
        """
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        help='Number of recipes to skip from the beginning (default: 0)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of recipes to process (default: all remaining)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='recipe_processing_log.json',
        help='Path to JSON log file (default: recipe_processing_log.json)'
    )
    parser.add_argument(
        '--ignore-log',
        action='store_true',
        help='Ignore existing log entries and process all recipes (useful with --offset)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Recipe Embedding Generator (One-by-One Processing)")
    print("=" * 60)
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"JSON File: {JSON_FILE_PATH}")
    print(f"MongoDB Database: {MONGODB_DATABASE}")
    print(f"MongoDB Collection: {MONGODB_COLLECTION}")
    print(f"Log File: {args.log_file}")
    if args.offset > 0 or args.limit:
        print(f"Offset: {args.offset}")
        print(f"Limit: {args.limit if args.limit else 'unlimited'}")
    print("=" * 60)
    print()
    
    # Initialize Hugging Face Inference Client
    print(f"\nInitializing Hugging Face Inference Client...")
    print(f"Model: {EMBEDDING_MODEL}")
    
    if not HUGGINGFACE_API_KEY:
        print("Error: HUGGINGFACE_API_KEY not found in environment variables.")
        print("Please add HUGGINGFACE_API_KEY to your .env file.")
        sys.exit(1)
    
    # For feature extraction (embeddings), use 'auto' or 'hf-inference'
    if INFERENCE_PROVIDER.lower() == "together":
        embedding_provider = "auto"
        print(f"Note: Using 'auto' provider for embeddings (Together AI doesn't support feature-extraction)")
    else:
        embedding_provider = INFERENCE_PROVIDER.lower()
    
    try:
        client = InferenceClient(
            provider=embedding_provider,
            api_key=HUGGINGFACE_API_KEY,
        )
        print(f"Inference client initialized successfully with provider: {embedding_provider}")
    except Exception as e:
        print(f"Error initializing Inference Client: {e}")
        print("\nPlease check your HUGGINGFACE_API_KEY.")
        sys.exit(1)
    
    # Connect to MongoDB
    print(f"\nConnecting to MongoDB at {MONGODB_URI}...")
    try:
        mongo_client = MongoClient(MONGODB_URI)
        db = mongo_client[MONGODB_DATABASE]
        collection = db[MONGODB_COLLECTION]
        print("MongoDB connected successfully!")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        sys.exit(1)
    
    # Process recipes one by one
    print("\nStarting one-by-one processing...")
    print("Press Ctrl+C to stop and save progress\n")
    
    try:
        # If offset is specified, ignore log to respect the offset
        ignore_log = args.ignore_log or (args.offset > 0)
        
        process_recipes_one_by_one(
            JSON_FILE_PATH,
            client,
            collection,
            args.log_file,
            offset=args.offset,
            limit=args.limit,
            ignore_log=ignore_log
        )
        
        print("\n" + "=" * 60)
        print("Process completed successfully!")
        print("=" * 60)
        
    finally:
        mongo_client.close()


if __name__ == "__main__":
    main()
