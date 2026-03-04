"""
Configuration file for MongoDB connection and embedding settings
"""
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "recipe_embeddings")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "recipes")

# Hugging Face Model Configuration
# Using a multilingual model that supports embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Hugging Face Inference API Configuration
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
# Note: For embeddings (feature-extraction), use 'auto' or 'hf-inference'
# Together AI doesn't support feature-extraction in huggingface_hub (only text-generation, conversational, text-to-image)
# The code will automatically use 'auto' for embeddings if you specify 'together'
INFERENCE_PROVIDER = os.getenv("INFERENCE_PROVIDER", "auto")  # Options: auto (recommended), hf-inference, together (for text generation only)

# Processing Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
JSON_FILE_PATH = os.getenv("JSON_FILE_PATH", "json-source/20170107-061401-recipeitems.json")
