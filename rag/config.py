# rag/config.py
# -----------------------------------
# This module contains the configuration for the RAG system.
# -----------------------------------

from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()  # this reads the .env file and sets os.environ

# Access API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 1. embedding model
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
AVAILABLE_EMBEDDING_MODELS = {
    "text-embedding-3-small": {"backend": "openai", "api_key": OPENAI_API_KEY}, # OpenAI
    "text-embedding-3-large": {"backend": "openai", "api_key": OPENAI_API_KEY}, # OpenAI
    "gemini-embedding-001": {"backend": "gemini", "api_key": GEMINI_API_KEY}, # Google
    "Qwen3-Embedding-8B": {"backend": "open_source", "api_key": None}, # open source
    "Qwen3-Embedding-4B": {"backend": "open_source", "api_key": None}, # open source
    "all-MiniLM-L6-v2": {"backend": "open_source", "api_key": None}, # open source
    "Linq-Embed-Mistral": {"backend": "open_source", "api_key": None}, # open source
}

# 2. LLM models
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
AVAILABLE_LLMS = {
    "gpt-4.1-mini": {"backend": "openai", "api_key": OPENAI_API_KEY},
    "gemini-2.5-flash": {"backend": "gemini", "api_key": GEMINI_API_KEY},
}

# 3. chunk settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 100