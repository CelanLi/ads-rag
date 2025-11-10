# rag/embeddings/models.py
# -----------------------------------
# This module is the factory for the embedding models.
# -----------------------------------

from abc import ABC
from typing import List, Tuple, Any
from openai import OpenAI
from google import genai
from google.genai import types
import numpy as np
from rag.config import OPENAI_API_KEY, GEMINI_API_KEY, AVAILABLE_EMBEDDING_MODELS


class BaseEmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    """

    def __init__(self, model_name: str, api_key: str, **kwargs):
        """
        Constructor for abstract base class.
        Parameters are accepted for interface consistency but are not stored.
        Subclasses should implement their own initialization as needed.
        """
        pass

    def encode(self, texts: List[str]) -> Tuple[np.ndarray, int]:
        raise NotImplementedError("Please implement encode method!")

    def encode_queries(self, text: str) -> np.ndarray:
        raise NotImplementedError("Please implement encode method!")

    def total_token_count(self, resp: Any) -> int:
        try:
            return resp.usage.total_tokens
        except Exception:
            pass
        try:
            return resp["usage"]["total_tokens"]
        except Exception:
            pass
        return 0


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str = OPENAI_API_KEY,
        base_url: str = "https://api.openai.com/v1",
    ):
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def encode(self, texts: List[str]) -> Tuple[np.ndarray, int]:
        # OpenAI requires batch size <=16
        batch_size = 16
        embeddings = []
        total_tokens = 0
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name,
                encoding_format="float",
                extra_body={"drop_params": True},
            )
            embeddings.extend([d.embedding for d in response.data])
            total_tokens += self.total_token_count(response)
        return np.array(embeddings), total_tokens

    def encode_queries(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name,
            encoding_format="float",
            extra_body={"drop_params": True},
        )
        return np.array(response.data[0].embedding)


class GeminiEmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self, model_name: str = "gemini-embedding-001", api_key: str = GEMINI_API_KEY
    ):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def encode(self, texts: List[str]) -> Tuple[np.ndarray, int]:
        batch_size = AVAILABLE_EMBEDDING_MODELS[self.model_name]["batch_size"]
        total_tokens = 0
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )
            result = np.array([np.array(e.values) for e in response.embeddings])
            total_tokens += self.total_token_count(response)
            return result, total_tokens

    def encode_queries(self, text: str) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        result = np.array([np.array(e.values) for e in response.embeddings])
        return result


class QwenEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str, api_key: str = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(f"Qwen/{model_name}")

    def encode(self, texts: List[str]) -> Tuple[np.ndarray, int]:
        if not texts:
            # Return empty 2D array with correct shape for FAISS
            return np.array([]).reshape(
                0, self.model.get_sentence_embedding_dimension()
            ), 0

        embeddings = self.model.encode(texts)
        # Ensure embeddings is 2D: (n_samples, embedding_dim)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # SentenceTransformer doesn't provide token count, return 0
        total_tokens = 0
        for text in texts:
            total_tokens += len(text.split())
        return embeddings, total_tokens

    def encode_queries(self, text: str) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        return self.model.encode(text, prompt_name="query")


class AllMiniLML6V2EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def encode(self, texts: List[str]) -> Tuple[np.ndarray, int]:
        if not texts:
            # Return empty 2D array with correct shape for FAISS
            return np.array([]).reshape(
                0, self.model.get_sentence_embedding_dimension()
            ), 0

        embeddings = self.model.encode(texts)
        # Ensure embeddings is 2D: (n_samples, embedding_dim)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # SentenceTransformer doesn't provide token count, return 0
        total_tokens = 0
        for text in texts:
            total_tokens += len(text.split())
        return embeddings, total_tokens

    def encode_queries(self, text: str) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        return self.model.encode(text, prompt_name="query")


class LinqEmbedMistralEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = "Linq-Embed-Mistral"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer("Linq-AI-Research/Linq-Embed-Mistral")

    def encode(self, texts: List[str]) -> Tuple[np.ndarray, int]:
        if not texts:
            # Return empty 2D array with correct shape for FAISS
            return np.array([]).reshape(
                0, self.model.get_sentence_embedding_dimension()
            ), 0

        embeddings = self.model.encode(texts)
        # Ensure embeddings is 2D: (n_samples, embedding_dim)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # SentenceTransformer doesn't provide token count, return 0
        total_tokens = 0
        for text in texts:
            total_tokens += len(text.split())
        return embeddings, total_tokens

    def encode_queries(self, text: str) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        return self.model.encode(text, prompt_name="query")


def get_embedding_model(name: str) -> BaseEmbeddingModel:
    backend = AVAILABLE_EMBEDDING_MODELS[name]["backend"]
    if backend == "openai":
        return OpenAIEmbeddingModel(model_name=name)
    elif backend == "gemini":
        return GeminiEmbeddingModel()
    elif backend == "qwen":
        return QwenEmbeddingModel(model_name=name)
    elif backend == "all-MiniLM-L6-v2":
        return AllMiniLML6V2EmbeddingModel(model_name=name)
    elif backend == "Linq-Embed-Mistral":
        return LinqEmbedMistralEmbeddingModel(model_name=name)
    else:
        raise ValueError(f"Unknown embedding model: {name}")


if __name__ == "__main__":
    model = get_embedding_model("Linq-Embed-Mistral")
    print(
        model.encode(
            [
                "Hello, world!",
                "Hello, universe!",
                "Hello, galaxy!",
                "The capital of France is Paris.",
                "The capital of Germany is Berlin.",
                "The capital of Italy is Rome.",
                "The capital of Spain is Madrid.",
                "The capital of Portugal is Lisbon.",
                "The capital of Greece is Athens.",
                "The capital of Turkey is Ankara.",
            ]
        )
    )
    print(model.encode_queries("Hello, world!"))
