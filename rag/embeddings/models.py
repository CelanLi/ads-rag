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


def get_embedding_model(name: str) -> BaseEmbeddingModel:
    name = name.lower()
    backend = AVAILABLE_EMBEDDING_MODELS[name]["backend"]
    if backend == "openai":
        return OpenAIEmbeddingModel()
    elif backend == "gemini":
        return GeminiEmbeddingModel()
    else:
        raise ValueError(f"Unknown embedding model: {name}")
