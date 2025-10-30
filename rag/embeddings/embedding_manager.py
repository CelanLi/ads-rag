# rag/embeddings/embedding_manager.py
# -----------------------------------
# This module loads chunked data,
# generates embeddings using the selected model (e.g. OpenAI or SentenceTransformers),
# and saves them into a vector store (e.g. FAISS, Chroma, or local file).
#
# usage:
#     python -m rag.embeddings.embedding_manager
# -----------------------------------

import faiss
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import numpy as np
import json

from pydantic import BaseModel

from rag.config import AVAILABLE_EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL
from rag.embeddings.chunk_text import ChunkMetadata
from rag.llm.embedding_models import get_embedding_model
from utils.file_utils import export_json


class EmbeddingMetadata(BaseModel):
    text: str
    src_path: List[str]
    index: int
    category: str

    def model_dump(self) -> dict:
        return {
            "text": self.text,
            "src_path": self.src_path,
            "index": self.index,
            "category": self.category,
        }


class EmbeddingManager:
    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        vector_store_dir: str = "data/vector_db/",
    ):
        """
        Initialize the embedding manager with a chosen model.
        """
        self.embedding_model = embedding_model
        self.vector_store_dir = Path(vector_store_dir)
        self.vector_store_path = (
            self.vector_store_dir / f"{embedding_model}_vector_store.index"
        )  # Each model gets its own index file
        self.metadata_path = (
            self.vector_store_dir / f"{embedding_model}_metadata.json"
        )  # Each model gets its own metadata file, includes the index and other attributes

        self.backend = AVAILABLE_EMBEDDING_MODELS[embedding_model]["backend"]

        self._init_model()
        self._init_vector_store()
        self._init_metadata()

        # assert the length of metadata and the number of chunks in the vector store are the same
        assert len(self.metadata) == self.index.ntotal, (
            f"The length of metadata and the number of chunks in the vector store are not the same. Metadata length: {len(self.metadata)}, Vector store length: {self.index.ntotal}"
        )

    def _init_model(self):
        """Initialize embedding model according to backend."""
        self.model = get_embedding_model(self.embedding_model)

    def _init_vector_store(self):
        """Initialize FAISS vector store (create or load)."""
        self.vector_dim = AVAILABLE_EMBEDDING_MODELS[self.embedding_model]["vector_dim"]

        if self.vector_store_path.exists():
            try:
                self.index = faiss.read_index(str(self.vector_store_path))
                print(
                    f"Loaded existing FAISS index from {self.vector_store_path} for {self.embedding_model}"
                )
            except Exception as e:
                logging.warning(
                    f"Failed to load existing FAISS index: {e}. Creating a new one for {self.embedding_model}."
                )
                self.index = faiss.IndexFlatL2(self.vector_dim)
        else:
            self.index = faiss.IndexFlatL2(self.vector_dim)
            print(
                f"Created new FAISS index with dimension {self.vector_dim} for {self.embedding_model}"
            )

    def _init_metadata(self):
        """Initialize metadata for the vector store."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    print(
                        f"Loaded existing metadata, the total number of chunks is {len(self.metadata)}"
                    )
            except Exception as e:
                logging.warning(
                    f"Failed to load existing metadata: {e}. Creating a new one for {self.embedding_model}."
                )
                self.metadata = []
        else:
            self.metadata = []

    def _construct_metadata(self, chunk_data: ChunkMetadata):
        """Construct a metadata record for a chunk."""
        for chunk in chunk_data.chunks:
            chunk_data = EmbeddingMetadata(
                text=chunk,
                src_path=chunk_data.src_path,
                index=len(self.metadata),
                category=chunk_data.category,
            )
            self.metadata.append(chunk_data.model_dump())

    def _save_metadata(self):
        """Save metadata to disk."""
        export_json(
            output_dir=self.vector_store_dir,
            file_name=f"{self.embedding_model}_metadata",
            content=self.metadata,
        )

    def _save_vector_store(self):
        """Persist FAISS index to disk."""
        index = faiss.write_index(self.index, str(self.vector_store_path))
        return index

    def embed_chunks(self, text: str | List[str]) -> Tuple[np.ndarray, int]:
        """Generate embeddings for one or more text chunks."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        embeddings, total_tokens = self.model.encode(texts)
        return embeddings, total_tokens

    def add_embeddings(self, chunk_data: ChunkMetadata):
        """Embed and add text chunks to FAISS vector store."""
        embeddings, total_tokens = self.embed_chunks(chunk_data.chunks)
        self.index.add(embeddings)
        self._save_vector_store()
        print(
            f"Added {len(chunk_data.chunks)} chunks to FAISS index. Used {total_tokens} tokens."
        )

        self._construct_metadata(chunk_data)
        self._save_metadata()
        print(
            f"Added {len(chunk_data.chunks)} chunks to metadata. Total number of chunks is {len(self.metadata)}"
        )

    def search(
        self, query: str | List[str], top_k: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Dict]]]:
        """Search similar chunks for a given query."""
        if isinstance(query, str):
            query = [query]
        else:
            query = query

        query_embs, _ = self.embed_chunks(query)
        distances, indices = self.index.search(query_embs, top_k)
        print("indices:", indices)
        metadata_list = []
        for index_list in indices:
            metadata_list.append([self.metadata[index] for index in index_list])
        return distances, indices, metadata_list

    def embed_all_chunks(self, chunks_dir: str = "data/chunks"):
        "embed all chunks in the chunks_dir"
        for chunk_file in Path(chunks_dir).glob("*.json"):
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunk_data = ChunkMetadata(**json.load(f))
                self.add_embeddings(chunk_data)


if __name__ == "__main__":
    embedding_manager = EmbeddingManager()
    # embedding_manager.add_embeddings(["Hello, world!", "Hello, universe!", "Hello, galaxy!", "The capital of France is Paris.", "The capital of Germany is Berlin.", "The capital of Italy is Rome.", "The capital of Spain is Madrid.", "The capital of Portugal is Lisbon.", "The capital of Greece is Athens.", "The capital of Turkey is Ankara."])
    # distances, indices, metadata = embedding_manager.search(["Berlin is my favorite city.", "Hello, Berlin."])
    # print(distances, indices, metadata)
    embedding_manager.embed_all_chunks()
