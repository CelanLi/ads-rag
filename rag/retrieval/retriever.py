# rag/retrieval/retriever.py
# -----------------------------------
# This module is the query interface for the RAG system. It handle the query, convert it to embedding, then search the vector store to get the most similar chunks.
# After that, it use a prompt template to combine the query, original chunks and feed it into the LLM to get the final answer.
#
# usage:
#     python -m rag.retrieval.retriever
# -----------------------------------

from typing import Dict, List
from rag.embeddings.embedding_manager import EmbeddingManager

class Retriever:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 3) -> str:
        "retrieve the most similar chunks"
        _, _, metadata = self.embedding_manager.search(query, top_k)
        return metadata

    def rerank_results(self, query: str, context: List[str]) -> List[str]:
        pass

    def get_context(self, metadata: List[Dict]) -> str:
        pass

if __name__ == "__main__":
    retriever = Retriever(EmbeddingManager())
    print(retriever.retrieve("What is the usage of diffusion model in autonomous driving?"))