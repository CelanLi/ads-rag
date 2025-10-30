# rag/retrieval/llm_interface.py
# -----------------------------------
# This module is the LLM interface for the RAG system. It handle the query, retrieve the most similar chunks, and generate the response.
# -----------------------------------

from rag.config import DEFAULT_LLM_MODEL
from rag.embeddings.embedding_manager import EmbeddingManager
from rag.llm.chat_models import get_llm_model
from rag.retrieval.retriever import Retriever

RAG_QA_PROMPT_PATH = "rag/prompts/llm/rag_qa.md"


class LLMInterface:
    def __init__(
        self, embedding_manager: EmbeddingManager, model_name: str = DEFAULT_LLM_MODEL
    ):
        self.llm_model = get_llm_model(model_name)
        self.embedding_manager = embedding_manager
        self.retriever = Retriever(embedding_manager)

    def rag_qa(self, query: str) -> str:
        # get context
        context = self.retriever.retrieve(query)
        # flatten the context
        context = context[0]
        context = [metadata["text"] for metadata in context]

        # generate the response
        response = self.llm_model.generate_rag_response(
            RAG_QA_PROMPT_PATH, context, query
        )
        return response


if __name__ == "__main__":
    llm_interface = LLMInterface(
        embedding_manager=EmbeddingManager(), model_name=DEFAULT_LLM_MODEL
    )
    print(llm_interface.rag_qa("What is BIM model?"))
