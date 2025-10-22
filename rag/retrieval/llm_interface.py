# rag/retrieval/llm_interface.py
# -----------------------------------
# This module is the LLM interface for the RAG system. It handle the query, retrieve the most similar chunks, and generate the response.
#
# usage:
#     python -m rag.retrieval.llm_interface
# -----------------------------------

from rag.embeddings.embedding_manager import EmbeddingManager
from rag.llm.models import get_llm_model
from rag.retrieval.retriever import Retriever


class LLMInterface:
    def __init__(self, model_name: str, embedding_manager: EmbeddingManager):
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
        rag_prompt_template_path = "rag/prompts/llm/rag_qa.md"
        response = self.llm_model.generate_rag_response(rag_prompt_template_path, context, query)
        return response

if __name__ == "__main__":
    llm_interface = LLMInterface(model_name="gemini-2.5-flash", embedding_manager=EmbeddingManager())
    print(llm_interface.rag_qa("What is BIM model?"))