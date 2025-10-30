# rag/retrieval/llm_interface.py
# -----------------------------------
# This module is the LLM interface for the RAG system. It handle the query, retrieve the most similar chunks, and generate the response.
# -----------------------------------

from typing import List
from rag.config import DEFAULT_LLM_MODEL
from rag.embeddings.embedding_manager import EmbeddingManager
from rag.llm.chat_models import get_llm_model
from rag.retrieval.retriever import Retriever
from utils.file_utils import export_txt


class LLMInterface:
    def __init__(
        self, embedding_manager: EmbeddingManager, model_name: str = DEFAULT_LLM_MODEL
    ):
        self.llm_model = get_llm_model(model_name)
        self.embedding_manager = embedding_manager
        self.retriever = Retriever(embedding_manager)

    def rag_qa(self, query: str | List[str]) -> str:
        # get context
        context = self.retriever.retrieve(query)
        # flatten the context
        context = context[0]
        context = [metadata["text"] for metadata in context]

        # generate the response
        response = self.llm_model.generate_rag_response(context, query)
        return response

    def ads_rag_gen(self, queries: str | List[str]) -> List[str]:
        if isinstance(queries, str):
            queries = [queries]
        contexts = self.retriever.retrieve(queries)
        # assert the length of contexts and queries are the same
        assert len(contexts) == len(queries), (
            "The length of contexts and queries must be the same, got {} and {}".format(
                len(contexts), len(queries)
            )
        )
        responses = []
        for query, context in zip(queries, contexts):
            response = self.llm_model.generate_ads_rag_response(context, query)
            responses.append(response)
        return responses


# usage:
# python -m rag.app.llm_interface
if __name__ == "__main__":
    llm_interface = LLMInterface(
        embedding_manager=EmbeddingManager(), model_name=DEFAULT_LLM_MODEL
    )
    # print(llm_interface.rag_qa("What is BIM model?"))
    responses = llm_interface.ads_rag_gen(
        "Please generate a autonomous driving scenario with the following requirements: 1. The map should contain a crossroad. 2. There is 1 lane on each side of the crossroad."
    )
    print(responses)
    content = "\n".join(responses)
    export_txt(
        output_dir="results/ads_rag",
        file_name="test-0",
        content=content,
    )
