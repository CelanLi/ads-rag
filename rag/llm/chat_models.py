from abc import ABC
from pathlib import Path
from typing import Dict, List

from google import genai
from openai import OpenAI
from langchain.prompts import PromptTemplate

from rag.config import AVAILABLE_LLMS

RAG_QA_PROMPT_PATH = "rag/prompts/llm/rag_qa.md"
ADS_RAG_PROMPT_PATH = "rag/prompts/llm/ads_rag.md"


class BaseLLMModel(ABC):
    """
    Abstract base class for LLM models.
    """

    def __init__(self, model_name: str, api_key: str, **kwargs):
        """
        Constructor for abstract base class.
        Parameters are accepted for interface consistency but are not stored.
        Subclasses should implement their own initialization as needed.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs

    def get_prompt(self, prompt_path: str) -> str:
        """
        Reads a .md file and returns its content as a string.
        """
        path = Path(prompt_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return path.read_text(encoding="utf-8")

    def generate_rag_response(
        self,
        context: List[str],
        question: str,
        prompt_template_path: str = RAG_QA_PROMPT_PATH,
    ) -> str:
        """
        Generate a response for a given context and question.
        """
        raise NotImplementedError("Please implement generate_response method!")

    def generate_ads_rag_response(
        self,
        context: List[Dict],
        query: str,
        prompt_template_path: str = ADS_RAG_PROMPT_PATH,
    ) -> str:
        """
        Generate an autonomous driving scenario with the given requirements and xml as reference.
        """
        raise NotImplementedError("Please implement generate_ads_rag_response method!")


class OpenAILLMModel(BaseLLMModel):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.client = OpenAI(api_key=api_key)

    def generate_rag_response(
        self,
        context: List[str],
        question: str,
        prompt_template_path: str = RAG_QA_PROMPT_PATH,
    ) -> str:
        prompt_str = self.get_prompt(prompt_template_path)
        prompt_template = PromptTemplate(
            input_variables=["context", "question"], template=prompt_str
        )
        # Flatten context if it contains nested lists
        flat_context = []
        for c in context:
            if isinstance(c, list):
                flat_context.extend(c)
            else:
                flat_context.append(str(c))  # make sure it's a string

        formatted_prompt = prompt_template.format(
            context="\n".join(flat_context), question=question
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
        )
        return response.choices[0].message.content.strip()

    def generate_ads_rag_response(
        self,
        context: List[str],
        query: str,
        prompt_template_path: str = ADS_RAG_PROMPT_PATH,
    ) -> str:
        pass


class GeminiLLMModel(BaseLLMModel):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.client = genai.Client(api_key=api_key)

    def generate_rag_response(
        self,
        context: List[str],
        question: str,
        prompt_template_path: str = RAG_QA_PROMPT_PATH,
    ) -> str:
        prompt_str = self.get_prompt(prompt_template_path)
        prompt_template = PromptTemplate(
            input_variables=["context", "question"], template=prompt_str
        )

        # Flatten context if it contains nested lists
        flat_context = []
        for c in context:
            if isinstance(c, list):
                flat_context.extend(c)
            else:
                flat_context.append(str(c))  # make sure it's a string

        formatted_prompt = prompt_template.format(
            context="\n".join(flat_context), question=question
        )
        response = self.client.models.generate_content(
            model=self.model_name, contents=[formatted_prompt]
        )
        return response.text

    def generate_ads_rag_response(
        self,
        contexts: List[str],
        query: str,
        prompt_template_path: str = ADS_RAG_PROMPT_PATH,
    ) -> str:
        # filter contexts
        contexts = [c for c in contexts if c.get("category") == "sumo-scenario"]

        # collect unique xml paths
        xml_paths: set[str] = set()
        for context in contexts:
            for path in context.get("src_path", []):
                if path.lower().endswith(".xml"):
                    xml_paths.add(path)

        # read and combine xml content
        xml_content = "\n".join(
            Path(path).read_text(encoding="utf-8")
            for path in xml_paths
            if Path(path).is_file()
        )

        # generate prompt
        prompt_template = self.get_prompt(prompt_template_path)
        prompt_template = PromptTemplate(
            input_variables=["context", "query"], template=prompt_template
        )
        formatted_prompt = prompt_template.format(context=xml_content, query=query)
        response = self.client.models.generate_content(
            model=self.model_name, contents=[formatted_prompt]
        ).text

        return response


def get_llm_model(model_name: str) -> BaseLLMModel:
    """
    Factory function to return the appropriate LLM model instance.
    """
    model_name_lower = model_name.lower()
    if model_name_lower not in AVAILABLE_LLMS.keys():
        raise ValueError(f"Unsupported model: {model_name}")

    backend = AVAILABLE_LLMS[model_name_lower]["backend"]
    if backend == "openai":
        return OpenAILLMModel(
            model_name=model_name, api_key=AVAILABLE_LLMS[model_name_lower]["api_key"]
        )
    elif backend == "gemini":
        return GeminiLLMModel(
            model_name=model_name, api_key=AVAILABLE_LLMS[model_name_lower]["api_key"]
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
