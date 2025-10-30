from abc import ABC
from pathlib import Path
from typing import List

from google import genai
from openai import OpenAI
from langchain.prompts import PromptTemplate

from rag.config import AVAILABLE_LLMS


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
        self, prompt_template_path: str, context: List[str], question: str
    ) -> str:
        raise NotImplementedError("Please implement generate_response method!")


class OpenAILLMModel(BaseLLMModel):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.client = OpenAI(api_key=api_key)

    def generate_rag_response(
        self, prompt_template_path: str, context: List[str], question: str
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


class GeminiLLMModel(BaseLLMModel):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.client = genai.Client(api_key=api_key)

    def generate_rag_response(
        self, prompt_template_path: str, context: List[str], question: str
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
