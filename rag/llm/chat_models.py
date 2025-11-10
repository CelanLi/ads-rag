from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional

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

    def _flatten_context(self, context: List[str]) -> List[str]:
        """
        Flatten context if it contains nested lists.
        Convert all items to strings to ensure consistency.

        Args:
            context: List of context strings (may contain nested lists)

        Returns:
            Flattened list of context strings
        """
        flat_context = []
        for c in context:
            if isinstance(c, list):
                flat_context.extend(c)
            else:
                flat_context.append(str(c))  # make sure it's a string
        return flat_context

    def _format_context(self, context: List[str], separator: str = "\n") -> str:
        """
        Format context list into a single string.

        Args:
            context: List of context strings
            separator: String to join contexts with (default: newline)

        Returns:
            Formatted context string
        """
        flattened = self._flatten_context(context)
        return separator.join(flattened)

    def _format_chat_history_as_string(
        self, chat_history: Optional[List[Dict[str, str]]], format_type: str = "simple"
    ) -> str:
        """
        Format chat history as a string for prompt injection.

        Args:
            chat_history: List of messages in format [{"role": "user/assistant", "content": "..."}]
            format_type: Format type - "simple" (User: ...) or "detailed" (role: ...)

        Returns:
            Formatted chat history string
        """
        if not chat_history:
            return "No previous conversation."

        formatted_messages = []
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if format_type == "simple":
                # Format: "User: ..." or "Assistant: ..."
                role_label = "User" if role == "user" else "Assistant"
                formatted_messages.append(f"{role_label}: {content}")
            elif format_type == "detailed":
                # Format: "role: user\ncontent: ..."
                formatted_messages.append(f"role: {role}\ncontent: {content}")
            else:
                # Default: just the content
                formatted_messages.append(content)

        return "\n\n".join(formatted_messages)

    def _format_chat_history_for_api(
        self, chat_history: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Format chat history for API calls (OpenAI style).

        Args:
            chat_history: List of messages in format [{"role": "user/assistant", "content": "..."}]

        Returns:
            List of messages formatted for API
        """
        if not chat_history:
            return []

        return [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in chat_history
        ]

    def _prepare_rag_prompt(
        self,
        context: List[str],
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        prompt_template_path: str = RAG_QA_PROMPT_PATH,
        chat_history_format: str = "string",
    ) -> Dict[str, str]:
        """
        Prepare RAG prompt components.

        Args:
            context: List of context strings
            question: User question
            chat_history: Optional conversation history
            prompt_template_path: Path to prompt template
            chat_history_format: "string" for prompt injection, "api" for API messages

        Returns:
            Dictionary with formatted components:
            - "context": Formatted context string
            - "question": Question string
            - "chat_history": Formatted chat history (string or list depending on format)
            - "prompt_template": PromptTemplate object
        """
        # Format context
        formatted_context = self._format_context(context)

        # Format chat history based on format type
        if chat_history_format == "api":
            formatted_chat_history = self._format_chat_history_for_api(chat_history)
        else:
            # Default: format as string for prompt injection
            formatted_chat_history = self._format_chat_history_as_string(
                chat_history, format_type="simple"
            )

        # Load and create prompt template
        prompt_str = self.get_prompt(prompt_template_path)
        prompt_template = PromptTemplate(
            input_variables=["context", "question", "chat_history"], template=prompt_str
        )

        return {
            "context": formatted_context,
            "question": question,
            "chat_history": formatted_chat_history,
            "prompt_template": prompt_template,
        }

    def generate_rag_response(
        self,
        context: List[str],
        question: str,
        prompt_template_path: str = RAG_QA_PROMPT_PATH,
        chat_history: Optional[List[Dict[str, str]]] = None,
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
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        # Prepare prompt components
        prompt_data = self._prepare_rag_prompt(
            context=context,
            question=question,
            chat_history=chat_history,
            prompt_template_path=prompt_template_path,
            chat_history_format="string",  # Format as string for prompt
        )

        # Format the prompt
        formatted_prompt = prompt_data["prompt_template"].format(
            context=prompt_data["context"],
            question=prompt_data["question"],
            chat_history=prompt_data["chat_history"],
        )

        # Build messages for OpenAI API
        messages = []

        # Add conversation history if provided (for better context)
        if chat_history:
            api_messages = self._format_chat_history_for_api(chat_history)
            messages.extend(api_messages)

        # Add current RAG prompt
        messages.append({"role": "user", "content": formatted_prompt})

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
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
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        # Prepare prompt components
        prompt_data = self._prepare_rag_prompt(
            context=context,
            question=question,
            chat_history=chat_history,
            prompt_template_path=prompt_template_path,
            chat_history_format="string",  # Gemini uses string format in prompt
        )

        # Format the prompt
        formatted_prompt = prompt_data["prompt_template"].format(
            context=prompt_data["context"],
            question=prompt_data["question"],
            chat_history=prompt_data["chat_history"],
        )

        # Call Gemini API
        response = self.client.models.generate_content(
            model=self.model_name, contents=[formatted_prompt]
        )
        return response.text

    def generate_ads_rag_response(
        self,
        contexts: List[Dict],
        query: str,
        prompt_template_path: str = ADS_RAG_PROMPT_PATH,
    ) -> str:
        # Filter contexts
        contexts = [c for c in contexts if c.get("category") == "sumo-scenario"]

        # Collect unique xml paths
        xml_paths: set[str] = set()
        for context in contexts:
            for path in context.get("src_path", []):
                if path.lower().endswith(".xml"):
                    xml_paths.add(path)

        # Read and combine xml content
        xml_content = "\n".join(
            Path(path).read_text(encoding="utf-8")
            for path in xml_paths
            if Path(path).is_file()
        )

        # Generate prompt
        prompt_template_str = self.get_prompt(prompt_template_path)
        prompt_template = PromptTemplate(
            input_variables=["context", "query"], template=prompt_template_str
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
