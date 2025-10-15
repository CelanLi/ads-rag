# rag/vlm/vlm_client.py
# -----------------------------------
# This module manage the connection to vlm server
# -----------------------------------

from typing import Optional
from pathlib import Path

from openai import OpenAI
from google import genai
from google.genai import types

from rag.config import DEFAULT_LLM_MODEL, AVAILABLE_LLMS

class VLMClient:
    """
    A flexible Vision-Language Model client that can work with different backends.
    Currently supports OpenAI or other models via a custom query function.
    """
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        """
        Args:
            model_name: name of the model to use
            backend: "openai", "gemini", or "custom"
            api_key: API key for the backend (if needed)
            custom_query_func: a callable for custom backends, signature: func(image_bytes, prompt) -> str
        """
        self.model_name = model_name
        self.backend: str = AVAILABLE_LLMS[DEFAULT_LLM_MODEL]["backend"]
        self.api_key: Optional[str] = AVAILABLE_LLMS[DEFAULT_LLM_MODEL]["api_key"]

        if self.backend == "openai":
            self.client = OpenAI(api_key=self.api_key)
        elif self.backend == "gemini":
            self.client = genai.Client(api_key=self.api_key)
        elif self.backend == "custom":
            raise NotImplementedError("function not implemented")

    def load_prompt(self, prompt_path: str) -> str:
        """
        Reads a .md file and returns its content as a string.
        """
        path = Path(prompt_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        return path.read_text(encoding="utf-8")

    def generate_img_description(self, image_path: str, prompt: str) -> str:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        if self.backend == "openai":
            return self._openai_query(image_bytes, prompt)
        elif self.backend == "gemini":
            # Replace with actual Gemini API call
            return self._gemini_query(image_bytes, prompt)
        elif self.backend == "custom":
            return self.custom_query_func(image_bytes, prompt)

    def _openai_query(self, image_bytes: bytes, prompt: str) -> str:
        response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        input=image_bytes
                    )
        return response.choices[0].message.content

    def _gemini_query(self, image_bytes: bytes, prompt: str) -> str:
        response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type='image/jpeg',
                        ),
                        prompt
                    ]
                )
        return response.text


# ------------------- Example Usage -------------------
if __name__ == "__main__":
    # test
    vlm_client = VLMClient(model_name=DEFAULT_LLM_MODEL)
    img_path = "data/raw/img/scenario-1.png"
    prompt = vlm_client.load_prompt("rag/prompts/vlm_img_description.md")
    print(vlm_client.generate_img_description(img_path, prompt))
