from abc import ABC
from pathlib import Path

from google import genai
from langchain.prompts import PromptTemplate

from rag.config import AVAILABLE_VISION_MODELS


IMG_DESCRIPTION_PROMPT_PATH = "rag/prompts/preproc/vlm_img_description.md"
SUMO_MAP_DESCRIPTION_PROMPT_PATH = "rag/prompts/preproc/sumo_map_description.md"
INPUT_IMG_DESCRIPTION_PROMPT_PATH = "rag/prompts/input/vlm_img_description.md"


class BaseVisionModel(ABC):
    def __init__(self, **kwargs):
        pass

    def load_prompt(self, prompt_path: str) -> str:
        path = Path(prompt_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return path.read_text(encoding="utf-8")

    def describe_img(
        self, img_path: str, input_variables: dict, prompt_template_path: str
    ) -> str:
        pass

    def generate_img_description(
        self, img_path: str, prompt_path: str = IMG_DESCRIPTION_PROMPT_PATH
    ) -> str:
        pass

    def generate_sumo_map_description(self, img_path: str, xml_path: str) -> str:
        pass

    def generate_input_img_description(
        self,
        img_path: str,
        requirement: str,
        prompt_path: str = INPUT_IMG_DESCRIPTION_PROMPT_PATH,
    ) -> str:
        pass


class OpenAI(BaseVisionModel):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def describe_img(
        self, img_path: str, input_variables: dict, prompt_template_path: str
    ) -> str:
        pass

    def generate_img_description(
        self, img_path: str, prompt_path: str = IMG_DESCRIPTION_PROMPT_PATH
    ) -> str:
        prompt_template = self.load_prompt(prompt_path)
        prompt = PromptTemplate(input_variables=[""], template=prompt_template).format()

        with open(img_path, "rb") as f:
            image_bytes = f.read()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            input=image_bytes,
        )
        return response.choices[0].message.content.strip()

    def generate_sumo_map_description(
        self,
        img_path: str,
        xml_path: str,
        prompt_path: str = SUMO_MAP_DESCRIPTION_PROMPT_PATH,
    ) -> str:
        pass

    def generate_input_img_description(
        self, img_path: str, prompt_path: str = INPUT_IMG_DESCRIPTION_PROMPT_PATH
    ) -> str:
        pass


class Gemini(BaseVisionModel):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def describe_img(
        self, img_path: str, input_variables: dict, prompt_template_path: str
    ) -> str:
        """
        Describe the image with the given input variables and prompt template.
        Args:
            img_path: The path to the image.
            input_variables: The input variables for the prompt template, in the form of a dictionary, e.g. {"requirement": "xxx", "xml": "xxx"}
            prompt_template_path: The path to the prompt template.
        Returns:
            The description of the image.
        """
        prompt_template = self.load_prompt(prompt_template_path)
        prompt = PromptTemplate(
            input_variables=list(input_variables.keys()), template=prompt_template
        ).format(**input_variables)

        with open(img_path, "rb") as f:
            image_bytes = f.read()

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                genai.types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                prompt,
            ],
        )
        return response.text

    def generate_img_description(
        self, img_path: str, prompt_path: str = IMG_DESCRIPTION_PROMPT_PATH
    ) -> str:
        return self.describe_img(img_path, {}, prompt_path=prompt_path)

    def generate_sumo_map_description(
        self,
        img_path: str,
        xml_path: str,
        prompt_path: str = SUMO_MAP_DESCRIPTION_PROMPT_PATH,
    ) -> str:
        with open(xml_path, "r", encoding="utf-8") as f:
            xml_content = f.read()

        return self.describe_img(
            img_path=img_path,
            input_variables={"xml": xml_content},
            prompt_template_path=prompt_path,
        )

    def generate_input_img_description(
        self,
        img_path: str,
        requirement: str,
        prompt_path: str = INPUT_IMG_DESCRIPTION_PROMPT_PATH,
    ) -> str:
        return self.describe_img(
            img_path=img_path,
            input_variables={"requirement": requirement},
            prompt_template_path=prompt_path,
        )


def get_vision_model(model_name: str) -> BaseVisionModel:
    model_name = model_name.lower()
    backend = AVAILABLE_VISION_MODELS[model_name]["backend"]
    if backend == "openai":
        return OpenAI(
            model_name=model_name,
            api_key=AVAILABLE_VISION_MODELS[model_name]["api_key"],
        )
    elif backend == "gemini":
        return Gemini(
            model_name=model_name,
            api_key=AVAILABLE_VISION_MODELS[model_name]["api_key"],
        )
    else:
        raise ValueError(f"Unknown vision model: {model_name}")
