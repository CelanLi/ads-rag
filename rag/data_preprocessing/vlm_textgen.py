from rag.config import DEFAULT_LLM_MODEL
from rag.vlm.vlm_client import VLMClient


class VLMGenerator:
    def __init__(self):
        self.vlm_client = VLMClient(model_name=DEFAULT_LLM_MODEL)

    def generate_img_description(self, img_path: str, ):
        # put the image into the vlm to generate the description, then save the description a text file
        prompt = self.vlm_client.load_prompt("rag/prompts/vlm_img_description.md")
        response = self.vlm_client.generate_img_description(img_path, prompt)
        return response


    def generate_sumo_map_description(self, img_path: str, xml_path: str):
        """
        take the map screenshot image and the map xml file together to generate the description
        """
        prompt = self.vlm_client.load_prompt("rag/prompts/vlm/sumo_map_description.md")
        # TODO
        # response = self.vlm_client.generate_sumo_map_description(img_path, xml_path, prompt)
        # return response
