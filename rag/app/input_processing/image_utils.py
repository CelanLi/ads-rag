# convert user imput image to text description


from rag.llm.vision_models import BaseVisionModel


class ImageUtils:
    def __init__(self, vlm: BaseVisionModel):
        self.vlm = vlm

    def convert_input_img_to_description(self, img_path: str, requirement: str) -> str:
        return self.vlm.generate_input_img_description(
            img_path=img_path,
            requirement=requirement,
        )
