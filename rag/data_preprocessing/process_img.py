# # rag/data_preprocessing/process_img.py
# -----------------------------------
# This module contains the functions for processing image data.
#
# usage:
#     python -m rag.data_preprocessing.process_img
# -----------------------------------

from pathlib import Path

from rag.vlm.vlm_client import VLMClient
from rag.config import DEFAULT_LLM_MODEL
from utils.file_utils import export_txt

class ImgProcessor:
    def __init__(self):
        pass

    def process_imgs(
        self,
        input_dir: str = "data/raw/img/",
        output_dir: str = "data/processed/img/",
        type: str = "vlm_description"
    ):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # List of common image extensions
        img_extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff", "*.webp"]

        # Collect all image files
        img_files = []
        for ext in img_extensions:
            img_files.extend(input_path.glob(ext))

        if not img_files:
            print("No image files found in", input_dir)
            return

        for img_file in img_files:
            try:
                if type == "vlm_description":
                    vlm_parser = VLMParser()
                    description = vlm_parser.generate_img_description(img_path=img_file)

                # Use stem to avoid double extensions (e.g., image.png.txt)
                export_txt(output_dir=output_dir, file_name=img_file.name, content={"filename": img_file.name, "text": description})

                print(f"Processed {img_file.name}")

            except Exception as e:
                print(f"Failed to process {img_file.name}: {e}")

class VLMParser:
    def __init__(self):
        pass

    def generate_img_description(self, img_path: str, ):
        # put the image into the vlm to generate the description, then save the description a text file
        vlm_client = VLMClient(model_name=DEFAULT_LLM_MODEL)
        prompt = vlm_client.load_prompt("rag/prompts/vlm_img_description.md")
        response = vlm_client.generate_img_description(img_path, prompt)
        return response


    def get_img_representation(self, img_path: str):
        # use some cv methods to get the representations
        pass

if __name__ == "__main__":
    img_processor = ImgProcessor()
    img_processor.process_imgs()