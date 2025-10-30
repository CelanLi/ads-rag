# # rag/data_preprocessing/process_img.py
# -----------------------------------
# This module contains the functions for processing image data.
#
# usage:
#     python -m rag.data_preprocessing.process_img
# -----------------------------------

from pathlib import Path

from rag.data_preprocessing.vlm_textgen import VLMGenerator
from utils.file_utils import export_json

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
            self.process_img(img_file=img_file, type=type, output_dir=output_dir)

    def process_img(self, img_file: str, type: str = "vlm_description", output_dir: str = "data/processed/img/"):
        try:
            if type == "vlm_description":
                vlm_generator = VLMGenerator()
                description = vlm_generator.generate_img_description(img_path=img_file)
            export_json(output_dir=output_dir, file_name=img_file.name, content={"filename": img_file.name, "text": description})
            print(f"Processed {img_file.name}")
        except Exception as e:
            print(f"Failed to process {img_file.name}: {e}")


if __name__ == "__main__":
    img_processor = ImgProcessor()
    img_processor.process_imgs()