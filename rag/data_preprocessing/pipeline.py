# rag/data_preprocessing/pipeline.py
# -----------------------------------
# This module contains the pipeline for preprocessing data.
# It will automatically convert data saved in data/raw/ into text and save it in data/processed/
#
# Usage:
#     python -m rag.data_preprocessing.pipeline
# -----------------------------------

from typing import List

from pydantic import BaseModel


class ProcessedData(BaseModel):
    filename: str
    src_path: List[str]
    text: str
    category: str

    def model_dump(self) -> dict:
        return {
            "filename": self.filename,
            "src_path": self.src_path,
            "text": self.text,
            "category": self.category,
        }


# def process_all_raw_data(
#     input_dir: str = "data/raw/", output_dir: str = "data/processed/"
# ):
#     pdf_processor = PDFProcessor()
#     pdf_processor.process_pdfs()

#     img_processor = ImgProcessor()
#     img_processor.process_imgs()


# if __name__ == "__main__":
#     process_all_raw_data()
