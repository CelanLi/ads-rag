# rag/data_preprocessing/pipeline.py
# -----------------------------------
# This module contains the pipeline for preprocessing data.
# It will automatically convert data saved in data/raw/ into text and save it in data/processed/
#
# Usage:
#     python -m src.data_preprocessing.pipeline
# -----------------------------------

from rag.data_preprocessing.process_pdf import PDFProcessor
from rag.data_preprocessing.process_img import ImgProcessor

def process_all_raw_data(input_dir: str = "data/raw/", output_dir: str = "data/processed/"):
    pdf_processor = PDFProcessor()
    pdf_processor.process_pdfs(
        input_dir="data/raw/pdf/",
        output_dir="data/processed/pdf/",
        type="plain"
    )

    img_processor = ImgProcessor()
    img_processor.process_imgs()
