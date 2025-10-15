# rag/data_preprocessing/process_pdf.py
# -----------------------------------
# This module contains the functions for processing pdf data.
# -----------------------------------

import logging
from pathlib import Path
from io import BytesIO

from pypdf import PdfReader as pdf2_read
from utils.file_utils import export_json

class PDFProcessor:
    def __init__(self):
        pass

    def process_pdfs(self, input_dir: str = "data/raw/pdf/", 
                    output_dir: str = "data/processed/pdf/", 
                    type: str = "plain"):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for pdf_file in input_path.glob("*.pdf"):
            try:
                with open(pdf_file, "rb") as f:
                    blob = f.read()

                if type == "plain":
                    lines, outlines = PlainParser()(blob)

                output_data = {
                    "filename": pdf_file.name,
                    "lines": [line for line, _ in lines],
                    "outlines": outlines
                }

                export_json(output_dir=output_dir, file_name=pdf_file.name, content=output_data)

                print(f"Processed {pdf_file.name}")

            except Exception as e:
                print(f"Failed to process {pdf_file.name}: {e}")
        
class PlainParser:
    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        self.outlines = []
        lines = []
        try:
            self.pdf = pdf2_read(filename if isinstance(filename, str) else BytesIO(filename))
            for page in self.pdf.pages[from_page:to_page]:
                lines.extend([t for t in page.extract_text().split("\n")])

            outlines = self.pdf.outline

            def dfs(arr, depth):
                for a in arr:
                    if isinstance(a, dict):
                        self.outlines.append((a["/Title"], depth))
                        continue
                    dfs(a, depth + 1)

            dfs(outlines, 0)
        except Exception:
            logging.exception("Outlines exception")
        if not self.outlines:
            logging.warning("Miss outlines")

        return [(line, "") for line in lines], self.outlines

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError

class DeepDocParser:
    pass

class VisionParser:
    pass

if __name__ == '__main__':
    processor = PDFProcessor()
    processor.process_pdfs(
        input_dir="data/raw/pdf/",
        output_dir="data/processed/pdf/",
        type="plain"
    )