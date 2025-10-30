# rag/data_preprocessing/process_pdf.py
# -----------------------------------
# This module contains the functions for processing pdf data.
#
# usage:
#     python -m rag.data_preprocessing.process_pdf
# -----------------------------------

import logging
import re
from pathlib import Path
from io import BytesIO

from pypdf import PdfReader as pdf2_read
from rag.data_preprocessing.pipeline import ProcessedData
from utils.file_utils import export_json


class PDFProcessor:
    def __init__(self):
        pass

    def process_pdfs(
        self,
        input_dir: str = "data/raw/pdf/",
        output_dir: str = "data/processed/pdf/",
        type: str = "plain",
    ):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for pdf_file in input_path.glob("*.pdf"):
            try:
                with open(pdf_file, "rb") as f:
                    blob = f.read()

                if type == "plain":
                    text, outlines = PlainParser()(blob)

                output_data = ProcessedData(
                    filename=pdf_file.name,
                    text=text,
                    src_path=[str(pdf_file)],
                    category="pdf",
                )
                export_json(
                    output_dir=output_dir,
                    file_name=pdf_file.name,
                    content=output_data.model_dump(),
                )

                print(f"Processed {pdf_file.name}")

            except Exception as e:
                print(f"Failed to process {pdf_file.name}: {e}")


class PlainParser:
    def __init__(self):
        self.outlines = []

    def clean_text(self, text: str) -> str:
        """Clean PDF text by fixing line breaks and hyphenations."""
        # Remove hyphenation across line breaks
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Replace single newlines (within paragraphs) with spaces
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # Normalize multiple newlines to exactly two (paragraph breaks)
        text = re.sub(r"\n{2,}", "\n\n", text)

        # Remove excessive spaces
        text = re.sub(r" +", " ", text)

        return text.strip()

    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        self.outlines = []
        lines = []

        try:
            # Read the PDF (string path or bytes)
            self.pdf = pdf2_read(
                filename if isinstance(filename, str) else BytesIO(filename)
            )

            # Extract text page by page
            for page in self.pdf.pages[from_page:to_page]:
                page_text = page.extract_text() or ""
                lines.append(page_text.strip())

            # Extract outlines (table of contents)
            outlines = getattr(self.pdf, "outline", None)
            if outlines:

                def dfs(arr, depth):
                    for a in arr:
                        if isinstance(a, dict):
                            title = a.get("/Title", "").strip()
                            if title:
                                self.outlines.append((title, depth))
                        else:
                            dfs(a, depth + 1)

                dfs(outlines, 0)
            else:
                logging.warning("No outlines found")

        except Exception as e:
            logging.exception(f"PlainParser error: {e}")

        # Combine all lines and clean up text
        raw_text = "\n".join(lines)
        cleaned_text = self.clean_text(raw_text)

        return cleaned_text, self.outlines

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError


class DeepDocParser:
    pass


class VisionParser:
    pass


if __name__ == "__main__":
    processor = PDFProcessor()
    processor.process_pdfs(
        input_dir="data/raw/pdf/", output_dir="data/processed/pdf/", type="plain"
    )
