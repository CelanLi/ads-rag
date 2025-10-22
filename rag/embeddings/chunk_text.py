# rag/embeddings/chunk_text.py
# -----------------------------------
# This module chunks text data into smaller pieces
#
# usage:
#     python -m rag.embeddings.chunk_text
# -----------------------------------

import json
from pathlib import Path
from typing import List
from pydantic import BaseModel
from rag.config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from utils.file_utils import export_json

class ChunkMetadata(BaseModel):
    filename: str
    chunks: List[str]

class TextChunker:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

    def chunk_all_text(self, input_dir: str = "data/processed", output_dir: str = "data/chunks"):
        "chunk all sub directories in the input_dir and save the chunks in output dir"
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for sub_dir in input_path.glob("**/*.json"):
            with open(sub_dir, "r", encoding="utf-8") as f:
                text_dict = json.load(f)
                text = text_dict.get("text", "")
                filename = sub_dir.stem
                chunks = self.chunk_text(text)
                chunk_metadata = ChunkMetadata(filename=filename, chunks=chunks)
                export_json(output_dir=output_dir, file_name=filename, content=chunk_metadata.model_dump())

if __name__ == "__main__":
    # text_str = "This is a test text to chunk. It is a long text that needs to be chunked into smaller pieces. This is a test text to chunk. It is a long text that needs to be chunked into smaller pieces. This is a test text to chunk. It is a long text that needs to be chunked into smaller pieces. This is a test text to chunk. It is a long text that needs to be chunked into smaller pieces. This is a test text to chunk. It is a long text that needs to be chunked into smaller pieces. This is a test text to chunk. It is a long text that needs to be chunked into smaller pieces. This is a test text to chunk. It is a long text that needs to be chunked into smaller pieces. This is a test text to chunk. It is a long text that needs to be chunked into smaller pieces."
    # text_dict = {
    #     "text": text_str,
    #     "outlines": [
    #         {"title": "Introduction", "depth": 0},
    #         {"title": "Chapter 1", "depth": 1},
    #         {"title": "Section 1.1", "depth": 2},
    #         {"title": "Subsection 1.1.1", "depth": 3},
    #         {"title": "Subsection 1.1.2", "depth": 3},
    #         {"title": "Chapter 2", "depth": 1},
    #         {"title": "Section 2.1", "depth": 2},
    #         {"title": "Subsection 2.1.1", "depth": 3},
    #         {"title": "Subsection 2.1.2", "depth": 3},
    #     ]
    # }
    # text_file_path = "data/processed/pdf/BIM Model Cheking Method.pdf.json"
    # with open(text_file_path, "r", encoding="utf-8") as f:
    #     text_dict = json.load(f)
    # chunker = TextChunker(text_dict)
    # chunks = chunker()
    # export_json(output_dir="data/chunks/pdf", file_name="BIM Model Cheking Method.pdf.chunks", content={"chunks": chunks})
    chunker = TextChunker()
    chunker.chunk_all_text()