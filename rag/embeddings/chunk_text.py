# rag/embeddings/chunk_text.py
# -----------------------------------
# This module chunks text data into smaller pieces
#
# usage:
#     python -m rag.embeddings.chunk_text
# -----------------------------------

import json
from rag.config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP

class TextChunker:
    def __init__(self, text: str | dict, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        if isinstance(text, str):
            self.text = text
        elif isinstance(text, dict):
            self.text = text.get("text", "")
        else:
            raise ValueError(f"Invalid text type: {type(text)}. Expected str or dict.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def __call__(self):
        return self.chunk_text()

    def chunk_text(self):
        chunks = []
        for i in range(0, len(self.text), self.chunk_size - self.overlap):
            chunks.append(self.text[i:i + self.chunk_size])
        return chunks

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
    text_file_path = "data/processed/pdf/BIM Model Cheking Method.pdf.json"
    with open(text_file_path, "r", encoding="utf-8") as f:
        text_dict = json.load(f)
    chunker = TextChunker(text_dict)
    print(chunker())