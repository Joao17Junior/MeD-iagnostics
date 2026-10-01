from pypdf import PdfReader
from typing import List
import uuid
from ingestion.schema import DocumentChunk

class PDF_Parser:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, file_path: str, doc_type: str = "literature") -> List[DocumentChunk]:
        reader = PdfReader(file_path)
        chunks: List[DocumentChunk] = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            # empty page
            if not text.strip():
                continue
    
            # start parsing - sliding window
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                
                # create chunk (based on schema)
                chunk = DocumentChunk(
                        id=str(uuid.uuid4()),
                        content=chunk_text,
                        metadata={
                            "source": file_path,
                            "page"  : page_num,
                            "type"  : doc_type
                            }
                        )
                chunks.append(chunk)

                # slide
                start += self.chunk_size - self.chunk_overlap


        return chunks

