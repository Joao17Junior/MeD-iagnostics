import os
import pickle
import re
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi
from ingestion.schema import DocumentChunk


class BM25_Storer:
    def __init__(self, index_path: str = "./data/bm25/bm25_index.pkl"):
        self.index_path = index_path
        self.chunks: List[DocumentChunk] = []
        self.tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._load_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _save_index(self) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump((self.chunks, self.tokens), f)

    def _load_index(self) -> None:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    self.chunks, self.tokens = pickle.load(f)
                    if self.tokens:
                        self.bm25 = BM25Okapi(self.tokens)
            except Exception:
                pass

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        replacements = {chunk.id: chunk for chunk in chunks}
        retained_chunks = [
            chunk for chunk in self.chunks if chunk.id not in replacements
        ]
        self.chunks = retained_chunks + list(replacements.values())
        self.tokens = [self._tokenize(chunk.content) for chunk in self.chunks]

        self.bm25 = BM25Okapi(self.tokens)
        self._save_index()

    def query_similar(
            self,
            query_text: str,
            n_results: int = 5
            ) -> List[Dict[str, Any]]:
                    
    
        if not self.bm25 or not self.chunks:
            return []
        
        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n_results]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] > 0:  # Retornar apenas correspondências com score > 0
                results.append(
                    {
                        "chunk": self.chunks[idx],
                        "score": float(scores[idx]),
                        "rank": rank,
                    }
                )

        return results
