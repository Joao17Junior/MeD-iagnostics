from typing import Any, Dict, List, Optional
from ingestion.schema import DocumentChunk
from ingestion.storer.bm25_storer import BM25_Storer
from ingestion.storer.chroma_storer import Chroma_Storer
from ingestion.vectoring.embedder import BMC_Embedder


class HybridRetriever:
    def __init__(
        self,
        chroma_storer: Chroma_Storer,
        bm25_storer: BM25_Storer,
        embedder: BMC_Embedder,
    ):
        self.chroma_storer = chroma_storer
        self.bm25_storer = bm25_storer
        self.embedder = embedder

    def add_chunks(
        self, chunks: List[DocumentChunk], embeddings: List[List[float]]
    ) -> None:
        self.chroma_storer.add_chunks(chunks, embeddings)
        self.bm25_storer.add_chunks(chunks)

    def search(
        self,
        query_text: str,
        n_results: int = 5,
        k: int = 60,
        query_embedding: Optional[List[float]] = None,
        max_chroma_distance: Optional[float] = None,
        min_bm25_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        query_vector = query_embedding or self.embedder.embed_txt(query_text)[0]

        chroma_res = self.chroma_storer.query_similar(
            query_embedding=query_vector, n_results=n_results * 2
        )

        bm25_res = self.bm25_storer.query_similar(
            query_text=query_text, n_results=n_results * 2
        )

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        if chroma_res and "ids" in chroma_res and chroma_res["ids"]:
            ids = chroma_res["ids"][0]
            documents = chroma_res["documents"][0]
            metadatas = chroma_res["metadatas"][0]
            distances = chroma_res.get("distances", [[]])[0]

            for rank, (doc_id, doc_text, meta, distance) in enumerate(
                zip(ids, documents, metadatas, distances), start=1
            ):
                if (
                    max_chroma_distance is not None
                    and distance > max_chroma_distance
                ):
                    continue
                if doc_id not in chunk_map:
                    chunk_map[doc_id] = DocumentChunk(
                        id=doc_id, content=doc_text, metadata=meta
                    )
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        if bm25_res:
            for item in bm25_res:
                chunk: DocumentChunk = item["chunk"]
                rank: int = item["rank"]
                if item["score"] < min_bm25_score:
                    continue
                doc_id = chunk.id

                if doc_id not in chunk_map:
                    chunk_map[doc_id] = chunk

                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        sorted_doc_ids = sorted(
            rrf_scores.keys(), key=lambda d_id: rrf_scores[d_id], reverse=True
        )[:n_results]

        final_results = [
            {
                "chunk": chunk_map[doc_id],
                "rrf_score": rrf_scores[doc_id],
            }
            for doc_id in sorted_doc_ids
        ]

        return final_results
