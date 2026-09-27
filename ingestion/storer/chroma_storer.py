import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from ingestion.schema import DocumentChunk

class Chroma_Storer:
    def __init__(
            self,
            db_path: str = "./data/chromadb",
            collection_name: str = "med-iagnostics"
            ):

        self.db_path = db_path
        os.makedirs(db_path, exist_ok = True)

        self.client = chromadb.PersistentClient(
                path = db_path,
                settings = Settings(anonymized_telemetry=False)
                )

        self.collection = self.client.get_or_create_collection(
                name = collection_name,
                metadata = {"hnsw:space": "cosine"}
                )

    def add_chunks(
            self,
            chunks: List[DocumentChunk],
            embeddings: List[List[float]]
            ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)} and embeddings ({len(embeddings)} must be the same!")

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.content)

            clean_metadata: Dict[str, Any] = {}
            for k, v in chunk.metadata.items():
                if isinstance(v, (int, float, str, bool)):
                    clean_metadata[k] = v
                else:
                    clean_metadata[k] = str(v)

            clean_metadata["has_img"] = chunk.image is not None
            metadatas.append(clean_metadata)

        self.collection.upsert(
                ids = ids,
                embeddings = embeddings,
                documents = documents,
                metadatas = metadatas
                )


    def query_similar(
            self,
            query_embedding: List[float],
            n_results: int = 5,
            where_filter: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:

        results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results = n_results,
                where = where_filter
                )
        return results


    def count(self) -> int:
        return self.collection.count()
