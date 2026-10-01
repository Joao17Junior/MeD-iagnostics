import os
from typing import Any, Optional, List, Dict
from PIL import Image
from pathlib import Path

from ingestion.schema import DocumentChunk
from ingestion.parsers.pdf_parser import PDF_Parser
from ingestion.parsers.dicom_parser import DICOM_Parser
from ingestion.vectoring.embedder import BMC_Embedder
from ingestion.storer.hybrid_retriever import HybridRetriever
from ingestion.storer.chroma_storer import Chroma_Storer
from ingestion.storer.bm25_storer import BM25_Storer
from src.vlmodel import Med_VLM
from src.ingest import ingest



# brings it all together in a solo class for the Front End to call:
# 1. input the query and get the context from the hybrid search
# 2. (Optional) input files - pdf or dicom - and call all the ingestion pipeline for that matter 
# 3. call the model with the prompt and context
# 4. get the results and send it to the front end
class RAG_Pipeline:
    def __init__(self):
        self.pdf_parser = PDF_Parser()
        self.dicom_parser = DICOM_Parser()

        self.embedder = BMC_Embedder()
        self.retriever = HybridRetriever(Chroma_Storer(), BM25_Storer(), self.embedder)
        self.vlm = Med_VLM()

    def query(
            self,
            query_text: str,
            n_results: int = 5,
            max_chroma_distance: int = 0,
            min_bm25_score: int = 0,
            input_folder: Optional[str] = "./data/input_data"
            ) -> str:
        
        query_prefix = ""
        image = Dict[id, Image]

        # parse the files in the input folder
        if os.path.exists(input_folder):
            for root, dirs, files in os.walk(input_folder):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        temp_chunk = self.pdf_parser.parse(root + file)
                        for chunk in temp_chunk:
                            query_prefix = query_prefix + f"[pdf context added: {chunk.content}]\n"
                    elif file.lower().endswith(".dicom") or file.lower().endswith(".dcm"):
                        temp_chunk = self.dicom_parser.parse(root + file)
                        for chunk in temp_chunk:
                            image[chunk.id] = image
                            query_prefix = query_prefix + f"[dicom context added: id = {chunk.id}, {chunk.content}"

        search_query = f"{query_prefix}{query_text}".strip()

        # call hybrid retriever for context
        retrieved_res = self.retriever.search(
                query_text = search_query, 
                n_results = n_results,
                max_chroma_distace = max_chroma_distance,
                min_bm25_score = min_bm25_score
                )

        # call vlm
        vlm_response = self.vlm.generate_response(
                query = query_text,
                retrieved_chunks = retrieved_res,
                image = None
                )

        # return
        return vlm_response

    def ingest(
            self,
            raw_dir: str = "./data/raw",
            ) -> int:
        # calls ingest method from ingest
        return ingest(
                raw_dir = Path(raw_dir),
                chroma_dir = Path("./data/chromadb"),
                bm25_index = Path("./data/bm25/bm25_index.plk"),
                embedder_factory = lambda: self.embedder
                )



