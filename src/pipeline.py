import os
from typing import Any, Optional, List, Dict
from PIL import Image

from ingestion.schema import DocumentChunk
from ingestion.parsers.pdf_parser import PDF_Parser
from ingestion.parsers.dicom_parser import DICOM_Parser
from ingestion.vectoring.embedder import BMC_Embedder
from ingestion.storer.hybrid_retriever import HybridRetriever
from ingestion.storer.chroma_storer import Chroma_Storer
from ingestion.storer.bm25_storer import BM25_Storer
from src.vlmodel import Med_VLM



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

    def query(
            self,
            query_text: str,
            n_results: int = None,
            k: int = None,
            query_embedding: Optional[List[float]] = None,
            max_chroma_distance: Optional[float] = None,
            min_bm25_score: float = None,
            input_folder: Optional[str] = "./data/input_data/",
            ) -> List[Dict[str, Any]]:
        
        query_prefix = ""
        image = Dict[id, Image]

        # parse the files in the input folder
        if os.path.exists(input_folder):
            for root, dirs, files in os.walk(input_folder):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        temp_chunk = self.pdf_parser.parse(input_folder + file)
                        query_prefix = query_prefix + f"[pdf context added: {temp_chunk.content}]\n"
                    elif file.lower().endswith(".dicom") or file.lower().endswith(".dcm"):
                        temp_chunk = self.dicom_parser.parse(input_folder + file)
                        image[temp_chunk.id] = image
                        query_prefix = query_prefix + f"[dicom context added: id = {temp_chunk.id}, {temp_chunk.content}"

        search_query = f"{query_prefix}{query_text}"

        # call hybrid retriever for context
        
