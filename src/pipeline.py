import os
from typing import Any, Optional, List, Dict
from ingestion.schema import DocumentChunk
from ingestion.parsers.pdf_parser import PDF_Parser
from ingestion.parsers.dicom_parser import DICOM_Parser
from ingestion.vectoring.embedder import BMC_Embedder
from ingestion.storer.hybrid_retriever import HybridRetriever
from src.vlmodel import Med_VLM



# brings it all together in a solo class for the Front End to call:
# 1. input the query and get the context from the hybrid search
# 2. (Optional) input files - pdf or dicom - and call all the ingestion pipeline for that matter 
# 3. call the model with the prompt and context
# 4. get the results and send it to the front end
class RAG_Pipeline:
    
