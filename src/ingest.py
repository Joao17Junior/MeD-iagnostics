import argparse
import hashlib
from pathlib import Path
from typing import Callable, Iterable, List

from ingestion.parsers.dicom_parser import DICOM_Parser
from ingestion.parsers.pdf_parser import PDF_Parser
from ingestion.schema import DocumentChunk
from ingestion.storer.bm25_storer import BM25_Storer
from ingestion.storer.chroma_storer import Chroma_Storer
from ingestion.storer.hybrid_retriever import HybridRetriever
from ingestion.vectoring.embedder import BMC_Embedder


SUPPORTED_SUFFIXES = {".dcm", ".dicom", ".pdf"}


def discover_files(raw_dir: Path) -> List[Path]:
    """Return supported files below raw_dir in a stable order."""
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _stable_chunk_id(path: Path, chunk_number: int) -> str:
    value = f"{path.resolve()}:{chunk_number}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def parse_files(
    files: Iterable[Path], pdf_parser: PDF_Parser, dicom_parser: DICOM_Parser
) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    for path in files:
        if path.suffix.lower() == ".pdf":
            parsed_chunks = pdf_parser.parse(str(path))
        else:
            parsed_chunks = [dicom_parser.parse(str(path))]

        for chunk_number, chunk in enumerate(parsed_chunks):
            chunk.id = _stable_chunk_id(path, chunk_number)
            chunks.append(chunk)
    return chunks


def ingest(
    raw_dir: Path,
    chroma_dir: Path,
    bm25_index: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedder_factory: Callable[[], BMC_Embedder] = BMC_Embedder,
) -> int:
    files = discover_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No supported documents found in {raw_dir}")

    parsers = (
        PDF_Parser(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        DICOM_Parser(),
    )
    chunks = parse_files(files, *parsers)
    if not chunks:
        raise ValueError("The discovered documents produced no text chunks")

    embedder = embedder_factory()
    embeddings = embedder.embed_txt([chunk.content for chunk in chunks])

    retriever = HybridRetriever(
        chroma_storer=Chroma_Storer(db_path=str(chroma_dir)),
        bm25_storer=BM25_Storer(index_path=str(bm25_index)),
        embedder=embedder,
    )
    retriever.add_chunks(chunks, embeddings)
    return len(chunks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--chroma-dir", type=Path, default=Path("data/chromadb"))
    parser.add_argument(
        "--bm25-index", type=Path, default=Path("data/bm25/bm25_index.pkl")
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.chunk_size <= args.chunk_overlap:
        raise ValueError("chunk-size must be greater than chunk-overlap")
    count = ingest(
        raw_dir=args.raw_dir,
        chroma_dir=args.chroma_dir,
        bm25_index=args.bm25_index,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Ingested {count} chunks from {args.raw_dir}")


if __name__ == "__main__":
    main()
