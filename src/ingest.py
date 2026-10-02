import argparse
import hashlib
import logging
import time
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
LOGGER = logging.getLogger(__name__)


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
    embedding_batch_size: int = 32,
    embedder_factory: Callable[[], BMC_Embedder] = BMC_Embedder,
) -> int:
    if embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be greater than zero")

    started_at = time.perf_counter()
    LOGGER.info("Starting ingestion from %s", raw_dir)
    files = discover_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No supported documents found in {raw_dir}")
    LOGGER.info("Discovered %d supported file(s)", len(files))

    parsers = (
        PDF_Parser(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        DICOM_Parser(),
    )
    parse_started_at = time.perf_counter()
    chunks = parse_files(files, *parsers)
    if not chunks:
        raise ValueError("The discovered documents produced no text chunks")
    LOGGER.info(
        "Parsed %d chunk(s) in %.2f seconds",
        len(chunks),
        time.perf_counter() - parse_started_at,
    )

    embed_started_at = time.perf_counter()
    embedder = embedder_factory()
    embeddings = []
    for start in range(0, len(chunks), embedding_batch_size):
        batch = chunks[start : start + embedding_batch_size]
        LOGGER.info(
            "Embedding batch %d-%d of %d",
            start + 1,
            start + len(batch),
            len(chunks),
        )
        embeddings.extend(embedder.embed_txt([chunk.content for chunk in batch]))
    LOGGER.info(
        "Generated %d embedding(s) in %.2f seconds",
        len(embeddings),
        time.perf_counter() - embed_started_at,
    )

    storage_started_at = time.perf_counter()
    retriever = HybridRetriever(
        chroma_storer=Chroma_Storer(db_path=str(chroma_dir)),
        bm25_storer=BM25_Storer(index_path=str(bm25_index)),
        embedder=embedder,
    )
    retriever.add_chunks(chunks, embeddings)
    LOGGER.info(
        "Stored %d chunk(s) in %.2f seconds; total ingestion time %.2f seconds",
        len(chunks),
        time.perf_counter() - storage_started_at,
        time.perf_counter() - started_at,
    )
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
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.chunk_size <= args.chunk_overlap:
        raise ValueError("chunk-size must be greater than chunk-overlap")
    count = ingest(
        raw_dir=args.raw_dir,
        chroma_dir=args.chroma_dir,
        bm25_index=args.bm25_index,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_batch_size=args.embedding_batch_size,
    )
    print(f"Ingested {count} chunks from {args.raw_dir}")


if __name__ == "__main__":
    main()
