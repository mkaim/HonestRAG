"""Ingest documents into the RAG corpus.

    uv run python ingest.py FILE [FILE ...]      # each file is one document
    uv run python ingest.py --dir docs/          # every *.txt / *.md under docs/
    cat notes.txt | uv run python ingest.py -    # stdin as a single document

Document id defaults to the file path (or "stdin"); pass --id to override for a
single input. Each document is split into chunks before being added; the chunk
retrieved and cited by the LLM is a slice of the file, not the whole file.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from chunkers import BlockChunker
from config import Settings
from db import PsqlRagDb
from embedders import SenTranEmbedder
from loaders import TextFileLoader, text_to_blocks
from models import Block, Chunk, Document

TEXT_SUFFIXES = {".txt", ".md"}


def _collect_documents(
    paths: list[str],
    directory: str | None,
    doc_id: str | None,
    max_block_size: int,
) -> list[tuple[Document, list[Block]]]:
    loader = TextFileLoader(max_size=max_block_size)
    docs: list[tuple[Document, list[Block]]] = []

    if directory:
        paths = paths + [
            str(p)
            for p in sorted(Path(directory).rglob("*"))
            if p.suffix.lower() in TEXT_SUFFIXES
        ]

    for raw in paths:
        if raw == "-":
            docs.append(
                (
                    Document(id=doc_id or "stdin"),
                    text_to_blocks(sys.stdin.read(), max_block_size),
                )
            )
        else:
            blocks, metadata = loader.load(raw)
            docs.append((Document(id=doc_id or raw, metadata=metadata), blocks))

    return docs


async def _run(documents: list[tuple[Document, list[Block]]]) -> None:
    cfg = Settings()
    chunker = BlockChunker()

    async with PsqlRagDb(cfg.dsn, SenTranEmbedder(cfg.embed_model)) as db:
        total_chunks = 0
        for document, blocks in documents:
            chunks: list[Chunk] = list(chunker.chunk(document.id, blocks))
            await db.add(document, chunks)
            total_chunks += len(chunks)

    print(f"Ingested {len(documents)} document(s) as {total_chunks} chunk(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="files to ingest, or - for stdin")
    parser.add_argument("--dir", help="ingest every .txt/.md file under this directory")
    parser.add_argument("--id", help="document id (only valid with a single input)")
    parser.add_argument(
        "--max-block-size",
        type=int,
        default=1000,
        help="hard-split loaded text so no paragraph exceeds this many characters",
    )
    args = parser.parse_args()

    if not args.files and not args.dir:
        parser.error("provide files, - for stdin, or --dir")
    if args.id and (args.dir or len(args.files) != 1):
        parser.error("--id requires exactly one file input")

    documents = _collect_documents(args.files, args.dir, args.id, args.max_block_size)
    if not documents:
        parser.error("nothing to ingest")

    asyncio.run(_run(documents))


if __name__ == "__main__":
    main()
