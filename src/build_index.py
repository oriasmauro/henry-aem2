"""
Build an index of document chunks and their embeddings for RAG FAQ support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from openai import OpenAI

from utils import (
    Chunk,
    chunk_text_by_tokens,
    configure_logging,
    count_tokens,
    getenv_required,
    load_text,
    save_json,
)

DOC_PATH = Path("data/faq_document.txt")
OUT_PATH = Path("storage/index.json")

logger = logging.getLogger("rag-index")
MIN_CHUNK_TOKENS = 50
MAX_CHUNK_TOKENS = 500


# ---------------------------------------------------------
# Stage 1: Load + Chunk
# ---------------------------------------------------------
def load_and_chunk_document(
    doc_path: Path,
    embedding_model: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    logger.info("Stage 1: Loading and chunking document")

    raw = load_text(doc_path)

    total_tokens = count_tokens(raw, embedding_model)
    logger.info("Document token count: %d", total_tokens)

    chunks = chunk_text_by_tokens(
        raw,
        chunk_size_tokens=chunk_size_tokens,
        overlap_tokens=overlap_tokens,
        tokenizer_model=embedding_model,
    )

    logger.info("Chunks created: %d", len(chunks))

    if len(chunks) < 20:
        raise RuntimeError(
            f"Expected >= 20 chunks, got {len(chunks)}. "
            "Increase document size or reduce chunk_size."
        )

    token_counts = [c.tokens for c in chunks]
    logger.info(
        "Chunk token stats | min=%d | avg=%.2f | max=%d",
        min(token_counts),
        mean(token_counts),
        max(token_counts),
    )

    last_chunk_id = chunks[-1].id
    invalid = [
        c
        for c in chunks
        if c.tokens > MAX_CHUNK_TOKENS or (c.tokens < MIN_CHUNK_TOKENS and c.id != last_chunk_id)
    ]
    if invalid:
        examples = ", ".join(f"id={c.id},tokens={c.tokens}" for c in invalid[:5])
        raise RuntimeError(
            f"Chunk token constraint failed (must be {MIN_CHUNK_TOKENS}-{MAX_CHUNK_TOKENS} tokens; "
            "last chunk may be smaller). "
            f"Examples: {examples}"
        )

    if chunks[-1].tokens < MIN_CHUNK_TOKENS:
        logger.warning(
            "Last chunk is smaller than recommended minimum | id=%d | tokens=%d",
            chunks[-1].id,
            chunks[-1].tokens,
        )

    logger.info("Stage 1 complete")
    return chunks


# ---------------------------------------------------------
# Stage 2: Generate Embeddings
# ---------------------------------------------------------
def generate_embeddings(
    client: OpenAI,
    embedding_model: str,
    chunks: list[Chunk],
) -> list[list[float]]:
    logger.info("Stage 2: Generating embeddings")

    texts = [c.text for c in chunks]

    response = client.embeddings.create(
        model=embedding_model,
        input=texts,
    )

    embeddings = [item.embedding for item in response.data]

    if len(embeddings) != len(chunks):
        raise RuntimeError("Embeddings count does not match chunks count")

    logger.info("Embeddings generated: %d", len(embeddings))
    logger.info("Embedding dimension: %d", len(embeddings[0]))

    logger.info("Stage 2 complete")
    return embeddings


# ---------------------------------------------------------
# Stage 3: Build Index Structure
# ---------------------------------------------------------
def build_index_payload(
    embedding_model: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> dict:
    logger.info("Stage 3: Building index payload")

    payload_chunks = []
    for chunk, emb in zip(chunks, embeddings, strict=True):
        payload_chunks.append(
            {
                "id": chunk.id,
                "text": chunk.text,
                "tokens": chunk.tokens,
                "embedding": emb,
            }
        )

    index_payload = {
        "embedding_model": embedding_model,
        "chunk_size_tokens": chunk_size_tokens,
        "overlap_tokens": overlap_tokens,
        "chunks": payload_chunks,
    }

    logger.info("Index payload ready")
    logger.info("Stage 3 complete")

    return index_payload


# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
def main() -> None:
    configure_logging("INFO")
    load_dotenv()

    logger.info("Starting Index Pipeline")

    # Environment validation
    _ = getenv_required("OPENAI_API_KEY")
    embedding_model = getenv_required("EMBEDDING_MODEL")

    chunk_size_tokens = 120
    overlap_tokens = 20

    logger.info(
        "Configuration | model=%s | chunk_size=%d | overlap=%d",
        embedding_model,
        chunk_size_tokens,
        overlap_tokens,
    )

    client = OpenAI()

    # Stage 1
    chunks = load_and_chunk_document(
        DOC_PATH,
        embedding_model=embedding_model,
        chunk_size_tokens=chunk_size_tokens,
        overlap_tokens=overlap_tokens,
    )

    # Stage 2
    embeddings = generate_embeddings(
        client,
        embedding_model,
        chunks,
    )

    # Stage 3
    index_payload = build_index_payload(
        embedding_model,
        chunk_size_tokens,
        overlap_tokens,
        chunks,
        embeddings,
    )

    # Stage 4: Save
    logger.info("Stage 4: Saving index to %s", OUT_PATH)
    save_json(OUT_PATH, index_payload)

    logger.info("Index Pipeline finished successfully")
    logger.info("Total chunks stored: %d", len(chunks))


if __name__ == "__main__":
    main()
