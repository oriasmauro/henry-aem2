"""
Build an index of document chunks and their embeddings for RAG FAQ support.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from openai import OpenAI

from embedding_provider import (
    OPENAI_PROVIDER,
    SENTENCE_TRANSFORMERS_PROVIDER,
    build_embedding_provider,
)
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

logger = logging.getLogger("rag-index")
MIN_CHUNK_TOKENS = 50
MAX_CHUNK_TOKENS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build embeddings index for the FAQ document")
    parser.add_argument(
        "--provider",
        choices=[OPENAI_PROVIDER, SENTENCE_TRANSFORMERS_PROVIDER],
        default=OPENAI_PROVIDER,
        help="Embedding backend to use",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=120,
        help="Chunk size in tokens",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=20,
        help="Overlap between consecutive chunks in tokens",
    )
    parser.add_argument(
        "--out",
        help="Output path for the generated index JSON",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def load_and_chunk_document(
    doc_path: Path,
    tokenizer_model_hint: str | None,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    logger.info("Stage 1: Loading and chunking document")

    raw = load_text(doc_path)
    total_tokens = count_tokens(raw, tokenizer_model_hint)
    logger.info("Document token count: %d", total_tokens)

    chunks = chunk_text_by_tokens(
        raw,
        chunk_size_tokens=chunk_size_tokens,
        overlap_tokens=overlap_tokens,
        tokenizer_model=tokenizer_model_hint,
    )

    logger.info("Chunks created: %d", len(chunks))

    if len(chunks) < 20:
        raise RuntimeError(
            f"Expected >= 20 chunks, got {len(chunks)}. "
            "Increase document size or reduce --chunk-size."
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
            f"Examples: {examples}. Adjust --chunk-size/--overlap."
        )

    if chunks[-1].tokens < MIN_CHUNK_TOKENS:
        logger.warning(
            "Last chunk is smaller than recommended minimum | id=%d | tokens=%d",
            chunks[-1].id,
            chunks[-1].tokens,
        )

    logger.info("Stage 1 complete")
    return chunks


def generate_embeddings(provider, chunks: list[Chunk]) -> list[list[float]]:
    logger.info("Stage 2: Generating embeddings | provider=%s", provider.name)

    texts = [chunk.text for chunk in chunks]
    embeddings = provider.embed_texts(texts)

    if len(embeddings) != len(chunks):
        raise RuntimeError("Embeddings count does not match chunks count")

    logger.info("Embeddings generated: %d", len(embeddings))
    logger.info("Embedding dimension: %d", len(embeddings[0]))
    logger.info("Stage 2 complete")
    return embeddings


def build_index_payload(
    provider_name: str,
    embedding_model_name: str,
    embedding_dim: int,
    chunk_size_tokens: int,
    overlap_tokens: int,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> dict:
    logger.info("Stage 3: Building index payload")

    payload_chunks = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        payload_chunks.append(
            {
                "id": chunk.id,
                "text": chunk.text,
                "tokens": chunk.tokens,
                "embedding": embedding,
            }
        )

    payload = {
        "provider": provider_name,
        "embedding_model": embedding_model_name,
        "embedding_dim": embedding_dim,
        "chunk_size_tokens": chunk_size_tokens,
        "overlap_tokens": overlap_tokens,
        "min_chunk_tokens": MIN_CHUNK_TOKENS,
        "max_chunk_tokens": MAX_CHUNK_TOKENS,
        "chunks": payload_chunks,
    }

    logger.info("Stage 3 complete")
    return payload


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    load_dotenv()

    if not (MIN_CHUNK_TOKENS <= args.chunk_size <= MAX_CHUNK_TOKENS):
        raise ValueError(f"--chunk-size must be between {MIN_CHUNK_TOKENS} and {MAX_CHUNK_TOKENS}")

    logger.info("Starting Index Pipeline")

    provider_name = args.provider
    output_path = Path(args.out) if args.out else Path(f"storage/index_{provider_name}.json")
    logger.info(
        "Config | provider=%s | chunk_size=%d | overlap=%d | out=%s",
        provider_name,
        args.chunk_size,
        args.overlap,
        output_path,
    )

    if provider_name == OPENAI_PROVIDER:
        _ = getenv_required("OPENAI_API_KEY")
        openai_model = getenv_required("EMBEDDING_MODEL")
        provider = build_embedding_provider(
            provider_name,
            openai_client=OpenAI(),
            openai_model=openai_model,
        )
        tokenizer_hint = openai_model
        embedding_model_name = openai_model
    else:
        local_model = getenv_required("LOCAL_EMBEDDING_MODEL")
        provider = build_embedding_provider(
            provider_name,
            local_model=local_model,
        )
        tokenizer_hint = None
        embedding_model_name = local_model

    chunks = load_and_chunk_document(
        DOC_PATH,
        tokenizer_model_hint=tokenizer_hint,
        chunk_size_tokens=args.chunk_size,
        overlap_tokens=args.overlap,
    )
    embeddings = generate_embeddings(provider, chunks)

    payload = build_index_payload(
        provider_name=provider.name,
        embedding_model_name=embedding_model_name,
        embedding_dim=len(embeddings[0]),
        chunk_size_tokens=args.chunk_size,
        overlap_tokens=args.overlap,
        chunks=chunks,
        embeddings=embeddings,
    )

    logger.info("Stage 4: Saving index | path=%s", output_path)
    save_json(output_path, payload)
    logger.info("Index Pipeline finished successfully")
    logger.info("Total chunks stored: %d", len(chunks))


if __name__ == "__main__":
    main()
