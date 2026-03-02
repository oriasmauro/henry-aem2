"""
RAG FAQ Query CLI: retrieve relevant document chunks and generate answers.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from embedding_provider import (
    OPENAI_PROVIDER,
    SENTENCE_TRANSFORMERS_PROVIDER,
    build_embedding_provider,
)
from rag import (
    build_response_json,
    embed_query,
    format_context,
    generate_answer,
    search_similar_chunks,
)
from utils import configure_logging, getenv_required, load_json

logger = logging.getLogger("rag-query")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG FAQ Query CLI")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--top-k", type=int, default=3, help="How many chunks to retrieve (2..5)")
    parser.add_argument(
        "--index-path",
        default="storage/index_openai.json",
        help="Path to a previously built index JSON",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Persist result into outputs/sample_queries.json",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def build_query_embedding_provider(index: dict, openai_client: OpenAI | None):
    provider_name = index["provider"]
    index_model = index["embedding_model"]

    if provider_name == OPENAI_PROVIDER:
        env_model = getenv_required("EMBEDDING_MODEL")
        if env_model != index_model:
            raise RuntimeError(
                "Embedding model mismatch between index and environment | "
                f"index={index_model} env={env_model}. Rebuild the index or align EMBEDDING_MODEL."
            )
        return build_embedding_provider(
            provider_name,
            openai_client=openai_client,
            openai_model=index_model,
        )

    if provider_name == SENTENCE_TRANSFORMERS_PROVIDER:
        env_model = getenv_required("LOCAL_EMBEDDING_MODEL")
        if env_model != index_model:
            raise RuntimeError(
                "Local embedding model mismatch between index and environment | "
                f"index={index_model} env={env_model}. Rebuild the index or align LOCAL_EMBEDDING_MODEL."
            )
        return build_embedding_provider(
            provider_name,
            local_model=index_model,
        )

    raise RuntimeError(f"Unsupported provider in index: {provider_name}")


def validate_index(index: dict[str, Any]) -> None:
    required_keys = {
        "provider",
        "embedding_model",
        "embedding_dim",
        "chunk_size_tokens",
        "overlap_tokens",
        "chunks",
    }
    missing_keys = sorted(required_keys - index.keys())
    if missing_keys:
        raise RuntimeError(f"Index is missing required keys: {', '.join(missing_keys)}")

    provider_name = index["provider"]
    if provider_name not in {OPENAI_PROVIDER, SENTENCE_TRANSFORMERS_PROVIDER}:
        raise RuntimeError(f"Unsupported provider in index: {provider_name}")

    embedding_dim = index["embedding_dim"]
    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise RuntimeError(f"Invalid embedding_dim in index: {embedding_dim}")

    chunks = index["chunks"]
    if not isinstance(chunks, list) or not chunks:
        raise RuntimeError("Index must contain a non-empty 'chunks' list")

    for idx, chunk in enumerate(chunks):
        for key in ("id", "text", "tokens", "embedding"):
            if key not in chunk:
                raise RuntimeError(f"Chunk at position {idx} is missing required key: {key}")
        if len(chunk["embedding"]) != embedding_dim:
            raise RuntimeError(
                "Chunk embedding dimension mismatch in index | "
                f"chunk_id={chunk['id']} expected={embedding_dim} actual={len(chunk['embedding'])}"
            )


def maybe_persist_output(output: dict) -> None:
    output_path = Path("outputs/sample_queries.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else []

    existing.append(output)
    output_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Output persisted to %s", output_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    load_dotenv()

    logger.info("Starting Query Pipeline")
    logger.info("Question received | top_k=%d | index=%s", args.top_k, args.index_path)

    index_path = Path(args.index_path)
    if not index_path.exists():
        raise RuntimeError(
            f"Index not found at {index_path}. Run: uv run python src/build_index.py"
        )

    llm_model = getenv_required("LLM_MODEL")
    _ = getenv_required("OPENAI_API_KEY")

    index = load_json(index_path)
    validate_index(index)
    openai_client = OpenAI()
    provider = build_query_embedding_provider(index, openai_client)

    query_embedding = embed_query(provider, args.question)
    if len(query_embedding) != index["embedding_dim"]:
        raise RuntimeError(
            "Query embedding dimension mismatch with index | "
            f"index={index['embedding_dim']} query={len(query_embedding)}"
        )
    retrieved = search_similar_chunks(query_embedding, index, top_k=args.top_k)
    context = format_context(retrieved)
    answer = generate_answer(openai_client, llm_model, args.question, context)

    output = build_response_json(args.question, answer, retrieved)
    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.save_output:
        maybe_persist_output(output)

    logger.info("Query Pipeline finished")


if __name__ == "__main__":
    main()
