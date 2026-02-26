"""
RAG FAQ Query CLI: Retrieve relevant document chunks and generate answers using LLM.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag import (
    build_response_json,
    embed_query,
    format_context,
    generate_answer,
    search_similar_chunks,
)
from utils import configure_logging, getenv_required, load_json

logger = logging.getLogger("rag-query")

INDEX_PATH = Path("storage/index.json")


def parse_args():
    p = argparse.ArgumentParser(description="RAG FAQ Query CLI")
    p.add_argument("--question", required=True, help="User question")
    p.add_argument("--top-k", type=int, default=3, help="How many chunks to retrieve (2..5)")
    p.add_argument(
        ("--save-output"),
        action="store_true",
        help="Persist result into outputs/sample_queries.json",
    )
    return p.parse_args()


def main() -> None:
    configure_logging("INFO")
    load_dotenv()

    logger.info("Starting Query Pipeline")

    _ = getenv_required("OPENAI_API_KEY")
    embedding_model = getenv_required("EMBEDDING_MODEL")
    llm_model = getenv_required("LLM_MODEL")

    args = parse_args()
    logger.info("Question received | top_k=%d", args.top_k)

    if not INDEX_PATH.exists():
        raise RuntimeError("Index not found. Run: uv run python src/build_index.py")

    index = load_json(INDEX_PATH)

    # Validación de consistencia
    if index.get("embedding_model") != embedding_model:
        logger.warning(
            "Embedding model mismatch | index=%s env=%s",
            index.get("embedding_model"),
            embedding_model,
        )

    client = OpenAI()

    q_emb = embed_query(client, embedding_model, args.question)
    retrieved = search_similar_chunks(q_emb, index, top_k=args.top_k)
    context = format_context(retrieved)
    answer = generate_answer(client, llm_model, args.question, context)

    out = build_response_json(args.question, answer, retrieved)

    # Solo JSON por stdout (para downstream)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.save_output:
        output_path = Path("outputs/sample_queries.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            existing = []

        existing.append(out)

        output_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info("Output persisted to %s", output_path)

    logger.info("Query Pipeline finished")


if __name__ == "__main__":
    main()
