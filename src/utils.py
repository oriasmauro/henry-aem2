"""
Utility functions for RAG implementation, including text loading, tokenization, chunking, similarity calculation, and JSON handling.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken

# -------------------------
# Logging configuration
# -------------------------


def configure_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


logger = logging.getLogger("rag-utils")


# -------------------------
# File handling
# -------------------------


def load_text(path: str | Path) -> str:
    p = Path(path)
    logger.info("Loading document from %s", p)

    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 failed, retrying with latin-1")
        text = p.read_text(encoding="latin-1")

    logger.info("Loaded document (%d characters)", len(text))
    return text


# -------------------------
# Token utilities
# -------------------------


def get_tokenizer(model_name: str | None = None):
    if model_name:
        try:
            logger.debug("Using tokenizer for model %s", model_name)
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning("Tokenizer not found for model %s, fallback to cl100k_base", model_name)
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model_name: str | None = None) -> int:
    enc = get_tokenizer(model_name)
    tokens = len(enc.encode(text))
    logger.debug("Counted %d tokens", tokens)
    return tokens


@dataclass(frozen=True)
class Chunk:
    id: int
    text: str
    tokens: int


# -------------------------
# Chunking
# -------------------------


def chunk_text_by_tokens(
    text: str,
    chunk_size_tokens: int = 120,
    overlap_tokens: int = 20,
    tokenizer_model: str | None = None,
) -> list[Chunk]:
    logger.info(
        "Starting chunking | chunk_size=%d | overlap=%d",
        chunk_size_tokens,
        overlap_tokens,
    )

    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be < chunk_size_tokens")

    enc = get_tokenizer(tokenizer_model)
    tokens = enc.encode(text)

    logger.info("Total tokens in document: %d", len(tokens))

    chunks: list[Chunk] = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = min(start + chunk_size_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens).strip()

        if chunk_text:
            chunk = Chunk(
                id=chunk_id,
                text=chunk_text,
                tokens=len(chunk_tokens),
            )
            chunks.append(chunk)
            logger.debug(
                "Created chunk id=%d | tokens=%d",
                chunk.id,
                chunk.tokens,
            )
            chunk_id += 1

        if end == len(tokens):
            break
        start = end - overlap_tokens

    logger.info("Chunking finished | total_chunks=%d", len(chunks))
    return chunks


# -------------------------
# Similarity
# -------------------------


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a_vec = np.array(list(a), dtype=np.float32)
    b_vec = np.array(list(b), dtype=np.float32)
    denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)

    if denom == 0:
        logger.warning("Zero norm encountered in cosine similarity")
        return 0.0

    score = float(np.dot(a_vec, b_vec) / denom)
    return score


# -------------------------
# JSON helpers
# -------------------------


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved JSON to %s", p)


def load_json(path: str | Path) -> Any:
    logger.info("Loading JSON from %s", path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def getenv_required(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val
