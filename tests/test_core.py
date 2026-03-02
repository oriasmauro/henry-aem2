from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from build_index import MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS, load_and_chunk_document
from evaluator import normalize_evaluation
from query import validate_index
from rag import RetrievedChunk, build_response_json, format_context, search_similar_chunks
from utils import Chunk, chunk_text_by_tokens, cosine_similarity


def test_chunk_text_by_tokens_validates_params() -> None:
    with pytest.raises(ValueError, match="chunk_size_tokens must be > 0"):
        chunk_text_by_tokens("hola", chunk_size_tokens=0, overlap_tokens=0)

    with pytest.raises(ValueError, match="overlap_tokens must be >= 0"):
        chunk_text_by_tokens("hola", chunk_size_tokens=10, overlap_tokens=-1)

    with pytest.raises(ValueError, match="overlap_tokens must be < chunk_size_tokens"):
        chunk_text_by_tokens("hola", chunk_size_tokens=10, overlap_tokens=10)


def test_chunk_text_by_tokens_respects_limits() -> None:
    text = " ".join(f"tok{i}" for i in range(500))
    chunks = chunk_text_by_tokens(text, chunk_size_tokens=120, overlap_tokens=20)
    assert len(chunks) >= 2
    assert all(1 <= c.tokens <= 120 for c in chunks)
    assert chunks[-1].tokens <= 120


def test_load_and_chunk_document_allows_short_last_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_chunks = [
        Chunk(id=0, text="a", tokens=120),
        Chunk(id=1, text="b", tokens=110),
    ] + [Chunk(id=i, text=f"c{i}", tokens=100) for i in range(2, 22)]
    fake_chunks.append(Chunk(id=22, text="last", tokens=42))

    monkeypatch.setattr("build_index.load_text", lambda *_: "doc")
    monkeypatch.setattr("build_index.count_tokens", lambda *_: 2242)
    monkeypatch.setattr("build_index.chunk_text_by_tokens", lambda *_, **__: fake_chunks)

    chunks = load_and_chunk_document(
        Path("data/faq_document.txt"),
        tokenizer_model_hint="text-embedding-3-small",
        chunk_size_tokens=120,
        overlap_tokens=20,
    )
    assert chunks[-1].tokens < MIN_CHUNK_TOKENS


def test_load_and_chunk_document_rejects_short_non_last_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_chunks = [Chunk(id=i, text=f"c{i}", tokens=100) for i in range(23)]
    fake_chunks[5] = Chunk(id=5, text="bad", tokens=42)

    monkeypatch.setattr("build_index.load_text", lambda *_: "doc")
    monkeypatch.setattr("build_index.count_tokens", lambda *_: 2242)
    monkeypatch.setattr("build_index.chunk_text_by_tokens", lambda *_, **__: fake_chunks)

    with pytest.raises(RuntimeError, match="Chunk token constraint failed"):
        load_and_chunk_document(
            Path("data/faq_document.txt"),
            tokenizer_model_hint="text-embedding-3-small",
            chunk_size_tokens=120,
            overlap_tokens=20,
        )


def test_load_and_chunk_document_rejects_over_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_chunks = [Chunk(id=i, text=f"c{i}", tokens=100) for i in range(23)]
    fake_chunks[4] = Chunk(id=4, text="too-big", tokens=MAX_CHUNK_TOKENS + 1)

    monkeypatch.setattr("build_index.load_text", lambda *_: "doc")
    monkeypatch.setattr("build_index.count_tokens", lambda *_: 2242)
    monkeypatch.setattr("build_index.chunk_text_by_tokens", lambda *_, **__: fake_chunks)

    with pytest.raises(RuntimeError, match="Chunk token constraint failed"):
        load_and_chunk_document(
            Path("data/faq_document.txt"),
            tokenizer_model_hint="text-embedding-3-small",
            chunk_size_tokens=120,
            overlap_tokens=20,
        )


def test_cosine_similarity_basic_cases() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 2]) == pytest.approx(0.0)


def test_search_similar_chunks_clamps_k_and_orders() -> None:
    index = {
        "chunks": [
            {"id": 1, "text": "a", "embedding": [1.0, 0.0]},
            {"id": 2, "text": "b", "embedding": [0.0, 1.0]},
            {"id": 3, "text": "c", "embedding": [0.8, 0.2]},
            {"id": 4, "text": "d", "embedding": [0.7, 0.3]},
            {"id": 5, "text": "e", "embedding": [0.6, 0.4]},
            {"id": 6, "text": "f", "embedding": [0.5, 0.5]},
        ]
    }

    top_small = search_similar_chunks([1.0, 0.0], index, top_k=1)
    top_large = search_similar_chunks([1.0, 0.0], index, top_k=20)

    assert len(top_small) == 2
    assert len(top_large) == 5
    assert top_large[0].score >= top_large[1].score
    assert top_large[0].id == 1


def test_format_context_and_response_json_shape() -> None:
    chunks = [
        RetrievedChunk(id=10, score=0.9876, text="Texto A"),
        RetrievedChunk(id=11, score=0.1234, text="Texto B"),
    ]

    context = format_context(chunks)
    assert "[chunk_id=10 score=0.988]" in context
    assert "Texto B" in context
    assert "---" in context

    out = build_response_json("Q", "A", chunks)
    assert out["user_question"] == "Q"
    assert out["system_answer"] == "A"
    assert len(out["chunks_related"]) == 2
    assert out["chunks_related"][0]["score"] == 0.9876


def test_validate_index_rejects_missing_required_keys() -> None:
    with pytest.raises(RuntimeError, match="missing required keys"):
        validate_index({"provider": "openai"})


def test_validate_index_rejects_embedding_dim_mismatch() -> None:
    bad_index = {
        "provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 3,
        "chunk_size_tokens": 120,
        "overlap_tokens": 20,
        "chunks": [
            {"id": 1, "text": "a", "tokens": 10, "embedding": [0.1, 0.2]},
        ],
    }

    with pytest.raises(RuntimeError, match="Chunk embedding dimension mismatch"):
        validate_index(bad_index)


def test_normalize_evaluation_accepts_valid_payload() -> None:
    payload = {
        "score": 8,
        "reason": (
            "Puntaje 8: la respuesta usa chunks relevantes y está bien apoyada "
            "en el contexto, aunque podría ser más completa."
        ),
    }

    normalized = normalize_evaluation(payload)
    assert normalized["score"] == 8
    assert "Puntaje 8" in normalized["reason"]


def test_normalize_evaluation_rejects_short_reason() -> None:
    with pytest.raises(RuntimeError, match="at least 50 characters"):
        normalize_evaluation({"score": 7, "reason": "Muy corto"})
