"""
RAG (Retrieval-Augmented Generation) core logic for FAQ support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from embedding_provider import EmbeddingProvider
from utils import cosine_similarity

logger = logging.getLogger("rag-core")


@dataclass(frozen=True)
class RetrievedChunk:
    id: int
    score: float
    text: str


def embed_query(provider: EmbeddingProvider, question: str) -> list[float]:
    logger.info("Embedding user question | provider=%s", provider.name)
    embedding = provider.embed_texts([question])[0]
    logger.info("Query embedding ready | dim=%d", len(embedding))
    return embedding


def search_similar_chunks(
    query_embedding: list[float],
    index: dict[str, Any],
    top_k: int = 3,
) -> list[RetrievedChunk]:
    # Rúbrica: devolver 2..5 chunks
    k = max(2, min(int(top_k), 5))
    logger.info("Searching similar chunks (k-NN) | top_k=%d", k)

    scored: list[RetrievedChunk] = []
    for ch in index["chunks"]:
        score = cosine_similarity(query_embedding, ch["embedding"])
        scored.append(RetrievedChunk(id=ch["id"], score=score, text=ch["text"]))

    scored.sort(key=lambda x: x.score, reverse=True)
    top = scored[:k]

    logger.info("Top chunks selected: %s", [(c.id, round(c.score, 4)) for c in top])
    return top


def format_context(chunks: list[RetrievedChunk]) -> str:
    # Contexto con ids para trazabilidad
    parts = []
    for c in chunks:
        parts.append(f"[chunk_id={c.id} score={c.score:.3f}]\n{c.text}")
    context = "\n\n---\n\n".join(parts)
    logger.info("Context assembled | chunks=%d | chars=%d", len(chunks), len(context))
    return context


def generate_answer(
    client: OpenAI,
    llm_model: str,
    question: str,
    context: str,
) -> str:
    logger.info("Generating answer with LLM | model=%s", llm_model)

    system_msg = (
        "Eres un asistente experto de soporte para FAQs internas. "
        "Usa los siguientes chunks recuperados para responder la pregunta. Asegúrate de referenciar la fuente de los datos. "
        "Si la respuesta no está en el contexto, responde exactamente: 'No lo sé con la información disponible en el documento.' "
        "Sé claro y directo."
    )

    user_msg = f"Contexto:\n{context}\n\nPregunta:\n{question}\n\nRespuesta:"

    resp = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0002,
    )

    answer = resp.choices[0].message.content.strip()
    logger.info("Answer generated | chars=%d", len(answer))
    return answer


def build_response_json(question: str, answer: str, chunks: list[RetrievedChunk]) -> dict:
    # Rúbrica: respuesta JSON con pregunta, respuesta y chunks relacionados (id, score, texto)
    return {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": [
            {"id": c.id, "score": round(c.score, 4), "text": c.text} for c in chunks
        ],
    }
