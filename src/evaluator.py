"""
LLM-based evaluator for RAG responses.

Evaluates a RAG result using a separate LLM call and returns:
- score: integer from 0 to 10
- reason: justification string
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from utils import configure_logging, getenv_required, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG responses with an LLM")
    parser.add_argument(
        "--input-path",
        default="outputs/sample_queries.json",
        help="Path to a JSON object or list of query results",
    )
    parser.add_argument(
        "--output-path",
        help="Optional path to save evaluation output as JSON",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def build_evaluator_prompt(result: dict[str, Any]) -> str:
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    return (
        "Evalúa la calidad de una respuesta RAG.\n"
        "Debes analizar al menos estas dimensiones:\n"
        "1. Relevancia de los chunks respecto de la pregunta.\n"
        "2. Fidelidad de la respuesta respecto de los chunks recuperados.\n"
        "3. Completitud de la respuesta.\n\n"
        "Devuelve SOLO un objeto JSON válido con exactamente estas claves:\n"
        '{\n  "score": 0,\n  "reason": "texto..."\n}\n\n'
        "Reglas:\n"
        "- score debe ser un entero entre 0 y 10.\n"
        "- reason debe tener al menos 50 caracteres.\n"
        "- reason debe justificar el puntaje con observaciones concretas.\n"
        "- Penaliza alucinaciones, inconsistencias o chunks poco relevantes.\n"
        "- Si la respuesta está bien sustentada, explícalo explícitamente.\n\n"
        f"Entrada a evaluar:\n{payload}"
    )


def normalize_evaluation(raw_payload: dict[str, Any]) -> dict[str, Any]:
    if "score" not in raw_payload:
        raise RuntimeError("Evaluator response is missing 'score'")
    if "reason" not in raw_payload:
        raise RuntimeError("Evaluator response is missing 'reason'")

    score = raw_payload["score"]
    if isinstance(score, float):
        score = round(score)
    if not isinstance(score, int):
        raise RuntimeError("Evaluator 'score' must be numeric")
    if not 0 <= score <= 10:
        raise RuntimeError(f"Evaluator 'score' must be between 0 and 10, got: {score}")

    reason = str(raw_payload["reason"]).strip()
    if len(reason) < 50:
        raise RuntimeError("Evaluator 'reason' must contain at least 50 characters")

    return {
        "score": score,
        "reason": reason,
    }


def evaluate_result(client: OpenAI, evaluator_model: str, result: dict[str, Any]) -> dict[str, Any]:
    prompt = build_evaluator_prompt(result)
    response = client.chat.completions.create(
        model=evaluator_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un evaluador estricto de sistemas RAG. "
                    "Debes responder únicamente con JSON válido."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Evaluator returned an empty response")

    parsed = json.loads(content)
    normalized = normalize_evaluation(parsed)
    return {
        "user_question": result["user_question"],
        **normalized,
    }


def evaluate_payload(client: OpenAI, evaluator_model: str, payload: Any) -> Any:
    if isinstance(payload, list):
        return [evaluate_result(client, evaluator_model, item) for item in payload]
    if isinstance(payload, dict):
        return evaluate_result(client, evaluator_model, payload)
    raise RuntimeError("Input JSON must be an object or a list of objects")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    load_dotenv()

    _ = getenv_required("OPENAI_API_KEY")
    evaluator_model = getenv_required("EVALUATOR_MODEL")

    payload = load_json(Path(args.input_path))
    client = OpenAI()
    evaluation = evaluate_payload(client, evaluator_model, payload)

    print(json.dumps(evaluation, ensure_ascii=False, indent=2))

    if args.output_path:
        save_json(Path(args.output_path), evaluation)


if __name__ == "__main__":
    main()
