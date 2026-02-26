# FAQ Support Chatbot (RAG) - HR SaaS

Este proyecto implementa un chatbot de soporte para FAQs internas usando arquitectura RAG (Retrieval-Augmented Generation). El objetivo es reducir preguntas repetitivas del equipo de soporte recuperando contexto relevante desde documentación privada de la empresa y generando respuestas fundamentadas con un LLM. El flujo está dividido en dos pipelines: indexación (documento -> chunks -> embeddings -> almacenamiento) y consulta (pregunta -> embedding -> búsqueda vectorial -> respuesta). El diseño prioriza trazabilidad y auditabilidad, devolviendo siempre un JSON estructurado con la respuesta y los chunks usados.

## Objetivo del sistema

- Procesar un documento de texto plano (`data/faq_document.txt`).
- Fragmentarlo en chunks para recuperación semántica (>=20 chunks).
- Generar embeddings y persistirlos junto al texto.
- Responder preguntas con RAG y devolver JSON con:
  - `user_question`
  - `system_answer`
  - `chunks_related`
- Mantener transparencia del contexto recuperado para depuración/auditoría.

## Arquitectura

### 1) Pipeline de indexación (`src/build_index.py`)

Etapas:

1. Carga de documento con manejo de codificación (`utf-8`, fallback `latin-1`).
2. Chunking por tokens con ventana fija + overlap.
3. Generación de embeddings de cada chunk con OpenAI Embeddings.
4. Persistencia en `storage/index.json` con texto + metadata + vector.

### 2) Pipeline de consulta (`src/query.py`)

Etapas:

1. Embedding de la pregunta del usuario.
2. Búsqueda vectorial tipo k-NN (brute force) con similitud coseno.
3. Recuperación de `top-k` chunks relevantes (acotado a 2..5).
4. Ensamblado de contexto.
5. Generación de respuesta con LLM usando solo ese contexto.
6. Salida JSON por `stdout` con las 3 claves requeridas.

## Estructura del proyecto

```text
.
├── data/
│   └── faq_document.txt
├── outputs/
│   └── sample_queries.json
├── src/
│   ├── build_index.py
│   ├── query.py
│   ├── rag.py
│   ├── utils.py
│   ├── inspect_index.py
│   └── evaluator.py
├── storage/
│   └── index.json
├── tests/
│   └── test_core.py
├── .env.example
├── pyproject.toml
└── uv.lock
```

## Requisitos

- Python `>=3.11`
- `uv` instalado (recomendado para reproducibilidad)
- API key de OpenAI

Dependencias principales (pinned en `pyproject.toml`):

- `openai==1.61.1`
- `tiktoken==0.8.0`
- `numpy==2.1.3`
- `python-dotenv==1.0.1`

## Instalación y configuración

1. Crear entorno e instalar dependencias:

```bash
uv sync --dev
```

2. Configurar variables de entorno:

```bash
cp .env.example .env
```

Variables requeridas:

```env
OPENAI_API_KEY=your-key-here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

3. (Opcional) Validar entorno:

```bash
uv run python -V
uv run ruff check .
uv run pytest -q
```

## Cómo ejecutar

### A) Construir índice vectorial

```bash
uv run python src/build_index.py
```

Salida esperada:

- Log de etapas 1..4.
- Archivo `storage/index.json` con chunks y embeddings.

### B) Realizar una consulta

```bash
uv run python src/query.py \
  --question "¿Cómo se solicita vacaciones y qué aprobaciones requiere?" \
  --top-k 3 \
  --save-output
```

Comportamiento:

- Imprime JSON en `stdout`.
- Si se usa `--save-output`, agrega el resultado a `outputs/sample_queries.json`.

## Formato de salida JSON

El contrato de salida contiene exactamente estas claves:

```json
{
  "user_question": "string",
  "system_answer": "string",
  "chunks_related": [
    {
      "id": 10,
      "score": 0.5521,
      "text": "..."
    }
  ]
}
```

## Decisiones técnicas y trade-offs

### Chunking: tamaño fijo por tokens + overlap

Decisión:

- Estrategia usada: ventana fija por tokens (`chunk_size=120`, `overlap=20`).
- Validación en indexación: chunks en rango recomendado 50-500 tokens; se permite que el último chunk sea menor para no perder cobertura del final del documento.

Por qué:

- Es simple, predecible y fácil de depurar.
- Mantiene continuidad semántica local gracias al overlap.

Trade-offs:

- Puede cortar ideas a mitad de oración (menos natural que chunking semántico).
- El último chunk puede quedar corto; se aceptó por completitud del documento.

### Búsqueda vectorial: k-NN exacto + coseno

Decisión:

- Método actual: k-NN exacto (scan de todos los chunks) con `cosine_similarity`.
- `top_k` acotado a 2..5 para reducir ruido y cumplir criterio de recuperación.

Por qué:

- Con un corpus pequeño/mediano, exacto es suficiente y transparente.
- Coseno funciona bien para embeddings densos de OpenAI.

Trade-offs:

- No escala tan bien como ANN para corpus muy grandes.
- No aplica reranking ni filtros híbridos (keyword + vector), por lo que aún hay margen de mejora en precisión.

### Almacenamiento: JSON local

Decisión:

- Se guarda índice en `storage/index.json` (texto + metadata + embedding).

Por qué:

- Facilita inspección manual y debugging.
- Cero dependencia de infraestructura externa.

Trade-offs:

- Menor rendimiento/escala que un vector store dedicado (FAISS, Qdrant, pgvector).

## RAG: por qué este enfoque

Esto es RAG porque el sistema primero recupera chunks relevantes desde una base de conocimiento privada y luego genera la respuesta condicionada por ese contexto. Beneficios:

- Conocimiento actualizable sin reentrenar el modelo.
- Mayor transparencia (se exponen chunks y scores).
- Mejor control para reducir alucinaciones frente a generación sin contexto.

## Calidad, pruebas y linting

Tests principales en `tests/test_core.py`:

- validación de parámetros de chunking,
- reglas de tamaño de chunk (incluyendo último chunk corto),
- similitud coseno,
- ranking y clamp de `top_k`,
- formato de contexto y JSON de salida.

Ejecutar:

```bash
uv run pytest -q
uv run ruff check . --fix
```

## Estado del entregable vs rúbrica

- Indexación modular: implementada.
- Query pipeline modular: implementado.
- JSON con claves requeridas: implementado.
- k-NN + coseno explícito: implementado.
- Documento y almacenamiento local: implementados.
- Ejemplos de salida: `outputs/sample_queries.json` (actualmente 2 ejemplos; se recomienda dejar >=3 para evaluación final).
- Agente evaluador (bonus): archivo `src/evaluator.py` existe pero aún no está implementado.

## Próximos pasos recomendados

1. Completar `src/evaluator.py` (score 0-10 + justificación de fidelidad al contexto).
2. Migrar búsqueda a ANN/vector DB para mayor escala.
3. Añadir retrieval híbrido (vector + keyword) y/o reranker.
4. Agregar pruebas de integración E2E con preguntas de negocio representativas.
