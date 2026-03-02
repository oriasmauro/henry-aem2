# FAQ Support Chatbot (RAG) - HR SaaS

Este proyecto implementa un chatbot de soporte para FAQs internas usando arquitectura RAG (Retrieval-Augmented Generation). El objetivo es reducir preguntas repetitivas del equipo de soporte recuperando contexto relevante desde documentación privada de la empresa y generando respuestas fundamentadas con un LLM. El flujo está dividido en dos pipelines principales: indexación (documento -> chunks -> embeddings -> almacenamiento) y consulta (pregunta -> embedding -> búsqueda vectorial -> respuesta), más un tercer flujo opcional de evaluación para revisar la calidad de las respuestas generadas. El diseño prioriza trazabilidad y auditabilidad, devolviendo siempre un JSON estructurado con la respuesta y los chunks usados. Como bonus, el proyecto también incorpora dos extensiones: soporte para dos backends de embeddings (OpenAI y `sentence-transformers`) para realizar comparaciones de calidad/costo/latencia a modo de benchmark, y un agente evaluador basado en LLM que puntúa la calidad de las respuestas RAG y justifica el resultado. 

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
3. Generación de embeddings de cada chunk con uno de dos backends:
   - OpenAI Embeddings
   - `sentence-transformers` (modelo local)
4. Persistencia en `storage/index_<provider>.json` con texto + metadata + vector.

### 2) Pipeline de consulta (`src/query.py`)

Etapas:

1. Embedding de la pregunta del usuario usando la misma opción de embeddings que se usó al crear ese índice.
2. Búsqueda vectorial tipo k-NN (brute force) con similitud coseno.
3. Recuperación de `top-k` chunks relevantes (acotado a 2..5).
4. Ensamblado de contexto.
5. Generación de respuesta con LLM usando solo ese contexto.
6. Salida JSON por `stdout` con las 3 claves requeridas.

### 3) Pipeline de evaluación (`src/evaluator.py`) - opcional

Etapas:

1. Carga de una salida del pipeline de consulta (`user_question`, `system_answer`, `chunks_related`).
2. Envío de esa salida a un segundo LLM que actúa como juez.
3. Evaluación de:
   - relevancia de los chunks,
   - fidelidad de la respuesta respecto del contexto,
   - completitud,
   - posibles alucinaciones o inconsistencias.
4. Salida JSON con:
   - `score` (0 a 10)
   - `reason` (justificación del puntaje)

## Estructura del proyecto

```text
.
├── data/
│   └── faq_document.txt
├── outputs/
│   ├── sample_queries.json
│   └── sample_queries_benchmark.json
├── src/
│   ├── build_index.py
│   ├── embedding_provider.py
│   ├── query.py
│   ├── rag.py
│   ├── utils.py
│   ├── inspect_index.py
│   └── evaluator.py
├── storage/
│   ├── index_openai.json
│   └── index_sentence-transformers.json
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
- `sentence-transformers==3.0.1`
- `torch==2.2.2` (con marcadores por plataforma)
- `tiktoken==0.8.0`
- `numpy==1.26.4`
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
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-4o-mini
EVALUATOR_MODEL=gpt-4o-mini
```

Notas:

- `OPENAI_API_KEY` sigue siendo necesaria para la generación final de respuestas con el LLM.
- `EMBEDDING_MODEL` se usa cuando el backend de embeddings es OpenAI.
- `LOCAL_EMBEDDING_MODEL` se usa cuando el backend de embeddings es `sentence-transformers`.
- `EVALUATOR_MODEL` se usa para el agente evaluador (bonus).

3. (Opcional) Validar entorno:

```bash
uv run python -V
uv run ruff check .
uv run pytest -q
```

## Cómo ejecutar

### A) Construir índice vectorial

Con OpenAI:

```bash
uv run python src/build_index.py \
  --provider openai \
  --out storage/index_openai.json
```

Con `sentence-transformers`:

```bash
uv run python src/build_index.py \
  --provider sentence-transformers \
  --out storage/index_sentence-transformers.json
```

Salida esperada:

- Log de etapas 1..4.
- Archivo `storage/index_<provider>.json` con chunks, embeddings y metadata del backend usado.

### B) Realizar una consulta

```bash
uv run python src/query.py \
  --index-path storage/index_openai.json \
  --question "¿Cómo se solicita vacaciones y qué aprobaciones requiere?" \
  --top-k 3 \
  --save-output
```

Comportamiento:

- Imprime JSON en `stdout`.
- Si se usa `--save-output`, agrega el resultado a `outputs/sample_queries.json`.
- La pregunta se convierte a embedding usando la misma configuración con la que se generaron los embeddings del documento al crear ese índice.

### C) Benchmark de embeddings (bonus)

Esto no es parte obligatoria de la consigna, pero el proyecto permite comparar dos estrategias de embeddings manteniendo el resto del pipeline igual:

1. Construir un índice con OpenAI.
2. Construir otro índice con `sentence-transformers`.
3. Ejecutar las mismas preguntas contra ambos índices.
4. Comparar relevancia de chunks, costo y latencia.

Ejemplo:

```bash
uv run python src/query.py \
  --index-path storage/index_openai.json \
  --question "¿Cómo se calcula la facturación por usuario activo?"

uv run python src/query.py \
  --index-path storage/index_sentence-transformers.json \
  --question "¿Cómo se calcula la facturación por usuario activo?"
```

La idea es usarlo como benchmark exploratorio, no como cambio de alcance del entregable base.

Archivo de referencia:

- `outputs/sample_queries_benchmark.json`: contiene las mismas preguntas ejecutadas contra ambos índices para comparar recuperación y respuesta.

### D) Evaluar respuestas con un agente evaluador (bonus)

El proyecto incluye un evaluador basado en LLM que revisa la calidad de una salida RAG usando:

- relevancia de los chunks recuperados,
- fidelidad de la respuesta respecto del contexto,
- completitud de la respuesta,
- posibles alucinaciones o inconsistencias.

Entrada:

- Un objeto o lista de objetos con el formato del pipeline de consulta:
  - `user_question`
  - `system_answer`
  - `chunks_related`

Salida:

- Un objeto o lista de objetos con:
  - `user_question`
  - `score` (entero de 0 a 10)
  - `reason` (justificación de al menos 50 caracteres)

Ejecutar:

```bash
uv run python src/evaluator.py \
  --input-path outputs/sample_queries.json \
  --output-path outputs/sample_queries_evaluation.json
```

Esto usa un segundo llamado al LLM como juez. No forma parte del flujo mínimo obligatorio, pero sirve para control de calidad automatizado.

Flujo recomendado de uso:

1. Generar o guardar respuestas con `src/query.py`.
2. Ejecutar `src/evaluator.py` sobre ese archivo de resultados.
3. Revisar el `score` y el `reason` para detectar respuestas débiles, poco fundamentadas o potencialmente alucinadas.

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

### Bonus: dos backends de embeddings para comparación

Decisión:

- Se agregó soporte dual para embeddings:
  - OpenAI (`text-embedding-3-small`)
  - `sentence-transformers` (por defecto `all-MiniLM-L6-v2`)

Por qué:

- Permite comparar una opción administrada y una local con el mismo corpus, chunking y método de búsqueda.
- Sirve como benchmark práctico de costo, portabilidad y calidad de recuperación.

Trade-offs:

- Agrega complejidad de configuración (más variables de entorno, más dependencias).
- Hay consideraciones de compatibilidad binaria por plataforma (`torch`, `numpy`) que no existen cuando se usa solo OpenAI.


### Comparación: OpenAI vs Sentence Transformers

Sí, hay una diferencia real en los vectores generados. Aunque ambos representan texto como embeddings densos, no producen el mismo espacio vectorial, ni la misma dimensionalidad, ni exactamente el mismo criterio de similitud. Eso significa que, ante la misma pregunta y el mismo documento, pueden recuperar chunks distintos o cambiar el orden de relevancia.

| Aspecto | OpenAI Embeddings | Sentence Transformers |
| --- | --- | --- |
| Backend | API remota administrada | Modelo local |
| Modelo actual | `text-embedding-3-small` | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensionalidad | Mayor (en este proyecto: 1536) | Menor (en este proyecto: 384) |
| Espacio vectorial | Propio de OpenAI | Propio del modelo local |
| Compatibilidad entre vectores | Solo comparable con embeddings del mismo modelo | Solo comparable con embeddings del mismo modelo |
| Costo por consulta | Sí, por uso de API | No por API; sí costo local de CPU/RAM |
| Latencia | Depende de red + API | Depende de hardware local |
| Portabilidad | Requiere credenciales y conexión | Puede correr offline una vez descargado |
| Variabilidad de ranking | Suele tener buena robustez semántica | Puede rendir muy bien, pero cambia según modelo elegido |

Implicación práctica:

- No se deben mezclar embeddings de OpenAI con embeddings de `sentence-transformers` dentro del mismo índice.
- La pregunta del usuario debe embebderse con el mismo backend usado para construir el índice.
- Si cambias el modelo de embeddings, debes reconstruir el índice.

En este proyecto, eso ya está contemplado:

- Cada índice guarda su `provider` y `embedding_model`.
- `query.py` detecta qué backend se usó al indexar y reutiliza ese mismo backend para la consulta.
- El archivo `outputs/sample_queries_benchmark.json` permite observar cómo cambian los chunks recuperados entre ambos enfoques.

### Almacenamiento: JSON local

Decisión:

- Se guardan índices en archivos JSON locales (`storage/index_openai.json`, `storage/index_sentence-transformers.json`).

Por qué:

- Facilita inspección manual y debugging.
- Cero dependencia de infraestructura externa.

Trade-offs:

- Menor rendimiento y escalabilidad que un vector store dedicado como ChromaDB, FAISS, Qdrant o pgvector.

### Bonus: agente evaluador con LLM

Decisión:

- Se implementó un evaluador con un segundo llamado a OpenAI que actúa como juez de calidad sobre la salida RAG.

Por qué:

- Se alinea mejor con la consigna de “agente evaluador especializado”.
- Permite evaluar grounding, relevancia y completitud con más flexibilidad.

Trade-offs:

- Aumenta costo y latencia porque agrega otra llamada al modelo.
- La evaluación sigue siendo probabilística: mejora el juicio cualitativo, pero no reemplaza validación humana en casos críticos.

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
- validación del esquema del índice,
- normalización del contrato del evaluador,
- formato de contexto y JSON de salida.

Ejecutar:

```bash
uv run pytest -q
uv run ruff check . --fix
```


## Próximos pasos 

1. Migrar el almacenamiento y la búsqueda a una base vectorial (por ejemplo, ChromaDB) o a un motor ANN para mayor escala.
2. Migrar búsqueda a ANN/vector DB para mayor escala.
3. Añadir retrieval híbrido (vector + keyword) y/o reranker.
4. Agregar pruebas de integración E2E con preguntas de negocio representativas.
