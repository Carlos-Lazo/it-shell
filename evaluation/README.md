# Evaluaciones de retrieval — Fase 2

Esta carpeta contiene los benchmarks de retrieval, reranking y búsqueda léxica
o híbrida. Los comandos de esta guía se ejecutan desde la raíz del repositorio.
El conjunto de evaluación principal es `evaluation/evaluation_v2_queries.json`
(160 consultas en inglés).

## Requisitos y artefactos

Los benchmarks usan el entorno virtual del indexer, que debe tener instaladas
las dependencias de evaluación e indexación:

```bash
dataset_builder/indexer/.venv/bin/pip install \
  -r evaluation/requirements.txt \
  -r dataset_builder/indexer/requirements.txt
```

También deben existir estos artefactos locales:

- Índice FAISS: `dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss`
- Aliases FAISS: `dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy`
- Índice BM25: `dataset_builder/indexer/artifacts/bm25_indexs/`
- Corpus: `dataset_builder/corpus/`

Si es necesario reconstruir el índice BM25, desde la raíz:

```bash
dataset_builder/indexer/.venv/bin/python \
  dataset_builder/indexer/bm25_indexer.py \
  --corpus dataset_builder/corpus \
  --output dataset_builder/indexer/artifacts/bm25_indexs \
  --threads 4
```

Para ejecución offline, los modelos de Sentence Transformers y Cross-Encoder
deben estar ya disponibles en la caché local. `HF_HUB_OFFLINE=1` y
`TRANSFORMERS_OFFLINE=1` evitan descargas durante el benchmark.

## Baseline FAISS y rerankers

`evaluate_reranker.py` evalúa a la vez FAISS puro y un Cross-Encoder sobre los
candidatos iniciales. El JSON generado contiene `baseline_summary` y
`reranked_summary`; por eso la ejecución MiniLM también proporciona las métricas
del baseline que usa el informe consolidado.

Se usa `--initial-top-k 15` para mantener la configuración evaluada. Cada modelo
se guarda en un archivo independiente:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
dataset_builder/indexer/.venv/bin/python evaluation/evaluate_reranker.py \
  --queries evaluation/evaluation_v2_queries.json \
  --corpus dataset_builder/corpus \
  --index dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss \
  --aliases dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy \
  --initial-top-k 15 \
  --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --output evaluation/reranker_benchmark_v2_top15.json
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
dataset_builder/indexer/.venv/bin/python evaluation/evaluate_reranker.py \
  --queries evaluation/evaluation_v2_queries.json \
  --corpus dataset_builder/corpus \
  --index dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss \
  --aliases dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy \
  --initial-top-k 15 \
  --cross-encoder cross-encoder/stsb-distilroberta-base \
  --output evaluation/reranker_benchmark_distilroberta.json
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
dataset_builder/indexer/.venv/bin/python evaluation/evaluate_reranker.py \
  --queries evaluation/evaluation_v2_queries.json \
  --corpus dataset_builder/corpus \
  --index dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss \
  --aliases dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy \
  --initial-top-k 15 \
  --cross-encoder BAAI/bge-reranker-base \
  --output evaluation/reranker_benchmark_bge.json
```

## BM25 con Pyserini/Lucene

La ejecución compara BM25 con FAISS dentro del mismo benchmark y exporta ambas
mediciones al JSON:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
dataset_builder/indexer/.venv/bin/python evaluation/evaluate_bm25_hybrid.py \
  --mode bm25 \
  --queries evaluation/evaluation_v2_queries.json \
  --corpus dataset_builder/corpus \
  --bm25-index dataset_builder/indexer/artifacts/bm25_indexs \
  --faiss-index dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss \
  --aliases dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy \
  --initial-top-k 15 \
  --output evaluation/bm25_benchmark_top15.json
```

## Búsqueda híbrida BM25 + dense

`--alpha 0.5` da el mismo peso a las puntuaciones normalizadas de BM25 y FAISS.
El nombre de salida debe ser `hybrid_benchmark_top15.json` para que
`compare_all.py` lo tome como artefacto preferido:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
dataset_builder/indexer/.venv/bin/python evaluation/evaluate_bm25_hybrid.py \
  --mode hybrid \
  --alpha 0.5 \
  --queries evaluation/evaluation_v2_queries.json \
  --corpus dataset_builder/corpus \
  --bm25-index dataset_builder/indexer/artifacts/bm25_indexs \
  --faiss-index dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss \
  --aliases dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy \
  --initial-top-k 15 \
  --output evaluation/hybrid_benchmark_top15.json
```

## Generación del informe comparativo completo

Después de ejecutar los benchmarks, genera la tabla Markdown y el JSON unificado:

```bash
python3 evaluation/compare_all.py
```

Para validar las entradas y ver la tabla sin sobrescribir los informes:

```bash
python3 evaluation/compare_all.py --dry-run
```

Los resultados quedan en:

- `evaluation/comparative_table.md`: tabla comparativa compacta.
- `evaluation/comparative_all.json`: métricas, latencias y marca de comparabilidad.
- `evaluation/conclusiones_fase_2.txt`: interpretación y decisión para cerrar la fase.

El comparador verifica que las configuraciones cubran el mismo número de
consultas y contrasta el baseline interno de cada experimento con el baseline
FAISS de referencia. Tener 160 consultas no garantiza por sí solo una
comparación equivalente: los BM25 e híbrido actuales quedan marcados como no
comparables porque sus JSON declaran un baseline interno distinto. Para una
comparación controlada, repetir esas evaluaciones usando el mismo snapshot de
corpus, índices, aliases y relevancias.

## Pruebas funcionales de índices

Con los artefactos requeridos ya creados, se pueden ejecutar las pruebas de
BM25 y del buscador híbrido:

```bash
dataset_builder/indexer/.venv/bin/python -m pytest tests/test_bm25.py -v
dataset_builder/indexer/.venv/bin/python -m pytest tests/test_hybrid.py -v
```
