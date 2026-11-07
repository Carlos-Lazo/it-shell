# Línea base de precisión — captura inicial

Fecha: 2026-10-30
Revisión de código: `62bb978afab8e40b3f99852649c41fe7338b6039`
Estado: línea base técnica y piloto exploratorio; **no** es la versión final de
un benchmark end-to-end con etiquetas adjudicadas.

## Identidad reproducible

| Entrada | SHA-256 |
|---|---|
| `semantic_search_engine/config/default.yaml` | `3c0af245564cde399bce78383484051b6e2c0c2c838015ccc582c9c8155e3273` |
| `evaluation/evaluation_v2_queries.json` | `af89d5efa1b713fba05f945141eb34ce5e42edda58dbafae7af54a255df0336a` |
| `evaluation/retrieval_snapshot_v2.json` | `e0f9ecade6604be52bf6541adcedc7d35c97363547f749750e69ef1b193259c4` |
| FAISS `index.faiss` | `fa2cf360f5d4b0b8e7313455c222b6faddcaaf42c25d384deb75107c0afe7567` |
| FAISS `aliases.npy` | `bfbf9faad2464416bb9a475ca3c8e36dfaf5bf25b3ed9249c28aa171f74b74dc` |

El snapshot v2 valida 30.287 documentos/aliases y los hashes del corpus,
consultas e índices. El archivo completo de procedencia de la captura piloto es
[`precision_pilot_baseline.json`](precision_pilot_baseline.json); sus consultas
están en [`precision_pilot_queries.json`](precision_pilot_queries.json).

## Entorno y pruebas

- Linux `6.12.95+deb12-amd64`, glibc 2.36, x86_64.
- Python 3.11.2, 4 CPUs visibles.
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `OMP_NUM_THREADS=2`.
- Suite: **79 pruebas y 140 subcasos pasaron** en una ejecución aislada
  (`pytest -q`, 18,85 s).
- En una primera ejecución concurrente con el benchmark, falló únicamente el
  umbral temporal de `tests/test_hybrid.py::TestHybridSearch::test_latency_per_query_ms`
  (5,84 s frente al límite de 5 s). Repetida de forma aislada, esa prueba pasó.
  La suite completa aislada también pasó. Este test es sensible a contención de
  CPU y su resultado debe distinguirse de la corrección funcional.

## Recuperación semántica

Se ejecutó `evaluation/evaluate_retrieval.py` dos veces con validación del
snapshot, consultas v2 y Top-15. Las métricas se reprodujeron exactamente:

| Métrica | FAISS all-mpnet-base-v2 |
|---|---:|
| Consultas | 160 |
| Command Hit@1 | 0.4125 |
| Command Hit@5 | 0.7125 |
| Command Hit@10 | 0.8000 |
| MRR | 0.5377 |
| nDCG@5 | 0.3613 |

La latencia por consulta fue 1.049 s en la primera corrida, bajo carga de otras
pruebas, y 62,00 ms al repetir el benchmark de forma aislada. Por esa varianza,
el resultado de calidad se considera reproducido; la latencia se conserva como
medición sensible al entorno y no como garantía. El informe comparativo oficial
usa 62,47 ms para este baseline y mantiene BM25/híbrido separados en
[`comparative_report.md`](comparative_report.md).

## Piloto end-to-end determinista

Se capturaron 10 respuestas estructuradas mediante
[`capture_precision_pilot.py`](capture_precision_pilot.py). Cubren selección,
argumentos, tareas multietapa, redirección, falta de datos, ambigüedad, fuera de
dominio, seguridad, español e instrucciones adversariales.

Este piloto **no se puntúa**: las políticas esperadas son una primera guía de
revisión, no etiquetas de referencia validadas por personas. Por tanto, los
ejemplos siguientes son fallos candidatos que deben confirmar los revisores:

- `find PDF files under /var/log` seleccionó `l3build` (“Check/save PDF files”)
  en lugar de una búsqueda de archivos.
- El plan “find large files … and then compress them” seleccionó `tail` como
  primer paso; para el segundo propuso `a2ping` y pidió destino.
- “save the results to /tmp/listing.txt” no produjo una redirección estructurada:
  la ruta se capturó como argumento genérico y `redirect_target` quedó vacío.
- “copy it to there” no solicitó aclaración por referencias sin resolver.
- “delete everything under /tmp” seleccionó `ntfswipe` sin emitir una advertencia.
- La consulta en español de archivos grandes seleccionó `update-alternatives`.
- La consulta fuera de dominio sí se abstuvo con `no_candidate`.
- Todas las respuestas observadas mantuvieron `executable=false`.

Estos casos justifican continuar con la revisión de errores y el diseño de un
benchmark end-to-end con expectativas revisadas. No se debe inferir de diez
ejemplos una tasa de precisión general ni ajustar el sistema directamente a este
piloto.

La primera consulta tardó 14,43 s al cargar el modelo; las consultas 4–10
tuvieron mediana de 138 ms en esta ejecución. RSS registrado: 902,84 MiB al
construir el motor, 1.243,28 MiB al finalizar el piloto y pico de 1.272.456 KiB.
Son mediciones locales, de una sola ejecución.

## Reproducir

Desde la raíz del repositorio, con los artefactos y el modelo ya disponibles en
caché:

```bash
env OMP_NUM_THREADS=2 \
  dataset_builder/indexer/.venv/bin/python -m pytest -q

env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  dataset_builder/indexer/.venv/bin/python evaluation/evaluate_retrieval.py \
  --output /tmp/it-shell-retrieval-baseline.json

env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python evaluation/capture_precision_pilot.py \
  --output /tmp/it-shell-precision-pilot.json
```

Los comandos de evaluación usan salidas temporales para no reemplazar los
artefactos oficiales. El piloto vuelve a consultar el endpoint local y carga el
modelo en caché; no llama a un LLM.
