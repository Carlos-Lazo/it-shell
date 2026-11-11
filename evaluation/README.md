# Evaluaciones de retrieval — Fase 2

Esta carpeta contiene los benchmarks de retrieval, reranking y búsqueda léxica
o híbrida. Los comandos de esta guía se ejecutan desde la raíz del repositorio.
El conjunto de evaluación principal es `evaluation/evaluation_v2_queries.json`
(160 consultas en inglés).

La línea base determinista y un piloto exploratorio de precisión end-to-end están
documentados en [`precision_baseline_v1.md`](precision_baseline_v1.md). El piloto
no tiene etiquetas adjudicadas ni sustituye el benchmark controlado de retrieval.
El diseño y estado del benchmark end-to-end se describe en
[`precision_benchmark_v1.md`](precision_benchmark_v1.md); sus anotaciones deben
revisarse antes de calcular métricas de calidad del producto.

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
- Índice BM25: `dataset_builder/indexer/artifacts/bm25_indexs_v2/`
- Corpus: `dataset_builder/corpus/`

Si es necesario reconstruir el índice BM25, desde la raíz:

```bash
PYTHONPATH=. dataset_builder/indexer/.venv/bin/python \
  dataset_builder/indexer/bm25_indexer.py \
  --corpus dataset_builder/corpus \
  --output dataset_builder/indexer/artifacts/bm25_indexs_v2 \
  --threads 4
```

Para probar la representación enriquecida sin modificar el índice histórico
`bm25_indexs_v2`, construí un índice independiente con el comando, hasta 256
caracteres de intención y el ejemplo dentro del campo buscable:

```bash
dataset_builder/indexer/.venv/bin/python \
  dataset_builder/indexer/bm25_indexer.py \
  --corpus dataset_builder/corpus \
  --output dataset_builder/indexer/artifacts/bm25_indexs_v4_enriched \
  --threads 4 \
  --representation enriched
```

La opción `--representation command` conserva el comportamiento anterior. Para
comparar la variante enriquecida sobre Debian 12, solo con development y
validation:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python evaluation/capture_precision_pilot.py \
  --queries evaluation/precision_benchmark_v1.json \
  --config evaluation/debian12_hybrid_enriched_precision.yaml \
  --split development \
  --split validation \
  --output evaluation/precision_benchmark_v1_debian12_hybrid_enriched_devval_iteration_2.json
```

Para ejecución offline, los modelos de Sentence Transformers y Cross-Encoder
deben estar ya disponibles en la caché local. `HF_HUB_OFFLINE=1` y
`TRANSFORMERS_OFFLINE=1` evitan descargas durante el benchmark.

## Baseline FAISS y rerankers

El benchmark oficial de retrieval usa `evaluation/evaluation_v2_queries.json`
(160 consultas). Antes de medir, validá y registrá el snapshot de corpus,
vectores, aliases e índice:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
dataset_builder/indexer/.venv/bin/python evaluation/create_snapshot_manifest.py
```

El manifiesto `evaluation/retrieval_snapshot_v2.json` incluye conteos, hashes
SHA-256 y comprobaciones de alineación de FAISS, BM25 y corpus. Ejecutá el
baseline FAISS sobre el mismo snapshot:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
dataset_builder/indexer/.venv/bin/python evaluation/evaluate_retrieval.py
```

La recuperación de candidatos usa Top-15 para que el baseline FAISS y BM25
compartan el mismo corte antes de calcular métricas hasta @10. El resultado
queda en `evaluation/retrieval_benchmark_v2.json`. Para generar el tablero HTML:

```bash
dataset_builder/indexer/.venv/bin/python evaluation/visualize_metrics.py
```

El archivo generado es `evaluation/retrieval_metrics_v2.html`.

## Comparación de retrieval para consultas por capacidad

Las consultas revisadas en `evaluation/capacity/plain_source.yaml` pueden
compararse entre FAISS, BM25 y búsqueda híbrida sobre las particiones
`development` y `test`, sin modificar el benchmark histórico ni su snapshot:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python evaluation/evaluate_capacity_retrieval.py
```

El informe queda en `evaluation/capacity/retrieval_comparison.json`. La
selección de `alpha` ocurre solo con `development`: se prueban 0.25, 0.5 y
0.75, maximizando en orden Hit@10, MRR y nDCG@10; los empates prefieren 0.5.
Después se compara el híbrido seleccionado con FAISS y BM25 en
`test`. La prueba no interviene en la selección. El JSON contiene métricas por
método, partición y categoría, resultados por consulta y hashes de procedencia.

La relevancia se puntúa a nivel de comando: todos los aliases indexados cuyo
comando coincide con el de la capacidad cuentan como relevantes. El índice
BM25 debe corresponder al corpus actual; por defecto se usa
`dataset_builder/indexer/artifacts/bm25_indexs_v2/`.
Ese índice conserva la representación legacy basada en nombres de comando; sus
resultados no deben generalizarse a los índices enriquecidos con intenciones o
frases de búsqueda.

### Comparación de MPNet y BGE-M3 en las capacidades revisadas

Para comparar los dos encoders con las mismas 122 consultas, etiquetas y
universo de aliases, ejecuta:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python evaluation/compare_capacity_embeddings.py
```

El informe queda en `evaluation/capacity/embedding_model_comparison.json`.
MPNet se filtra al universo de aliases compartidos antes de aplicar el Top-K;
la relevancia se limita también a ese universo. El informe separa
`development` y `test` y muestra métricas en inglés y español dentro de cada
partición. No se seleccionan parámetros usando `test`.

Los índices pueden tener dimensiones distintas: MPNet usa 768 y BGE-M3 1024;
cada consulta se codifica con el encoder correspondiente a su índice. Aunque
los aliases BGE coinciden con el subconjunto compartido, el artefacto de
vectores BGE no registra hashes de los textos fuente. Por eso no puede
confirmarse de forma independiente que sus embeddings se generaron a partir de
la revisión exacta del corpus actual; el JSON deja constancia de esta
limitación.

### Comparación BGE-M3 + reglas frente a BGE-M3 + Ollama

Este primer benchmark aísla la selección de candidatos: ambos métodos reciben
exactamente los mismos resultados de retrieval BGE-M3 para la consulta original,
con `top-k=15`, etiquetas y umbral fijos. `rules` elige el candidato elegible de
mayor rango; Ollama selecciona dentro del mismo conjunto. Se excluyen
deliberadamente la reformulación de consultas, el análisis de pasos y la
generación de explicaciones para atribuir diferencias solo al selector.

Con Ollama local disponible y `qwen2.5:1.5b` descargado:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python evaluation/compare_bge_ollama.py
```

Los informes quedan en `evaluation/capacity/bge_ollama_comparison.json` y
`evaluation/capacity/bge_ollama_comparison.md`. Hit@k/MRR y cobertura de
candidatos son compartidos por diseño; la métrica comparativa es la exactitud
del comando seleccionado, desglosada por partición e idioma. El informe también
registra latencia del selector, llamadas/fallbacks de Ollama, digest del modelo
y hashes del índice, aliases, corpus y consultas. El test se evalúa una vez con
configuración fija y no se usa para elegir parámetros. La selección se puntúa
por comando: no valida completamente los argumentos ni ejecuta propuestas. Se
guarda un checkpoint reanudable durante la ejecución y se elimina al completar
exitosamente.

El benchmark falla explícitamente si Ollama/modelo no están disponibles o el
índice no es BGE-M3 de 1024 dimensiones; no sustituye Ollama por el mock. La
procedencia BGE tiene una limitación: el archivo de vectores no registra hashes
de los textos fuente y no permite identificar de manera independiente su
revisión exacta.

#### Resultados de la ejecución actual

En las 122 consultas, la selección por reglas acertó 48 (39,3 %) y Ollama
acertó 67 (54,9 %), una diferencia de +15,6 puntos porcentuales. En el test
fijo de 36 consultas, las cifras fueron 10/36 (27,8 %) y 16/36 (44,4 %),
respectivamente. El retrieval compartido alcanzó Hit@15 de 84,4 % y MRR de
0,524; estos valores no miden una diferencia entre selectores. La latencia
media total reportada fue 7,22 s para reglas y 20,45 s para Ollama.

Se registró una llamada de fallback entre las 122 consultas: `networking.tar::3`
(development, inglés). La llamada no produjo una selección válida y se conservó
el primer candidato elegible (`mksquashfs`), que no coincide con la etiqueta
`tar`; este caso cuenta como selección incorrecta en el resultado de Ollama.
El informe no conserva la causa de ese error, por lo que no se atribuye a un
fallo específico del modelo o del servicio. Las métricas completas, los
desgloses por idioma/partición y la procedencia están en el informe JSON y
Markdown indicados arriba. Estos resultados describen esta muestra y no validan
la corrección integral de argumentos ni la ejecución segura de comandos.

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
  --bm25-index dataset_builder/indexer/artifacts/bm25_indexs_v2 \
  --faiss-index dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss \
  --aliases dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy \
  --initial-top-k 15 \
  --output evaluation/bm25_benchmark_v2_top15.json
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
  --bm25-index dataset_builder/indexer/artifacts/bm25_indexs_v2 \
  --faiss-index dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss \
  --aliases dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy \
  --initial-top-k 15 \
  --output evaluation/hybrid_benchmark_v2_top15.json
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

Todas las evaluaciones nuevas deben usar `evaluation/evaluation_v2_queries.json`
y un snapshot registrado. Los artefactos BM25 e híbrido existentes son
resultados históricos; para compararlos con el baseline actual, repetí sus
evaluaciones con el mismo snapshot de corpus, índices, aliases y relevancias.

## Benchmark de precisión end-to-end — Debian 12

La captura de precisión usa únicamente las 12 categorías de comandos del corpus
Debian 12. Docker y Git se mantienen en el corpus y en los índices, pero se
excluyen de esta ejecución y deben medirse como dominios separados. La
configuración dedicada valida además que el índice, aliases y corpus locales
existan:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python evaluation/capture_precision_pilot.py \
  --queries evaluation/precision_benchmark_v1.json \
  --config evaluation/debian12_precision.yaml \
  --output evaluation/precision_benchmark_v1_debian12_baseline.json
```

El capturador rechaza una configuración cuyo `allowed_categories` no coincida
con el dominio declarado por el benchmark. Las anotaciones siguen pendientes de
revisión humana, por lo que la captura no debe interpretarse como métricas de
calidad adjudicadas. El procedimiento completo está en
[`precision_benchmark_v1.md`](precision_benchmark_v1.md).

## Prueba de frases de búsqueda curadas para BM25

La prueba añade paráfrasis breves a siete aliases Debian, sin editar el corpus ni
añadirlas a las descripciones largas de los manuales. Las frases están en
[`bm25_search_phrases_debian12.json`](bm25_search_phrases_debian12.json).
Construí un índice independiente `bm25_indexs_v5_curated`, preservando las
versiones anteriores:

```bash
PYTHONPATH=. dataset_builder/indexer/.venv/bin/python \
  dataset_builder/indexer/bm25_indexer.py \
  --corpus dataset_builder/corpus \
  --output dataset_builder/indexer/artifacts/bm25_indexs_v5_curated \
  --threads 4 \
  --representation curated \
  --search-phrases evaluation/bm25_search_phrases_debian12.json
```

La configuración híbrida de prueba está en
[`debian12_hybrid_curated_precision.yaml`](debian12_hybrid_curated_precision.yaml).
Para reproducir solo development y validation, excluyendo holdout:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python evaluation/capture_precision_pilot.py \
  --queries evaluation/precision_benchmark_v1.json \
  --config evaluation/debian12_hybrid_curated_precision.yaml \
  --split development \
  --split validation \
  --output evaluation/precision_benchmark_v1_debian12_hybrid_curated_devval_iteration_1.json
```

La captura registra los hashes del índice y del archivo de frases, y no produce
métricas de precisión hasta que las etiquetas sean adjudicadas.

## Generar paráfrasis con Ollama

El generador lee YAML del corpus, consulta Ollama registro por registro y guarda
un JSON por archivo YAML, con el mismo nombre base y el formato
`alias: [frases]`. Por defecto usa el corpus reducido
`dataset_builder/corpus/ollama_corpus`, el modelo `qwen2.5:3b` y escribe los
resultados en `evaluation/ollama/corpus_paraphrases/` (por ejemplo,
`filesystem.yaml` produce `filesystem.json`). Primero conviene probar una
muestra pequeña:

```bash
python3 evaluation/ollama/generate_paraphrases_.py --limit 10
```

Para continuar luego con los registros restantes, sin repetir los aliases ya
guardados:

```bash
python3 evaluation/ollama/generate_paraphrases_.py --resume
```

Las respuestas inválidas y los timeouts de red se reintentan dos veces con una
pausa creciente; `--retries` permite cambiar el número de intentos y `--timeout`
el límite en segundos por petición. Cada alias se guarda inmediatamente en su
JSON, para que un corte no haga perder el último bloque. Si se agotan los
reintentos por timeout/conexión, el alias y el error se guardan en un archivo
lateral como `filesystem.errors.json` y el proceso se detiene. Al volver a
ejecutar con `--resume`, se saltean los aliases completados y se reintenta el
primero que quedó pendiente.

Se pueden indicar otro archivo o directorio YAML, modelo y salida con
`--corpus`, `--model` y `--output-dir`. La generación es local, pero las frases
producidas deben revisarse antes de usarlas en un índice o benchmark; los archivos
generados no modifican el corpus ni las frases curadas.

El modelo recibe solo el campo `intent`; `command` y `example` no se envían,
para que las variantes sean paráfrasis de la intención y no del ejemplo de CLI.
`--resume` continúa cada categoría desde su JSON y conserva los aliases ya
generados. Para regenerarlos desde cero, usá otro `--output-dir`.

## Índice BM25 enriquecido con paráfrasis de Ollama

Para medir el efecto de las paráfrasis sin cambiar la representación base,
construí dos índices independientes sobre el mismo corpus Debian 12:

- `enriched`: comando + intención (hasta 256 caracteres) + ejemplo.
- `enriched_paraphrases`: el mismo contenido, más las paráfrasis de cada alias.

Prepará el corpus filtrado desde la raíz del repositorio:

```bash
mkdir -p /tmp/it-shell-corpus-debian12
find dataset_builder/corpus -maxdepth 1 -type f -name '*.yaml' \
  ! -name 'docker.yaml' ! -name 'git.yaml' \
  -exec cp {} /tmp/it-shell-corpus-debian12/ \;
```

El directorio resultante contiene los 12 YAML Debian, sin Docker ni Git. El
corpus y los dos archivos de paráfrasis deben estar disponibles en el cluster:
[`filesystem.json`](ollama/corpus_paraphrases_intent_only/filesystem.json) y
[`networking.json`](ollama/corpus_paraphrases_intent_only/networking.json).
Las listas vacías se aceptan como abstenciones; las frases repetidas dentro de
un alias se deduplican conservando el orden. Para las categorías incluidas en
los archivos JSON, el indexador comprueba que todos los aliases del corpus
tengan una entrada y que no existan aliases desconocidos.

Desde la raíz del repositorio, construí los índices en directorios nuevos:

```bash
PYTHONPATH=. python dataset_builder/indexer/bm25_indexer.py \
  --corpus /tmp/it-shell-corpus-debian12 \
  --output /ruta/indices/bm25_debian12_enriched_base \
  --threads 4 \
  --representation enriched

PYTHONPATH=. python dataset_builder/indexer/bm25_indexer.py \
  --corpus /tmp/it-shell-corpus-debian12 \
  --output /ruta/indices/bm25_debian12_enriched_paraphrases \
  --threads 4 \
  --representation enriched_paraphrases \
  --search-phrases \
    evaluation/ollama/corpus_paraphrases_intent_only/filesystem.json \
    evaluation/ollama/corpus_paraphrases_intent_only/networking.json
```

No se regeneran embeddings ni FAISS. Conservá las rutas de salida separadas y
registrá los hashes del corpus y de ambos JSON junto con los resultados para
que la comparación sea reproducible.

## Comparar índices BM25 con y sin paráfrasis

Con los dos índices construidos, ejecutá la comparación pareada para las 16
consultas de `filesystem` y las 16 de `networking`. Reporta BM25 puro y, como
resultado secundario, BM25+FAISS (`alpha=0.5`), con métricas por categoría,
deltas por consulta y hashes de los índices/corpus:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  dataset_builder/indexer/.venv/bin/python evaluation/compare_bm25_paraphrases.py \
  --corpus /tmp/it-shell-corpus-debian12 \
  --base-index dataset_builder/indexer/artifacts/bm25_debian12_enriched_base_lucene9 \
  --paraphrase-index dataset_builder/indexer/artifacts/bm25_debian12_enriched_paraphrases_lucene9 \
  --output evaluation/bm25_paraphrase_comparison_debian12.json
```

El benchmark no modifica los índices ni usa el holdout de precisión end-to-end.
Con 16 consultas por categoría, los resultados son exploratorios y no bastan
por sí solos para afirmar una mejora general.

## Pruebas funcionales de índices

Con los artefactos requeridos ya creados, se pueden ejecutar las pruebas de
BM25 y del buscador híbrido:

```bash
dataset_builder/indexer/.venv/bin/python -m pytest tests/test_bm25.py -v
dataset_builder/indexer/.venv/bin/python -m pytest tests/test_hybrid.py -v
```
