# it-shell

`it-shell` es un asistente de búsqueda y planificación de comandos CLI. Recibe una
consulta en lenguaje natural, recupera comandos relevantes del corpus, propone
alternativas y secuencias, y señala parámetros faltantes y posibles riesgos.

**El sistema no ejecuta comandos en la shell.** Las propuestas son informativas y
deben revisarse antes de utilizarlas.

El proyecto tiene dos componentes principales:

## `dataset_builder`: construcción de datos e índices

Pipeline offline para preparar la base de conocimiento:

1. **`webcrawling/`** obtiene información de fuentes como Debian, Linux, Git y Docker.
2. **`classifier/`** contiene un clasificador léxico TF-IDF en Go y herramientas para
   organizar documentos por categoría.
3. **`corpus/`** contiene los documentos YAML con categoría, identificador, intención,
   comando y ejemplo. El embedder utiliza principalmente el campo `intent`.
4. **`embedder/`** genera vectores normalizados con
   `sentence-transformers/all-mpnet-base-v2` y aliases `categoria:id`.
5. **`indexer/`** crea el índice vectorial FAISS y proporciona utilidades para
   consultarlo. También contiene el indexador BM25 experimental.

Cada etapa tiene documentación propia en [`dataset_builder/README.md`](dataset_builder/README.md),
[`embedder/README.md`](dataset_builder/embedder/README.md) y
[`indexer/README.md`](dataset_builder/indexer/README.md).

## `semantic_search_engine`: búsqueda y propuestas

Motor Python que:

- analiza la consulta y descompone pasos explícitos mediante el planner;
- busca candidatos con FAISS y, opcionalmente, permite configurar un reranker;
- resuelve aliases contra el corpus, aplica umbrales de confianza y valida la propuesta;
- detecta ciertos comandos peligrosos o datos faltantes, con avisos y recomendaciones;
- genera una explicación local determinista y puede conectarse a un proveedor de
  inferencia opcional mediante la interfaz `LLMClient`;
- presenta resultados en texto para terminal, terminal ANSI, Markdown o JSON.

La configuración predeterminada está en
[`semantic_search_engine/config/default.yaml`](semantic_search_engine/config/default.yaml).
El detalle de arquitectura está en
[`semantic_search_engine/README.md`](semantic_search_engine/README.md).

## Inicio rápido: instalar y ejecutar

Requisitos: Python 3.10 o superior. Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r semantic_search_engine/requirements.txt
```

La ejecución requiere que existan el corpus YAML y los artefactos indicados en la
configuración: el índice FAISS, `aliases.npy` y el modelo
`sentence-transformers/all-mpnet-base-v2` (se descarga al primer uso si no está en
caché). Los corpus YAML y algunos artefactos generados no se distribuyen en Git.
Si aún no están disponibles en tu entorno, consulta la sección siguiente antes de
ejecutar la CLI.

Con los datos preparados, ejecuta una consulta:

```bash
PYTHONPATH=. python semantic_search_engine/cli.py "List files in a directory"
```

Formatos disponibles:

```bash
PYTHONPATH=. python semantic_search_engine/cli.py "find large log files" --format terminal
PYTHONPATH=. python semantic_search_engine/cli.py "find large log files" --format markdown
PYTHONPATH=. python semantic_search_engine/cli.py "find large log files" --format json
```

`terminal` incluye colores ANSI para los avisos; `cli` es texto plano. También se
puede indicar la cantidad de candidatos con `--top-k 10` y una configuración
alternativa con `--config ruta/al/config.yaml`.

### Preparar embeddings e índice FAISS

Este paso requiere que `dataset_builder/corpus/` ya contenga los YAML. Desde la raíz,
con las dependencias instaladas en `.venv`:

```bash
PYTHONPATH=. .venv/bin/python dataset_builder/embedder/generate_embeddings.py \
  --corpus dataset_builder/corpus \
  --output dataset_builder/embedder/artifacts

PYTHONPATH=. .venv/bin/python dataset_builder/indexer/build_index.py \
  --vectors dataset_builder/embedder/artifacts/mpnet_base_v2_vectors.npz \
  --output dataset_builder/indexer/artifacts/all-mpnet-base-v2
```

Para obtener o generar el corpus (incluyendo la descarga/procesamiento de fuentes),
consulta [`dataset_builder/README.md`](dataset_builder/README.md) y los README de
[`webcrawling/`](dataset_builder/webcrawling/README.md) y
[`classifier/`](dataset_builder/classifier/README.md). La generación de embeddings
puede requerir descargar el modelo de Hugging Face la primera vez.

## Pruebas

Con pytest instalado:

```bash
PYTHONPATH=. python -m pytest -q
```

La suite cubre planificación, recuperación, propuestas, reglas de seguridad,
formatos y explicación. No ejecuta los comandos propuestos.
