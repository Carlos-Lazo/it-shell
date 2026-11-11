# Construcción del corpus y de los índices

`dataset_builder` reúne las herramientas offline que preparan la base de
conocimiento de IT Shell: recopilan documentación, la transforman en registros
YAML y generan los artefactos que consume el motor de búsqueda.

> **Importante:** el corpus, los embeddings y los índices deben pertenecer a la
> misma revisión de datos. No mezcles un índice con aliases o corpus de otra
> generación.

## Flujo de datos

1. **Extracción:** scripts de `webcrawling/` obtienen datos de Debian 12, Linux,
   Git y Docker.
2. **Clasificación y normalización:** herramientas de `classifier/` producen
   documentos YAML por categoría, con campos como `id`, `intent`, `command` y
   `example`.
3. **Vectorización:** `embedder/` codifica el texto del corpus y guarda vectores
   junto con sus aliases `categoria:id`.
4. **Indexación:** `indexer/` construye un índice FAISS alineado con los
   vectores y aliases. También contiene un indexador BM25 experimental.

La estructura principal es:

```text
dataset_builder/
├── webcrawling/   # extracción desde fuentes documentales
├── classifier/    # clasificación y generación de YAML
├── corpus/        # documentos YAML usados por el motor
├── embedder/      # embeddings MPNet y BGE-M3
└── indexer/       # índices FAISS y BM25
```

Los pasos para regenerar fuentes están en
[`corpus/REPRODUCING.md`](corpus/REPRODUCING.md). Cada herramienta tiene
documentación adicional en [`classifier/README.md`](classifier/README.md),
[`embedder/README.md`](embedder/README.md) e
[`indexer/README.md`](indexer/README.md).

## Reproducir el corpus

Los crawlers tienen requisitos y directorios de salida propios. En particular,
la extracción Debian debe realizarse en un entorno Debian 12 estable. La guía
de reproducción describe cómo guardar las salidas en directorios temporales y
transformarlas a YAML sin sobrescribir el corpus existente.

No ejecutes crawlers o generadores sobre datos de producción sin revisar antes
las rutas de salida. Para consultar el proceso completo, sigue
[`corpus/REPRODUCING.md`](corpus/REPRODUCING.md).

## Embeddings e índices

Las herramientas se invocan desde la raíz del repositorio con las dependencias
correspondientes instaladas. Los modelos pueden requerir una descarga inicial;
para ejecuciones offline deben estar previamente disponibles en caché.

### MPNet: configuración predeterminada del motor

El pipeline predeterminado usa
`sentence-transformers/all-mpnet-base-v2`, vectores normalizados de 768
dimensiones y FAISS `IndexFlatIP`:

```bash
PYTHONPATH=. .venv/bin/python dataset_builder/embedder/generate_embeddings.py \
  --corpus dataset_builder/corpus \
  --output dataset_builder/embedder/artifacts

PYTHONPATH=. .venv/bin/python dataset_builder/indexer/build_index.py \
  --vectors dataset_builder/embedder/artifacts/mpnet_base_v2_vectors.npz \
  --output dataset_builder/indexer/artifacts/all-mpnet-base-v2
```

Como los vectores están normalizados, el producto interno equivale a similitud
coseno. El archivo de vectores contiene la matriz y los IDs alineados; el índice
FAISS y `aliases.npy` deben conservar ese mismo orden.

### BGE-M3: variante experimental

BGE-M3 produce vectores normalizados de 1.024 dimensiones. Para generarlos y
construir un índice separado:

```bash
PYTHONPATH=. .venv/bin/python \
  dataset_builder/embedder/bge-m3/generate_embeddings.py \
  --corpus dataset_builder/corpus \
  --output dataset_builder/embedder/artifacts

PYTHONPATH=. .venv/bin/python \
  dataset_builder/indexer/artifacts/bge-m3/build_index.py \
  --vectors dataset_builder/embedder/artifacts/bge_m3_vectors.npz \
  --output dataset_builder/indexer/artifacts/bge-m3
```

Los scripts admiten opciones de dispositivo y paralelismo para la generación
de embeddings y la construcción FAISS. Consulta `--help` antes de ajustar esos
parámetros. No intercambies el encoder, vectores, índice o aliases entre MPNet
y BGE-M3.

La comparación de capacidades BGE-M3 registró que los aliases se alinean con
el subconjunto actual del corpus, pero el artefacto de vectores no incluye
hashes de los textos fuente. Por ello, no se puede certificar
independientemente qué revisión exacta de texto generó esos embeddings. Ver
el [informe de evaluación](../evaluation/capacity/bge_ollama_comparison.md).

### BM25: índice experimental

El indexador BM25 permite representar los documentos a partir de comandos,
contenido enriquecido o frases curadas. Por ejemplo, para construir el índice
legacy basado en nombres de comando:

```bash
PYTHONPATH=. .venv/bin/python dataset_builder/indexer/bm25_indexer.py \
  --corpus dataset_builder/corpus \
  --output dataset_builder/indexer/artifacts/bm25_indexs_v2 \
  --threads 4 \
  --representation command
```

Las opciones `enriched`, `curated` y `enriched_paraphrases` tienen requisitos
adicionales; las dos últimas requieren archivos JSON de frases de búsqueda.
BM25 y la búsqueda híbrida son variantes experimentales y no reemplazan la
configuración densa predeterminada. Consulta
[`indexer/README.md`](indexer/README.md) y el
[protocolo de evaluación](../evaluation/README.md).

## Validación de artefactos

Antes de usar un índice, verifica que:

- el número de vectores coincida con el de aliases;
- los aliases sean únicos y resuelvan contra el corpus;
- la dimensión del índice corresponda al encoder seleccionado;
- las consultas de evaluación utilicen el mismo corpus e índice que se
  registran en el manifiesto.

Para validar y guardar la procedencia del snapshot de retrieval, sigue las
instrucciones de [`evaluation/README.md`](../evaluation/README.md), que incluyen
`evaluation/create_snapshot_manifest.py`.
