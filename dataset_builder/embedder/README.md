# Embedder

Genera representaciones semánticas con `sentence-transformers/all-mpnet-base-v2`.
El modelo produce vectores normalizados de 768 dimensiones.

## Preparación

```bash
./setup_env.sh
source .venv/bin/activate
```

## Generación

Los embeddings se calculan únicamente desde el campo `intent` de cada documento YAML:

```bash
python generate_embeddings.py
```

La salida `artifacts/vectors.npz` contiene dos arreglos alineados:

- `vectors`: matriz `float32` normalizada (`N × 768`).
- `ids`: aliases con el formato `categoria:id`, por ejemplo `filesystem:12`.

El registro lógico de índice `i` es `[vectors[i], ids[i]]`.

## Índice vectorial

La construcción y consulta del índice FAISS se encuentran en `../indexer/`.