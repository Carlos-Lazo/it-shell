# Indexer

Construye y consulta índices vectoriales FAISS a partir de embeddings ya generados.

## Build

```bash
./setup_env.sh
source .venv/bin/activate
python build_index.py
```

El índice actual está en `artifacts/all-mpnet-base-v2/` y utiliza `IndexFlatIP`.
Como los vectores están normalizados, el producto interno equivale a similitud coseno.

## Consulta

```bash
python search_index.py "set http proxy" --top-k 5
```

Los resultados relacionan cada posición del índice con `aliases.npy`, usando aliases
`categoria:id`, por ejemplo `filesystem:12`.

## bm25
Construcción de índices bm25
```bash
python dataset_builder/indexer/bm25_indexer.py \
    --corpus ../corpus \
    --output artifacts/bm25_indexs_v2 \
    --threads 4
```