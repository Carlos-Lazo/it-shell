# dataset_builder/embedder/hybrid_search.py
import json
import numpy as np
import faiss
from pyserini.search import SimpleSearcher
from pathlib import Path


BM25_COMMANDS_PATH = "../artifacts/bm25_indexs"
MODEL_MP = "sentence-transformers/all-mpnet-base-v2"
INDEX_PATH = "../indexes/faiss/mpnet.index"
LANGUAGE = "en"
TOP_K_MAX = 15
# Cargar embedder (MPNet o e5-large)
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(MODEL_MP)  

# Cargar índice FAISS (pre‑generado)
faiss_index_path = Path(INDEX_PATH)
faiss_index = faiss.read_index(str(faiss_index_path))

# Cargar BM25 buscador
bm25_searcher = SimpleSearcher(str(Path(BM25_COMMANDS_PATH)))
bm25_searcher.set_language(LANGUAGE)          # opcional
bm25_searcher.set_hits(TOP_K_MAX)                # número máximo de hits que devolvemos

def hybrid_query(query: str, top_k: int = TOP_K_MAX, alpha: float = 0.5):
    """
    - `alpha` controla la mezcla: 0 → solo BM25, 1 → solo dense.
    - Devuelve una lista de tuplas (doc_id, score) ordenada por puntuación combinada.
    """
    # 1️⃣ Dense scores
    q_vec = embedder.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(q_vec, top_k)      # D: distances, I: ids
    dense_scores = {int(idx): float(score) for idx, score in zip(I[0], D[0])}

    # 2️⃣ BM25 scores
    bm25_hits = bm25_searcher.search(query)
    bm25_scores = {int(hit.docid): float(hit.score) for hit in bm25_hits}

    # 3️⃣ Normalizar ambos rangos a [0,1]
    if dense_scores:
        max_dense = max(dense_scores.values())
        min_dense = min(dense_scores.values())
    else:
        max_dense = min_dense = 0.0
    if bm25_scores:
        max_bm25 = max(bm25_scores.values())
        min_bm25 = min(bm25_scores.values())
    else:
        max_bm25 = min_bm25 = 0.0

    def norm(val, minv, maxv):
        return (val - minv) / (maxv - minv + 1e-12)

    # 4️⃣ Mezcla lineal
    fused = {}
    for doc_id in set(dense_scores) | set(bm25_scores):
        d = norm(dense_scores.get(doc_id, min_dense), min_dense, max_dense)
        b = norm(bm25_scores.get(doc_id, min_bm25), min_bm25, max_bm25)
        fused[doc_id] = alpha * d + (1 - alpha) * b

    # 5️⃣ Ordenar
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked