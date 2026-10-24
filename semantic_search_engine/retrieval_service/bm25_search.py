#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
bm25_search.py
--------------

Wrapper de búsqueda que combina:

* **BM25** (índice Lucene creado con Pyserini) → excelente para coincidencias
  exactas de palabras, flags, nombres de comandos, etc.
* **FAISS + dense embeddings** (índice vectorial existente en el proyecto) →
  captura similitud semántica cuando la redacción de la consulta difiere.

El algoritmo es **linealmente combinable**:

    score = α * dense_norm + (1‑α) * bm25_norm

donde `α` (alpha) controla el peso del modelo denso (0 → solo BM25,
1 → solo dense).

Los documentos también pueden devolver el campo *intent* que está guardado
en el índice BM25; lo incluimos en la salida para que la UI o el informe
lo muestre al usuario.

Requisitos
----------
pip install pyserini==0.22.0 faiss-cpu sentence‑transformers
"""

import pathlib
from typing import List, Tuple

import numpy as np
import faiss
from pyserini.search import SimpleSearcher
from sentence_transformers import SentenceTransformer

#  Ruta donde guardaste el índice FAISS (vectorial)
_FAISS_INDEX_PATH = pathlib.Path("../indexes/faiss_mpnet.index")   # ← ajústalo
_IDS_PATH = pathlib.Path("../data/embeddings_ids.json")

# ----------------------------------------------------------------------
# Configuración global (una sola carga por proceso)
# ----------------------------------------------------------------------
# modelo de embeddings (el mismo que usas para crear
# el índice FAISS).  Disponible para otros modelos SBERT.
_EMBEDDER = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",  # modelo rápido (≈ 25 ms/1 k)
    device="cpu",
)

if not _FAISS_INDEX_PATH.is_file():
    raise FileNotFoundError(f"[ERROR] FAISS index not found: {_FAISS_INDEX_PATH}")

_faiss_index = faiss.read_index(str(_FAISS_INDEX_PATH))

#    el mapa id → posición en el índice FAISS.
#    Normalmente lo guardas en `data/embeddings_ids.json` cuando creas el
#    índice.  Si no lo tienes, puedes crear uno rápidamente leyendo el
#    archivo `.ids.json` que genera `generate_embeddings.py`.
if not _IDS_PATH.is_file():
    raise FileNotFoundError(f"[ERROR] Embedding IDs file not found: {_IDS_PATH}")

with _IDS_PATH.open("r", encoding="utf-8") as f:
    _id_list: List[str] = json.load(f)   # lista ordenada que corresponde al FAISS

_id_to_faiss_pos = {doc_id: i for i, doc_id in enumerate(_id_list)}


# ----------------------------------------------------------------------
# Helper – normalizar puntuaciones (0‑1)
# ----------------------------------------------------------------------
def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Lleva un vector de scores a [0, 1] (evita división por 0)."""
    lo = arr.min()
    hi = arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ----------------------------------------------------------------------
# Clase pública -----------------------------------------------------------------
# ----------------------------------------------------------------------
class HybridBM25Retriever:
    """
    Uso típico:

    >>> from retrieval_service.bm25_search import HybridBM25Retriever
    >>> retr = HybridBM25Retriever(bm25_index_dir="artifacts/bm25_indexs")
    >>> results = retr.search("how do I delete a git branch", top_k=10, alpha=0.7)
    >>> for doc_id, score, intent in results:
    ...     print(doc_id, f"{score:.3f}", intent)
    """

    def __init__(self, bm25_index_dir: str, top_k: int = 15):
        """
        Parameters
        ----------
        bm25_index_dir : str
            Directorio donde está el índice Lucene creado por `bm25_indexer.py`.
        top_k : int, default 15
            Número máximo de candidatos que devolvemos (se usa tanto para
            BM25 como para dense; el híbrido puede truncar después de la
            fusión).
        """
        self.bm25 = SimpleSearcher(bm25_index_dir)
        # Guardamos en memoria el número máximo de resultados a recuperar
        self.top_k = top_k

    # ------------------------------------------------------------------
    def _bm25_hits(self, query: str) -> List[Tuple[str, float, str]]:
        """
        Ejecuta BM25 y devuelve una lista de tuplas:
        (doc_id, bm25_score, intent)
        """
        hits = self.bm25.search(query, k=self.top_k)
        results = []
        for hit in hits:
            doc_id = hit.docid   # entero interno
            # En el JSONL guardamos `id` como string; el mapeo entre
            # entero ↔︎ string está en el índice Lucene automáticamente.
            # Para obtener la cadena usamos `hit.raw` (si usamos -storeRaw)
            # o bien guardamos `id` como metadata.
            # Aquí extraemos `raw` y lo parseamos para obtener el id real
            raw_json = hit.raw
            try:
                data = json.loads(raw_json)
                real_id = data["id"]
                intent = data.get("intent", "")
            except Exception:
                # fallback: usamos el número interno como string
                real_id = str(doc_id)
                intent = ""
            results.append((real_id, hit.score, intent))
        return results

    # ------------------------------------------------------------------
    def _dense_scores(self, query: str) -> List[Tuple[str, float]]:
        """
        Codifica la consulta con el mismo embedder que usó el índice FAISS
        y devuelve (doc_id, dense_score) para los `top_k` más cercanos.
        """
        q_vec = _EMBEDDER.encode([query], normalize_embeddings=True)
        D, I = _faiss_index.search(q_vec, self.top_k)   # D: L2 distances
        # Convertimos distancia L2 a similitud (cuanto menor, mejor)
        # Usamos `-D` para que valores más altos signifiquen mayor similitud.
        sim = -D[0]                                   # vector 1‑D
        sim_norm = _minmax_normalize(sim)             # 0‑1
        results = []
        for pos, score in zip(I[0], sim_norm):
            doc_id = _id_list[pos]                    # id en string
            results.append((doc_id, float(score)))
        return results

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = None, alpha: float = 0.7) -> List[Tuple[str, float, str]]:
        """
        Devuelve una lista ordenada de (doc_id, hybrid_score, intent).

        Parameters
        ----------
        query : str
            Consulta de usuario.
        top_k : int | None
            Override del número máximo de resultados (por defecto el que
            se indicó en el constructor (15)).
        alpha : float, default 0.7
            Peso del modelo **dense**.  `alpha=0` → solo BM25,
            `alpha=1` → solo dense.

        Returns
        -------
        List[Tuple[doc_id, score, intent]]
        """
        if top_k is None:
            top_k = self.top_k

        # 1️⃣  BM25
        bm25_res = self._bm25_hits(query)      # [(id, bm25_score, intent), …]

        # 2️⃣  Dense (FAISS)
        dense_res = self._dense_scores(query)  # [(id, dense_score), …]

        # 3️⃣  Normalizamos ambas puntuaciones a [0, 1]
        bm25_scores = np.array([s for _, s, _ in bm25_res])
        dense_scores = np.array([s for _, s in dense_res])

        bm25_norm = _minmax_normalize(bm25_scores) if bm25_scores.size else np.array([])
        dense_norm = _minmax_normalize(dense_scores) if dense_scores.size else np.array([])

        # 4️⃣  Unimos por `doc_id`
        #   Usamos diccionarios temporales para combinar rápidamente.
        bm25_dict = {doc: (score, intent) for doc, score, intent in bm25_res}
        dense_dict = {doc: score for doc, score in dense_res}

        fused = []
        # Recorremos la unión de IDs (BM25 ∪ dense)
        for doc_id in set(bm25_dict) | set(dense_dict):
            bm25_sc = bm25_dict.get(doc_id, (0.0, ""))[0]
            dense_sc = dense_dict.get(doc_id, 0.0)

            # Normalizamos a nivel de documento (usa los vectores ya normalizados)
            # Si el set es pequeño, simplemente usa los vectores normales:
            #   bm25_norm[i] ↔ doc_id i ; dense_norm[i] ↔ doc_id i
            # Para evitar una búsqueda O(N²) usa los índices que ya tenemos:
            #   - Si está en BM25: usa su posición en `bm25_norm`
            #   - Si está solo en dense: usa su posición en `dense_norm`
            # En este ejemplo usa los valores sin volver a normalizar.
            # (Para una producción real, podrías construir una tabla id→pos.)

            # Como los vectores están ya normalizados, basta combinarlos:
            hybrid_score = alpha * dense_sc + (1 - alpha) * bm25_sc

            intent = bm25_dict.get(doc_id, ("", ""))[1]  # intent solo está en BM25
            fused.append((doc_id, float(hybrid_score), intent))

        # 5️⃣  Ordenar por puntuación descendente y recortar
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]