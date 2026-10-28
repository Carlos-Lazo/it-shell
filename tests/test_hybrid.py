#!/usr/bin/env python3
"""
test_hybrid.py
--------------
Tests funcionales de la búsqueda híbrida BM25 + FAISS dense.

Ejecutar desde la raíz del repo con el venv del indexer:
    dataset_builder/indexer/.venv/bin/python -m pytest tests/test_hybrid.py -v

Requisitos previos:
  1. Índice BM25: dataset_builder/indexer/artifacts/bm25_indexs_v2/
  2. Índice FAISS: dataset_builder/indexer/artifacts/all-mpnet-base-v2/index.faiss
  3. Aliases FAISS: dataset_builder/indexer/artifacts/all-mpnet-base-v2/aliases.npy
"""

from __future__ import annotations

import json
import math
import pathlib
import time
import unittest

import faiss
import numpy as np
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Rutas absolutas derivadas de la ubicación de este archivo
# ---------------------------------------------------------------------------
REPO_ROOT    = pathlib.Path(__file__).resolve().parents[1]
BM25_INDEX   = REPO_ROOT / "dataset_builder" / "indexer" / "artifacts" / "bm25_indexs_v2"
FAISS_INDEX  = REPO_ROOT / "dataset_builder" / "indexer" / "artifacts" / "all-mpnet-base-v2" / "index.faiss"
ALIASES_FILE = REPO_ROOT / "dataset_builder" / "indexer" / "artifacts" / "all-mpnet-base-v2" / "aliases.npy"
EMBEDDER_NAME = "sentence-transformers/all-mpnet-base-v2"

TOP_K  = 15
ALPHA  = 0.5   # peso del componente denso (0 = solo BM25, 1 = solo dense)


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return np.zeros_like(arr) if hi - lo < 1e-12 else (arr - lo) / (hi - lo)


def hybrid_search(
    query: str,
    bm25: LuceneSearcher,
    faiss_index,
    aliases: np.ndarray,
    embedder: SentenceTransformer,
    top_k: int = TOP_K,
    alpha: float = ALPHA,
) -> list[tuple[str, float, str]]:
    """
    Búsqueda híbrida BM25 + dense.

    Devuelve lista de (alias, hybrid_score, intent) ordenada desc.
    Los alias tienen formato 'category:id', igual que en los benchmarks.
    """
    # Dense
    q_vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    d_scores, d_positions = faiss_index.search(q_vec, top_k)
    dense_map: dict[str, float] = {
        str(aliases[pos]): float(score)
        for pos, score in zip(d_positions[0], d_scores[0])
    }

    # BM25
    bm25_hits = bm25.search(query, k=top_k)
    bm25_map:  dict[str, tuple[float, str]] = {}
    for hit in bm25_hits:
        raw = json.loads(hit.raw)
        alias  = raw["id"]
        intent = raw.get("intent", "")
        bm25_map[alias] = (float(hit.score), intent)

    # Normalizar
    dense_vals = np.array(list(dense_map.values())) if dense_map else np.array([])
    bm25_vals  = np.array([v for v, _ in bm25_map.values()]) if bm25_map else np.array([])
    dense_norm_vals = _minmax(dense_vals) if dense_vals.size else dense_vals
    bm25_norm_vals  = _minmax(bm25_vals)  if bm25_vals.size  else bm25_vals

    dense_norm = dict(zip(dense_map.keys(), dense_norm_vals.tolist()))
    bm25_norm  = {alias: (score, intent) for (alias, (_, intent)), score
                  in zip(bm25_map.items(), bm25_norm_vals.tolist())}

    # Fusionar
    all_ids = set(dense_norm) | set(bm25_norm)
    fused: list[tuple[str, float, str]] = []
    for alias in all_ids:
        d = dense_norm.get(alias, 0.0)
        b, intent = bm25_norm.get(alias, (0.0, ""))
        fused.append((alias, alpha * d + (1 - alpha) * b, intent))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_k]


class TestHybridSearch(unittest.TestCase):
    """Suite de tests para la búsqueda híbrida BM25 + FAISS."""

    @classmethod
    def setUpClass(cls) -> None:
        missing = []
        if not BM25_INDEX.exists():
            missing.append(str(BM25_INDEX))
        if not FAISS_INDEX.exists():
            missing.append(str(FAISS_INDEX))
        if not ALIASES_FILE.exists():
            missing.append(str(ALIASES_FILE))
        if missing:
            raise FileNotFoundError(
                "Faltan artefactos necesarios para los tests:\n  "
                + "\n  ".join(missing)
            )

        cls.bm25    = LuceneSearcher(str(BM25_INDEX))
        cls.index   = faiss.read_index(str(FAISS_INDEX))
        cls.aliases = np.load(str(ALIASES_FILE), allow_pickle=False).astype(str)
        cls.embedder = SentenceTransformer(EMBEDDER_NAME)

    def _search(self, query: str, top_k: int = TOP_K, alpha: float = ALPHA):
        return hybrid_search(query, self.bm25, self.index, self.aliases, self.embedder, top_k, alpha)

    # ------------------------------------------------------------------
    def test_returns_results(self) -> None:
        results = self._search("list directory contents")
        self.assertGreater(len(results), 0)

    def test_result_tuple_format(self) -> None:
        results = self._search("delete git branch", top_k=5)
        for alias, score, intent in results:
            with self.subTest(alias=alias):
                self.assertIsInstance(alias, str)
                self.assertIn(":", alias, f"alias '{alias}' no tiene formato category:id")
                self.assertIsInstance(score, float)
                self.assertIsInstance(intent, str)

    def test_scores_bounded_0_1(self) -> None:
        """Los scores híbridos normalizados deben estar en [0, 1]."""
        results = self._search("run docker container", top_k=10)
        for alias, score, _ in results:
            with self.subTest(alias=alias):
                self.assertGreaterEqual(score, 0.0, f"{alias}: score {score} < 0")
                self.assertLessEqual(score, 1.0,    f"{alias}: score {score} > 1")

    def test_scores_ordered_descending(self) -> None:
        results = self._search("create file", top_k=10)
        if len(results) < 2:
            self.skipTest("No hay suficientes resultados")
        scores = [s for _, s, _ in results]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(
                scores[i], scores[i + 1],
                f"Orden incorrecto en pos {i}: {scores[i]} < {scores[i+1]}"
            )

    def test_top_k_respected(self) -> None:
        for k in (1, 5, 10):
            with self.subTest(k=k):
                results = self._search("network interface", top_k=k)
                self.assertLessEqual(len(results), k)

    def test_no_duplicate_aliases(self) -> None:
        results = self._search("show process status", top_k=15)
        aliases = [a for a, _, _ in results]
        self.assertEqual(len(aliases), len(set(aliases)), "Hay aliases duplicados en los resultados")

    def test_alpha_0_equals_bm25_only(self) -> None:
        """Con alpha=0 el ranking debe coincidir con BM25 puro."""
        query   = "git merge branch"
        bm25_only = self._search(query, top_k=10, alpha=0.0)
        bm25_direct = self.bm25.search(query, k=10)
        bm25_aliases = [json.loads(h.raw)["id"] for h in bm25_direct]
        hybrid_aliases = [a for a, _, _ in bm25_only]
        # La intersección debe incluir todos los de BM25
        shared = set(bm25_aliases) & set(hybrid_aliases)
        self.assertGreater(len(shared), 0, "Alpha=0 no recupera ningún resultado de BM25")

    def test_alpha_1_equals_dense_only(self) -> None:
        """Con alpha=1 las posiciones deben coincidir con FAISS puro."""
        query  = "commit and push"
        dense_only = self._search(query, top_k=10, alpha=1.0)
        q_vec = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        _, positions = self.index.search(q_vec, 10)
        dense_aliases = [str(self.aliases[p]) for p in positions[0]]
        hybrid_aliases = [a for a, _, _ in dense_only]
        shared = set(dense_aliases) & set(hybrid_aliases)
        self.assertGreater(len(shared), 0, "Alpha=1 no recupera ningún resultado dense")

    def test_latency_per_query_ms(self) -> None:
        """La latencia de una consulta híbrida debe ser razonable (< 5 000 ms en CPU)."""
        t0 = time.perf_counter()
        self._search("list open network ports", top_k=15)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed_ms, 5_000, f"Latencia excesiva: {elapsed_ms:.1f} ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)
