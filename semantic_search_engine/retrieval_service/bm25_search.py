"""Hybrid BM25 and FAISS retriever used by the search service and benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
from pyserini.search.lucene import LuceneSearcher

from semantic_search_engine.retrieval_service.retriever import Encoder


class SearchHit(Protocol):
    raw: str
    score: float


class BM25Searcher(Protocol):
    def search(self, query: str, k: int) -> list[SearchHit]: ...


class HybridBM25Retriever:
    """Fuse Pyserini BM25 and FAISS dense rankings over shared document aliases."""

    def __init__(
        self,
        bm25_searcher: BM25Searcher,
        faiss_index: faiss.Index,
        aliases: np.ndarray,
        encoder: Encoder,
        top_k: int = 15,
        alpha: float = 0.5,
        allowed_categories: tuple[str, ...] | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k debe ser mayor que cero.")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha debe estar entre 0 y 1.")
        if faiss_index.ntotal != len(aliases):
            raise ValueError("El índice FAISS y sus aliases no están alineados.")

        self.bm25 = bm25_searcher
        self.faiss_index = faiss_index
        self.aliases = np.asarray(aliases).astype(str)
        self.encoder = encoder
        self.top_k = top_k
        self.alpha = alpha
        self.allowed_categories = (
            frozenset(allowed_categories) if allowed_categories is not None else None
        )
        self.document_count = max(faiss_index.ntotal, len(self.aliases))
        if self.allowed_categories is not None:
            bm25_document_count = getattr(bm25_searcher, "num_docs", None)
            if not isinstance(bm25_document_count, int) or bm25_document_count <= 0:
                raise ValueError(
                    "El buscador BM25 debe exponer num_docs para filtrar categorías"
                )
            self.document_count = max(self.document_count, bm25_document_count)
            indexed_categories = {self._category(alias) for alias in self.aliases}
            missing_categories = self.allowed_categories - indexed_categories
            if missing_categories:
                raise ValueError(
                    "Categorías ausentes en aliases híbridos: "
                    + ", ".join(sorted(missing_categories))
                )

    @classmethod
    def from_paths(
        cls,
        bm25_index_dir: Path,
        faiss_index_path: Path,
        aliases_path: Path,
        encoder: Encoder,
        top_k: int = 15,
        alpha: float = 0.5,
        allowed_categories: tuple[str, ...] | None = None,
    ) -> HybridBM25Retriever:
        """Load the Pyserini and FAISS indices from the project's artifact paths."""
        return cls(
            LuceneSearcher(str(bm25_index_dir)),
            faiss.read_index(str(faiss_index_path)),
            np.load(aliases_path, allow_pickle=False),
            encoder,
            top_k,
            alpha,
            allowed_categories,
        )

    @staticmethod
    def _category(alias: str) -> str:
        category, separator, _ = alias.partition(":")
        if not separator or not category:
            raise ValueError(f"Alias híbrido inválido: {alias!r}")
        return category

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        if not scores:
            return []
        low, high = min(scores), max(scores)
        if high - low < 1e-12:
            return [0.0] * len(scores)
        return [(score - low) / (high - low) for score in scores]

    def _bm25_results(self, query: str, top_k: int) -> list[tuple[str, float, str]]:
        results = []
        for hit in self.bm25.search(query, k=top_k):
            try:
                raw = json.loads(hit.raw)
                alias = raw["id"]
            except (TypeError, json.JSONDecodeError, KeyError) as exc:
                raise ValueError("El resultado BM25 no contiene un JSON raw con 'id'.") from exc
            results.append((str(alias), float(hit.score), str(raw.get("intent", ""))))
        return results

    def _dense_results(self, query: str, top_k: int) -> list[tuple[str, float]]:
        count = min(top_k, self.faiss_index.ntotal)
        if count == 0:
            return []
        query_vector = self.encoder.encode([query])
        scores, positions = self.faiss_index.search(query_vector, count)
        return [
            (self.aliases[position], float(score))
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0
        ]

    def search(
        self,
        query: str,
        top_k: int | None = None,
        alpha: float | None = None,
    ) -> list[tuple[str, float, str]]:
        """Return (alias, fused_score, intent), ordered by descending score.

        ``alpha`` is the dense-search weight: 0 selects BM25, 1 selects FAISS.
        """
        weight = self.alpha if alpha is None else alpha
        if not 0 <= weight <= 1:
            raise ValueError("alpha debe estar entre 0 y 1.")
        limit = self.top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k debe ser mayor que cero.")

        bm25_results = self._bm25_results(query, limit)
        dense_results = self._dense_results(query, limit)
        bm25_scores = self._normalize([score for _, score, _ in bm25_results])
        dense_scores = self._normalize([score for _, score in dense_results])
        bm25_ranked = {
            alias: (bm25_scores[rank], intent, rank)
            for rank, (alias, _, intent) in enumerate(bm25_results)
        }
        dense_ranked = {
            alias: (dense_scores[rank], rank)
            for rank, (alias, _) in enumerate(dense_results)
        }

        aliases = dict.fromkeys(
            [alias for alias, _, _ in bm25_results]
            + [alias for alias, _ in dense_results]
        )
        fused = []
        for alias in aliases:
            bm25_score, intent, bm25_position = bm25_ranked.get(alias, (0.0, "", limit))
            dense_score, dense_position = dense_ranked.get(alias, (0.0, limit))
            score = weight * dense_score + (1 - weight) * bm25_score
            if weight == 0:
                tie_break = bm25_position
            elif weight == 1:
                tie_break = dense_position
            else:
                tie_break = min(bm25_position, dense_position)
            fused.append((alias, score, tie_break, intent))

        fused.sort(key=lambda result: (-result[1], result[2], result[0]))
        return [(alias, score, intent) for alias, score, _, intent in fused[:limit]]

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Expose the retriever protocol used by the semantic search engine."""
        if top_k <= 0:
            raise ValueError("top_k debe ser mayor que cero.")
        search_limit = self.document_count if self.allowed_categories is not None else top_k
        results = []
        for alias, score, _ in self.search(query, search_limit):
            if (
                self.allowed_categories is not None
                and self._category(alias) not in self.allowed_categories
            ):
                continue
            results.append((alias, score))
            if len(results) == top_k:
                break
        return results
