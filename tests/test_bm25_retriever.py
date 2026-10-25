from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

import faiss
import numpy as np

from semantic_search_engine.retrieval_service.bm25_search import HybridBM25Retriever


@dataclass
class Hit:
    raw: str
    score: float


class FakeBM25Searcher:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits

    def search(self, query: str, k: int) -> list[Hit]:
        return self.hits[:k]


class FakeEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0]], dtype=np.float32)


def hit(alias: str, score: float, intent: str = "") -> Hit:
    return Hit(json.dumps({"id": alias, "intent": intent}), score)


class TestHybridBM25Retriever(unittest.TestCase):
    def setUp(self) -> None:
        self.index = faiss.IndexFlatIP(2)
        self.index.add(
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        self.aliases = np.array(["dense:1", "shared:1", "dense:2"])
        self.searcher = FakeBM25Searcher(
            [
                hit("shared:1", 8.0, "shared intent"),
                hit("bm25:1", 4.0, "lexical intent"),
            ]
        )
        self.retriever = HybridBM25Retriever(
            self.searcher, self.index, self.aliases, FakeEncoder(), top_k=2
        )

    def test_alpha_zero_preserves_bm25_order_and_intents(self) -> None:
        results = self.retriever.search("query", alpha=0.0)
        self.assertEqual([alias for alias, _, _ in results], ["shared:1", "bm25:1"])
        self.assertEqual(results[0][2], "shared intent")

    def test_alpha_one_preserves_dense_order(self) -> None:
        results = self.retriever.search("query", alpha=1.0)
        self.assertEqual([alias for alias, _, _ in results], ["dense:1", "shared:1"])

    def test_hybrid_merges_rankings_and_retriever_protocol(self) -> None:
        results = self.retriever.search("query", alpha=0.5, top_k=3)
        self.assertEqual(
            {alias for alias, _, _ in results},
            {"dense:1", "shared:1", "bm25:1"},
        )
        self.assertEqual(len(self.retriever.retrieve("query", 2)), 2)
        self.assertTrue(all(0.0 <= score <= 1.0 for _, score, _ in results))

    def test_invalid_parameters_and_misaligned_aliases_fail(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.search("query", alpha=1.1)
        with self.assertRaises(ValueError):
            self.retriever.search("query", top_k=0)
        with self.assertRaises(ValueError):
            HybridBM25Retriever(
                self.searcher, self.index, self.aliases[:2], FakeEncoder()
            )

    def test_malformed_bm25_result_fails_explicitly(self) -> None:
        retriever = HybridBM25Retriever(
            FakeBM25Searcher([Hit("{}", 1.0)]),
            self.index,
            self.aliases,
            FakeEncoder(),
        )
        with self.assertRaisesRegex(ValueError, "JSON raw"):
            retriever.search("query")


if __name__ == "__main__":
    unittest.main()
