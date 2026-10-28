#!/usr/bin/env python3
"""
test_bm25.py
------------
Tests funcionales del índice BM25 (Lucene / Pyserini).

Ejecutar desde la raíz del repo con el venv del indexer:
    dataset_builder/indexer/.venv/bin/python -m pytest tests/test_bm25.py -v
"""

from __future__ import annotations

import json
import pathlib
import unittest

from pyserini.search.lucene import LuceneSearcher

# ---------------------------------------------------------------------------
# Rutas absolutas derivadas de la ubicación de este archivo
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]   # tests/ -> code/
BM25_INDEX = REPO_ROOT / "dataset_builder" / "indexer" / "artifacts" / "bm25_indexs_v2"


class TestBM25Index(unittest.TestCase):
    """Suite de tests para el índice BM25 creado con bm25_indexer.py."""

    @classmethod
    def setUpClass(cls) -> None:
        """Carga el índice una sola vez para toda la suite."""
        if not BM25_INDEX.exists():
            raise FileNotFoundError(
                f"Índice BM25 no encontrado en: {BM25_INDEX}\n"
                "Ejecútalo primero con:\n"
                "  dataset_builder/indexer/.venv/bin/python "
                "dataset_builder/indexer/bm25_indexer.py"
            )
        cls.searcher = LuceneSearcher(str(BM25_INDEX))

    # ------------------------------------------------------------------
    def test_index_not_empty(self) -> None:
        """El índice debe contener documentos."""
        hits = self.searcher.search("git", k=1)
        self.assertGreater(len(hits), 0, "El índice BM25 no devuelve resultados")

    def test_hit_has_raw_json(self) -> None:
        """Cada hit debe exponer el campo raw con JSON parseable."""
        hits = self.searcher.search("git add", k=5)
        for hit in hits:
            with self.subTest(docid=hit.docid):
                self.assertIsNotNone(hit.raw, "hit.raw no debe ser None")
                parsed = json.loads(hit.raw)
                self.assertIn("id", parsed, "El JSON del hit debe tener 'id'")
                self.assertIn("contents", parsed, "El JSON del hit debe tener 'contents'")

    def test_alias_format_category_colon_id(self) -> None:
        """Los IDs del índice deben tener formato 'category:numero'."""
        hits = self.searcher.search("list files", k=10)
        self.assertGreater(len(hits), 0)
        for hit in hits:
            raw = json.loads(hit.raw)
            alias = raw["id"]
            self.assertIn(
                ":", alias,
                f"El alias '{alias}' no sigue el formato 'category:id'"
            )

    def test_intent_field_present(self) -> None:
        """El campo intent debe estar disponible en el JSON del hit."""
        hits = self.searcher.search("remove git branch", k=5)
        for hit in hits:
            raw = json.loads(hit.raw)
            self.assertIn("intent", raw, f"Falta 'intent' en el hit {raw.get('id')}")
            self.assertIsInstance(raw["intent"], str)

    def test_scores_are_positive_and_ordered(self) -> None:
        """Los scores BM25 deben ser positivos y devueltos en orden descendente."""
        hits = self.searcher.search("docker run container", k=10)
        if len(hits) < 2:
            self.skipTest("No hay suficientes resultados para verificar orden")
        scores = [h.score for h in hits]
        self.assertTrue(all(s > 0 for s in scores), "Todos los scores deben ser > 0")
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(
                scores[i], scores[i + 1],
                f"Score no ordenado descendente: pos {i}={scores[i]} > pos {i+1}={scores[i+1]}"
            )

    def test_top_k_respected(self) -> None:
        """El buscador no debe devolver más resultados de los pedidos."""
        for k in (1, 5, 10):
            with self.subTest(k=k):
                hits = self.searcher.search("network connection", k=k)
                self.assertLessEqual(len(hits), k)

    def test_relevant_category_in_top_results_git(self) -> None:
        """Para una consulta git, debe aparecer al menos un resultado 'git:...' en top-5."""
        hits = self.searcher.search("commit changes to repository", k=5)
        aliases = [json.loads(h.raw)["id"] for h in hits]
        has_git = any(a.startswith("git:") for a in aliases)
        self.assertTrue(has_git, f"Ningún resultado git en top-5: {aliases}")

    def test_relevant_category_in_top_results_docker(self) -> None:
        """Para una consulta docker, debe aparecer al menos un resultado 'docker:...' en top-5."""
        hits = self.searcher.search("build docker image from dockerfile", k=5)
        aliases = [json.loads(h.raw)["id"] for h in hits]
        has_docker = any(a.startswith("docker:") for a in aliases)
        self.assertTrue(has_docker, f"Ningún resultado docker en top-5: {aliases}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
