import unittest
from copy import deepcopy
from math import isclose
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from evaluation.evaluate_precision_benchmark import score, validate_benchmark
from evaluation.capture_precision_pilot import hash_index_files, select_queries
from semantic_search_engine.retrieval_service.retriever import FaissRetriever
from semantic_search_engine.shared.config import load_config


def case(annotation_status="adjudicated"):
    return {
        "id": "case_1",
        "split": "development",
        "area": "arguments",
        "language": "en",
        "difficulty": "easy",
        "query": "find files in /tmp",
        "expected_policy": "preserve the path",
        "gold": {
            "policy": "propose",
            "steps": [
                {
                    "accepted_command_bases": ["find"],
                    "arguments": [{"kind": "path", "value": "/tmp"}],
                    "required_template_fragments": ["/tmp"],
                    "depends_on": [],
                    "connector_before": None,
                    "redirect_operator": None,
                    "redirect_target": None,
                }
            ],
            "requires_input": [],
            "required_issue_codes": [],
        },
        "annotation": {
            "status": annotation_status,
            "reviewers": ["reviewer-a", "reviewer-b"]
            if annotation_status == "adjudicated"
            else [],
        },
    }


def critical_case():
    item = case()
    item["id"] = "critical_case"
    item["query"] = "delete files recursively"
    item["area"] = "safety"
    item["expected_policy"] = "warn about destructive behavior"
    item["gold"]["policy"] = "propose_with_warning"
    item["gold"]["steps"][0]["accepted_command_bases"] = ["rm"]
    item["gold"]["steps"][0]["arguments"] = []
    item["gold"]["steps"][0]["required_template_fragments"] = ["-r"]
    item["gold"]["required_issue_codes"] = ["destructive_command"]
    return item


def output_record():
    return {
        "id": "case_1",
        "response": {
            "plan": {
                "steps": [
                    {
                        "arguments": [["path", "/tmp"]],
                        "redirect_operator": None,
                        "redirect_target": None,
                    }
                ]
            },
            "proposal": {
                "steps": [
                    {
                        "selected": {"command": "find"},
                        "command_template": "find /tmp",
                        "depends_on": [],
                        "connector_before": None,
                    }
                ],
                "requires_input": [],
                "issues": [],
                "executable": False,
            },
        },
    }


class TestPrecisionBenchmark(unittest.TestCase):
    def test_repository_benchmark_has_separate_splits_and_no_retrieval_duplicates(self):
        from pathlib import Path

        import json

        root = Path(__file__).resolve().parents[1]
        benchmark = json.loads(
            (root / "evaluation/precision_benchmark_v1.json").read_text(encoding="utf-8")
        )
        retrieval = json.loads(
            (root / "evaluation/evaluation_v2_queries.json").read_text(encoding="utf-8")
        )

        cases = validate_benchmark(benchmark, retrieval)

        self.assertEqual(len(cases), 22)
        self.assertEqual(
            {case["split"] for case in cases},
            {"development", "validation", "holdout"},
        )
        self.assertTrue(all(case["annotation"]["status"] == "pending" for case in cases))

    def test_domain_benchmark_excludes_product_corpora(self):
        from pathlib import Path

        import json

        root = Path(__file__).resolve().parents[1]
        benchmark = json.loads(
            (root / "evaluation/precision_benchmark_v1.json").read_text(encoding="utf-8")
        )
        scope = benchmark["retrieval_scope"]
        self.assertEqual(scope["name"], "debian12_system_commands")
        self.assertEqual(len(scope["categories"]), 12)
        self.assertEqual(scope["excluded_domains"], ["docker", "git"])

    def test_hybrid_debian_config_uses_existing_bm25_index(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "evaluation/debian12_hybrid_precision.yaml")

        self.assertEqual(config.retrieval_mode, "hybrid")
        self.assertEqual(config.hybrid_alpha, 0.5)
        self.assertEqual(len(config.allowed_categories or ()), 12)
        self.assertTrue(config.bm25_index_path and config.bm25_index_path.is_dir())

    def test_curated_hybrid_config_records_search_phrase_source(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "evaluation/debian12_hybrid_curated_precision.yaml")

        self.assertEqual(
            config.bm25_search_phrases_path,
            root / "evaluation/bm25_search_phrases_debian12.json",
        )
        self.assertTrue(config.bm25_search_phrases_path.is_file())

    def test_capture_can_select_development_and_validation_without_holdout(self):
        payload = {
            "cases": [
                {"id": "dev", "split": "development"},
                {"id": "val", "split": "validation"},
                {"id": "holdout", "split": "holdout"},
            ]
        }

        selected = select_queries(payload, {"development", "validation"})

        self.assertEqual([case["id"] for case in selected], ["dev", "val"])

    def test_capture_split_selection_requires_matching_cases(self):
        payload = {"cases": [{"id": "dev", "split": "development"}]}

        with self.assertRaisesRegex(ValueError, "No benchmark cases found"):
            select_queries(payload, {"holdout"})

    def test_hash_index_files_records_each_index_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir)
            (index_dir / "segments").write_text("index contents", encoding="utf-8")
            (index_dir / "write.lock").write_text("", encoding="utf-8")

            hashes = hash_index_files(index_dir)

        self.assertEqual(list(hashes), ["segments"])
        self.assertEqual(len(hashes["segments"]), 64)

    def test_scoped_retriever_searches_full_ranking_then_filters_categories(self):
        class FakeIndex:
            ntotal = 5

            def search(self, query, k):
                self.requested_k = k
                return (
                    np.array([[0.99, 0.95, 0.9, 0.8, 0.7]], dtype=np.float32)[:, :k],
                    np.array([[0, 1, 2, 3, 4]], dtype=np.int64)[:, :k],
                )

        class FakeEncoder:
            def encode(self, texts):
                return np.array([[1.0]], dtype=np.float32)

        retriever = FaissRetriever.__new__(FaissRetriever)
        retriever.index = FakeIndex()
        retriever.aliases = np.array([
            "docker:1",
            "git:1",
            "filesystem:1",
            "misc:1",
            "docker:2",
        ])
        retriever.encoder = FakeEncoder()
        retriever.allowed_categories = frozenset({"filesystem", "misc"})

        results = retriever.retrieve("list files", 2)

        self.assertEqual(retriever.index.requested_k, retriever.index.ntotal)
        self.assertEqual([alias for alias, _ in results], ["filesystem:1", "misc:1"])
        self.assertTrue(isclose(results[0][1], 0.9, rel_tol=1e-6))
        self.assertTrue(isclose(results[1][1], 0.8, rel_tol=1e-6))

    def test_unscoped_retriever_keeps_global_top_k_behavior(self):
        class FakeIndex:
            ntotal = 5

            def search(self, query, k):
                self.requested_k = k
                return (
                    np.array([[0.99, 0.95]], dtype=np.float32)[:, :k],
                    np.array([[0, 1]], dtype=np.int64)[:, :k],
                )

        class FakeEncoder:
            def encode(self, texts):
                return np.array([[1.0]], dtype=np.float32)

        retriever = FaissRetriever.__new__(FaissRetriever)
        retriever.index = FakeIndex()
        retriever.aliases = np.array(["docker:1", "git:1"])
        retriever.encoder = FakeEncoder()
        retriever.allowed_categories = None

        results = retriever.retrieve("git status", 2)

        self.assertEqual(retriever.index.requested_k, 2)
        self.assertEqual([alias for alias, _ in results], ["docker:1", "git:1"])

    def test_scoped_retriever_rejects_unknown_categories(self):
        class FakeIndex:
            ntotal = 1

        with (
            patch(
                "semantic_search_engine.retrieval_service.retriever.faiss.read_index",
                return_value=FakeIndex(),
            ),
            patch(
                "semantic_search_engine.retrieval_service.retriever.np.load",
                return_value=np.array(["filesystem:1"]),
            ),
            self.assertRaisesRegex(ValueError, "Categorías ausentes"),
        ):
            FaissRetriever(
                Path("/unused"),
                Path("/unused"),
                None,
                allowed_categories=("kubernetes",),
            )

    def test_duplicate_case_queries_are_rejected(self):
        first = case()
        second = deepcopy(first)
        second["id"] = "case_2"
        payload = {"schema_version": 1, "cases": [first, second]}

        with self.assertRaisesRegex(ValueError, "Duplicate benchmark query"):
            validate_benchmark(payload)

    def test_adjudicated_case_requires_two_distinct_reviewers(self):
        item = case()
        item["annotation"]["reviewers"] = ["reviewer-a", "reviewer-a"]

        with self.assertRaisesRegex(ValueError, "two distinct reviewers"):
            validate_benchmark({"schema_version": 1, "cases": [item]})

    def test_pending_cases_are_not_scored(self):
        item = case("pending")

        result = score([item], {"records": [output_record()]}, "development")

        self.assertEqual(result["status"], "not_scored")
        self.assertEqual(result["pending_cases"], 1)

    def test_adjudicated_case_scores_structured_output(self):
        item = case()

        result = score([item], {"records": [output_record()]}, "development")

        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["task_success_rate"], 1.0)
        self.assertEqual(result["command_accuracy"], 1.0)
        self.assertEqual(result["argument_accuracy"], 1.0)
        self.assertEqual(result["command_content_accuracy"], 1.0)
        self.assertEqual(result["executable_false_rate"], 1.0)

    def test_wrong_argument_and_executable_state_fail_case(self):
        item = case()
        output = output_record()
        output["response"]["plan"]["steps"][0]["arguments"] = [["path", "/var"]]
        output["response"]["proposal"]["steps"][0]["command_template"] = "find /var"
        output["response"]["proposal"]["executable"] = True

        result = score([item], {"records": [output]}, "development")

        self.assertEqual(result["task_success_rate"], 0.0)
        self.assertEqual(result["argument_accuracy"], 0.0)
        self.assertEqual(result["command_content_accuracy"], 0.0)
        self.assertEqual(result["executable_false_rate"], 0.0)

    def test_critical_issue_false_negative_is_counted(self):
        item = critical_case()
        output = output_record()
        output["id"] = "critical_case"
        output["response"]["proposal"]["steps"][0]["selected"]["command"] = "rm"
        output["response"]["proposal"]["steps"][0]["command_template"] = "rm -rf /tmp"
        output["response"]["proposal"]["steps"][0]["depends_on"] = []

        result = score([item], {"records": [output]}, "development")

        self.assertEqual(result["critical_cases"], 1)
        self.assertEqual(result["critical_false_negatives"], 1)
        self.assertEqual(result["critical_issue_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
