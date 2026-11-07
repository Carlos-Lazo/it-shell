import unittest
from pathlib import Path

from evaluation.evaluate_slm_pipeline import (
    MockSLMAssistant,
    aggregate_metrics,
    generate_markdown_report,
)


class TestEvaluateSLMPipeline(unittest.TestCase):
    def test_aggregate_metrics_computes_correct_percentages(self):
        sample_results = [
            {
                "classification": {
                    "is_correct": True,
                    "is_false_exclusion": False,
                    "latency_s": 0.05,
                },
                "retrieval": {
                    "global": {"hit@1": 0.0, "hit@3": 1.0, "hit@5": 1.0, "hit@10": 1.0, "mrr": 0.5},
                    "slm_hard": {"hit@1": 0.0, "hit@3": 1.0, "hit@5": 1.0, "hit@10": 1.0, "mrr": 0.5},
                    "slm_boosted": {"hit@1": 1.0, "hit@3": 1.0, "hit@5": 1.0, "hit@10": 1.0, "mrr": 1.0},
                    "oracle_hard": {"hit@1": 0.0, "hit@3": 1.0, "hit@5": 1.0, "hit@10": 1.0, "mrr": 0.5},
                    "oracle_boosted": {"hit@1": 1.0, "hit@3": 1.0, "hit@5": 1.0, "hit@10": 1.0, "mrr": 1.0},
                },
                "selection": {
                    "baseline_correct": False,
                    "slm_correct": True,
                    "latency_s": 0.1,
                },
            },
            {
                "classification": {
                    "is_correct": True,
                    "is_false_exclusion": False,
                    "latency_s": 0.04,
                },
                "retrieval": {
                    "global": {"hit@1": 1.0, "hit@3": 1.0, "hit@5": 1.0, "hit@10": 1.0, "mrr": 1.0},
                    "slm_hard": {"hit@1": 1.0, "hit@3": 1.0, "hit@5": 1.0, "hit@10": 1.0, "mrr": 1.0},
                    "slm_boosted": {"hit@1": 1.0, "hit@3": 1.0, "hit@5": 1.0, "hit@10": 1.0, "mrr": 1.0},
                    "oracle_hard": {"hit@1": 1.0, "hit@3": 1.0, "hit@5": 1.0, "hit@10": 1.0, "mrr": 1.0},
                    "oracle_boosted": {"hit@1": 1.0, "hit@3": 1.0, "hit@5": 1.0, "hit@10": 1.0, "mrr": 1.0},
                },
                "selection": {
                    "baseline_correct": True,
                    "slm_correct": True,
                    "latency_s": 0.08,
                },
            },
        ]

        metrics = aggregate_metrics(sample_results)
        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["classification"]["accuracy"], 1.0)
        self.assertEqual(metrics["classification"]["false_exclusion_rate"], 0.0)
        self.assertEqual(metrics["retrieval"]["global"]["hit@1"], 0.5)
        self.assertEqual(metrics["retrieval"]["slm_boosted"]["hit@1"], 1.0)
        self.assertEqual(metrics["selection"]["baseline_accuracy"], 0.5)
        self.assertEqual(metrics["selection"]["slm_accuracy"], 1.0)
        self.assertEqual(metrics["selection"]["relative_improvement"], 0.5)

    def test_mock_assistant_classifies_filesystem_and_networking(self):
        assistant = MockSLMAssistant()
        res_fs = assistant.analyze_query("crear un nuevo directorio anidado")
        self.assertEqual(res_fs.predicted_category, "filesystem")
        self.assertEqual(res_fs.category_confidence, "high")

        res_net = assistant.analyze_query("ver puertos y conexiones activas")
        self.assertEqual(res_net.predicted_category, "networking")
        self.assertEqual(res_net.category_confidence, "high")

    def test_markdown_report_generation(self, tmp_path=None):
        import tempfile

        report_data = {
            "metadata": {"model": "qwen2.5:1.5b"},
            "overall": {
                "count": 10,
                "classification": {"accuracy": 0.9, "false_exclusion_rate": 0.05},
                "retrieval": {
                    "global": {"hit@1": 0.4, "hit@3": 0.5, "hit@5": 0.6, "hit@10": 0.7, "mrr": 0.5},
                    "slm_hard": {"hit@1": 0.35, "hit@3": 0.45, "hit@5": 0.55, "hit@10": 0.65, "mrr": 0.45},
                    "slm_boosted": {"hit@1": 0.8, "hit@3": 0.85, "hit@5": 0.9, "hit@10": 0.95, "mrr": 0.85},
                    "oracle_hard": {"hit@1": 0.4, "hit@3": 0.5, "hit@5": 0.6, "hit@10": 0.7, "mrr": 0.5},
                    "oracle_boosted": {"hit@1": 0.85, "hit@3": 0.9, "hit@5": 0.92, "hit@10": 0.98, "mrr": 0.88},
                },
                "selection": {"baseline_accuracy": 0.4, "slm_accuracy": 0.8},
            },
            "by_partition": {
                "development": {
                    "count": 7,
                    "classification": {"accuracy": 0.9},
                    "retrieval": {
                        "global": {"hit@1": 0.4},
                        "slm_boosted": {"hit@1": 0.8, "hit@5": 0.9},
                    },
                    "selection": {"baseline_accuracy": 0.4, "slm_accuracy": 0.8},
                },
                "test": {
                    "count": 3,
                    "classification": {"accuracy": 0.9},
                    "retrieval": {
                        "global": {"hit@1": 0.4},
                        "slm_boosted": {"hit@1": 0.8, "hit@5": 0.9},
                    },
                    "selection": {"baseline_accuracy": 0.4, "slm_accuracy": 0.8},
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "report.md"
            content = generate_markdown_report(report_data, out_file)
            self.assertTrue(out_file.exists())
            self.assertIn("# Informe de Evaluación Técnica", content)
            self.assertIn("qwen2.5:1.5b", content)
            self.assertIn("80.0%", content)


if __name__ == "__main__":
    unittest.main()

