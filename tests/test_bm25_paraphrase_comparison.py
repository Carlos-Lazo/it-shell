import unittest

from evaluation.compare_bm25_paraphrases import metrics, summarize


class BM25ParaphraseComparisonTests(unittest.TestCase):
    def test_metrics_use_ranked_relevant_aliases(self):
        result = metrics(
            ["filesystem:1", "filesystem:2", "filesystem:3"],
            {"filesystem:2"},
        )

        self.assertEqual(result["hit@1"], 0.0)
        self.assertEqual(result["hit@3"], 1.0)
        self.assertEqual(result["mrr"], 0.5)
        self.assertAlmostEqual(result["ndcg@5"], 1 / 1.584962500721156)

    def test_summary_averages_per_query_metrics(self):
        summary = summarize([
            {
                "hit@1": 1.0,
                "hit@3": 1.0,
                "hit@5": 1.0,
                "hit@10": 1.0,
                "ndcg@5": 1.0,
                "ndcg@10": 1.0,
                "mrr": 1.0,
            },
            {
                "hit@1": 0.0,
                "hit@3": 0.0,
                "hit@5": 0.0,
                "hit@10": 0.0,
                "ndcg@5": 0.0,
                "ndcg@10": 0.0,
                "mrr": 0.0,
            },
        ])

        self.assertEqual(summary["queries"], 2)
        self.assertEqual(summary["hit@1"], 0.5)
        self.assertEqual(summary["mrr"], 0.5)


if __name__ == "__main__":
    unittest.main()
