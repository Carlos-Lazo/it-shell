import unittest
import numpy as np

from semantic_search_engine.retrieval_service.retriever import FaissRetriever


class FakeIndex:
    def __init__(self, scores, positions):
        self.scores = scores
        self.positions = positions
        self.ntotal = len(positions[0])
        self.requested_k = None

    def search(self, query, k):
        self.requested_k = k
        return (
            self.scores[:, :k],
            self.positions[:, :k],
        )


class FakeEncoder:
    def encode(self, texts):
        return np.array([[1.0]], dtype=np.float32)


class TestRetrieverCategoryBoosting(unittest.TestCase):
    def setUp(self):
        self.retriever = FaissRetriever.__new__(FaissRetriever)
        self.retriever.encoder = FakeEncoder()
        self.retriever.allowed_categories = None
        self.retriever.default_boosted_categories = None
        self.retriever.default_category_boost = 0.0

    def test_boosting_promotes_target_category_to_top_1(self):
        # Initial ranking: docker:1 (0.75) > filesystem:1 (0.70)
        self.retriever.aliases = np.array(["docker:1", "filesystem:1"])
        self.retriever.index = FakeIndex(
            scores=np.array([[0.75, 0.70]], dtype=np.float32),
            positions=np.array([[0, 1]], dtype=np.int64),
        )

        # Unboosted: docker:1 is #1
        res_unboosted = self.retriever.retrieve("create folder", top_k=2)
        self.assertEqual([alias for alias, _ in res_unboosted], ["docker:1", "filesystem:1"])

        # Boosted with 0.10 for filesystem: filesystem:1 becomes 0.70 + 0.10 = 0.80 > 0.75
        res_boosted = self.retriever.retrieve(
            "create folder",
            top_k=2,
            boosted_categories=("filesystem",),
            category_boost=0.10,
        )
        self.assertEqual([alias for alias, _ in res_boosted], ["filesystem:1", "docker:1"])
        self.assertAlmostEqual(res_boosted[0][1], 0.80, places=4)
        self.assertAlmostEqual(res_boosted[1][1], 0.75, places=4)

    def test_soft_boost_preserves_strong_out_of_category_candidate(self):
        # Initial ranking: textprocessing:1 (grep) has high score (0.92), filesystem:1 has (0.60)
        self.retriever.aliases = np.array(["textprocessing:1", "filesystem:1"])
        self.retriever.index = FakeIndex(
            scores=np.array([[0.92, 0.60]], dtype=np.float32),
            positions=np.array([[0, 1]], dtype=np.int64),
        )

        # Even with 0.15 boost to filesystem (0.60 + 0.15 = 0.75), textprocessing:1 (0.92) is not excluded!
        res = self.retriever.retrieve(
            "search pattern in file",
            top_k=2,
            boosted_categories=("filesystem",),
            category_boost=0.15,
        )
        self.assertEqual([alias for alias, _ in res], ["textprocessing:1", "filesystem:1"])
        self.assertAlmostEqual(res[0][1], 0.92, places=4)
        self.assertAlmostEqual(res[1][1], 0.75, places=4)

    def test_instance_default_boosting(self):
        self.retriever.default_boosted_categories = frozenset({"networking"})
        self.retriever.default_category_boost = 0.20
        self.retriever.aliases = np.array(["misc:1", "networking:1"])
        self.retriever.index = FakeIndex(
            scores=np.array([[0.65, 0.55]], dtype=np.float32),
            positions=np.array([[0, 1]], dtype=np.int64),
        )

        # Uses instance defaults
        res = self.retriever.retrieve("show open ports", top_k=2)
        # networking:1 (0.55 + 0.20 = 0.75) > misc:1 (0.65)
        self.assertEqual(res[0][0], "networking:1")
        self.assertAlmostEqual(res[0][1], 0.75, places=4)


if __name__ == "__main__":
    unittest.main()

