import unittest
from unittest.mock import MagicMock
from semantic_search_engine.shared.contracts import CandidateCommand
from semantic_search_engine.reranker_service.reranker import (
    IdentityReranker,
    CrossEncoderReranker,
)
from semantic_search_engine.reranker_service.service import RerankerService


class TestReranker(unittest.TestCase):
    def setUp(self):
        self.c1 = CandidateCommand("fs:1", "fs", 1, "list files", "ls", "ls", 0.70)
        self.c2 = CandidateCommand("fs:2", "fs", 2, "print working directory", "pwd", "pwd", 0.65)
        self.c3 = CandidateCommand("fs:3", "fs", 3, "change directory", "cd", "cd", 0.60)

    def test_identity_reranker_preserves_order(self):
        reranker = IdentityReranker()
        service = RerankerService(reranker)
        ranked = service.rerank("list files", [self.c1, self.c2, self.c3])
        self.assertEqual([c.alias for c in ranked], ["fs:1", "fs:2", "fs:3"])
        self.assertIsNone(ranked[0].reranker_score)

    def test_cross_encoder_reranks_and_sorts(self):
        reranker = CrossEncoderReranker("mock-model")
        mock_model = MagicMock()
        # Mock scores that reverse the initial order
        mock_model.predict.return_value = [0.10, 0.95, 0.40]
        reranker._model = mock_model

        service = RerankerService(reranker)
        ranked = service.rerank("show current dir", [self.c1, self.c2, self.c3])

        # c2 had score 0.95, c3 had 0.40, c1 had 0.10
        self.assertEqual([c.alias for c in ranked], ["fs:2", "fs:3", "fs:1"])
        self.assertEqual(ranked[0].reranker_score, 0.95)
        self.assertEqual(ranked[0].score, 0.95)
        self.assertEqual(ranked[1].reranker_score, 0.40)
        self.assertEqual(ranked[2].reranker_score, 0.10)

    def test_cross_encoder_empty_candidates(self):
        reranker = CrossEncoderReranker("mock-model")
        service = RerankerService(reranker)
        ranked = service.rerank("query", [])
        self.assertEqual(ranked, [])


if __name__ == "__main__":
    unittest.main()

