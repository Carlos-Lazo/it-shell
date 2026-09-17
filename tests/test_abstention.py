import unittest
from semantic_search_engine.shared.contracts import (
    CandidateCommand,
    PlanStep,
    QueryAnalysis,
    QueryPlan,
    SearchResponse,
)
from semantic_search_engine.shared.config import SelectionConfig
from semantic_search_engine.pipeline_builder_service.pipeline_builder import PipelineBuilder
from semantic_search_engine.pipeline_builder_service.validator import ProposalValidator
from semantic_search_engine.response_formatter_service.service import ResponseFormatterService


class TestAbstention(unittest.TestCase):
    def setUp(self):
        self.config = SelectionConfig(min_score=0.45, alternative_margin=0.04, max_alternatives=3)
        self.builder = PipelineBuilder(self.config)
        self.validator = ProposalValidator()
        self.formatter = ResponseFormatterService()

    def test_choose_above_threshold(self):
        candidate1 = CandidateCommand("fs:1", "filesystem", 1, "list files", "ls", "ls -la", 0.75)
        candidate2 = CandidateCommand("fs:2", "filesystem", 2, "list dir", "dir", "dir", 0.73)
        primary, alternatives = self.builder.choose([candidate1, candidate2])
        self.assertIsNotNone(primary)
        self.assertEqual(primary.command, "ls")
        self.assertEqual(len(alternatives), 1)

    def test_choose_below_threshold_abstains(self):
        low1 = CandidateCommand("misc:1", "misc", 1, "unrelated", "cmd", "cmd", 0.30)
        low2 = CandidateCommand("misc:2", "misc", 2, "other", "cmd2", "cmd2", 0.28)
        primary, alternatives = self.builder.choose([low1, low2])
        self.assertIsNone(primary)
        self.assertEqual(len(alternatives), 0)

    def test_choose_excludes_alternatives_below_threshold(self):
        high = CandidateCommand("fs:1", "filesystem", 1, "list files", "ls", "ls", 0.46)
        low = CandidateCommand("fs:2", "filesystem", 2, "list dir", "dir", "dir", 0.44)
        primary, alternatives = self.builder.choose([high, low])
        self.assertIsNotNone(primary)
        self.assertEqual(len(alternatives), 0)

    def test_abstention_flow_and_formatting(self):
        plan = QueryPlan("make a coffee", (PlanStep("step_1", "make a coffee"),))
        low = CandidateCommand("misc:1", "misc", 1, "irrelevant", "dummy", "dummy", 0.20)
        proposal = self.validator.validate(self.builder.build(plan, {"step_1": [low]}))

        self.assertIsNone(proposal.steps[0].selected)
        self.assertEqual(len(proposal.issues), 1)
        self.assertEqual(proposal.issues[0].code, "no_candidate")

        response = SearchResponse(
            QueryAnalysis("make a coffee", "make a coffee", "en", False),
            plan,
            proposal,
        )

        cli_out = self.formatter.format(response, "cli")
        self.assertIn("No se encontró una recomendación confiable", cli_out)
        self.assertIn("No se encontró un candidato confiable", cli_out)

        md_out = self.formatter.format(response, "markdown")
        self.assertIn("*No se encontró una recomendación confiable*", md_out)
        self.assertIn("No se encontró un candidato confiable", md_out)

        json_out = self.formatter.format(response, "json")
        self.assertIn('"code": "no_candidate"', json_out)


if __name__ == "__main__":
    unittest.main()
