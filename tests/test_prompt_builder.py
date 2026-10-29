import json
import unittest

from semantic_search_engine.pipeline_builder_service.validator import ProposalValidator
from semantic_search_engine.prompt_builder_service.llm_client import LLMClientError
from semantic_search_engine.prompt_builder_service.prompt_builder import PromptBuilder
from semantic_search_engine.prompt_builder_service.service import PromptBuilderService
from semantic_search_engine.response_formatter_service.service import ResponseFormatterService
from semantic_search_engine.engine import SemanticSearchEngine
from semantic_search_engine.shared.contracts import (
    CandidateCommand,
    PipelineProposal,
    PipelineStep,
    QueryAnalysis,
    QueryPlan,
    QueryRequest,
    SearchResponse,
)


class FakeLLM:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.prompt = None

    def generate(self, prompt):
        self.prompt = prompt
        if self.error:
            raise self.error
        return self.result


class TestPromptBuilder(unittest.TestCase):
    def make_proposal(self, command="find /var/log -type f"):
        candidate = CandidateCommand(
            "filesystem:1",
            "filesystem",
            1,
            "buscar archivos de log",
            "find",
            command,
            0.9,
        )
        step = PipelineStep("step_1", "buscar logs", (), candidate, (), command)
        proposal = PipelineProposal("buscar logs", (step,))
        return ProposalValidator().validate(proposal)

    def test_builds_structured_context_and_restricted_prompt(self):
        proposal = self.make_proposal()
        builder = PromptBuilder()

        context = builder.explanation_context(proposal)
        prompt = builder.explanation_prompt(proposal)

        self.assertEqual(context.goal, "buscar logs")
        self.assertEqual(context.steps[0].command, "find /var/log -type f")
        self.assertEqual(context.steps[0].flags, ("-type",))
        self.assertIn("No cambies, completes ni reordenes", prompt)
        self.assertIn("comando exacto=find /var/log -type f", prompt)

    def test_offline_explanation_is_deterministic(self):
        service = PromptBuilderService(PromptBuilder())

        explanation = service.explain(self.make_proposal())

        self.assertIn("Objetivo: buscar logs", explanation)
        self.assertIn("Comando sugerido: find /var/log -type f", explanation)

    def test_provider_error_falls_back_to_complete_offline_explanation(self):
        llm = FakeLLM(error=LLMClientError("provider unavailable"))
        service = PromptBuilderService(PromptBuilder(), llm)

        explanation = service.explain(self.make_proposal())

        self.assertIn("Comando sugerido: find /var/log -type f", explanation)
        self.assertIn("No cambies, completes ni reordenes", llm.prompt)

    def test_nonempty_provider_response_is_used(self):
        llm = FakeLLM(result="Explicación generada.")
        service = PromptBuilderService(PromptBuilder(), llm)

        self.assertEqual(service.explain(self.make_proposal()), "Explicación generada.")

    def test_empty_provider_response_falls_back_offline(self):
        service = PromptBuilderService(PromptBuilder(), FakeLLM(result="  "))

        explanation = service.explain(self.make_proposal())

        self.assertIn("Comando sugerido: find /var/log -type f", explanation)

    def test_unexpected_provider_errors_are_not_silently_hidden(self):
        service = PromptBuilderService(PromptBuilder(), FakeLLM(error=ValueError("bug")))

        with self.assertRaisesRegex(ValueError, "bug"):
            service.explain(self.make_proposal())

    def test_outputs_include_explanation_and_warnings_in_supported_formats(self):
        proposal = self.make_proposal("sudo rm -rf /tmp/logs")
        response = SearchResponse(
            QueryAnalysis("buscar logs", "buscar logs", "es", False),
            QueryPlan("buscar logs", ()),
            proposal,
            explanation="Explicación local.",
        )
        formatter = ResponseFormatterService()

        cli = formatter.format(response, "cli")
        terminal = formatter.format(response, "terminal")
        markdown = formatter.format(response, "markdown")
        payload = json.loads(formatter.format(response, "json"))

        self.assertIn("Explicación local.", cli)
        self.assertIn("\033[", terminal)
        self.assertIn("Explicación local.", markdown)
        self.assertEqual(payload["explanation"], "Explicación local.")
        self.assertTrue(payload["proposal"]["issues"])
        self.assertTrue(payload["proposal"]["issues"][0]["recommendation"])

    def test_engine_includes_offline_explanation_in_response(self):
        proposal = self.make_proposal()
        analysis = QueryAnalysis("buscar logs", "buscar logs", "es", False)
        plan = QueryPlan("buscar logs", ())

        class Retrieval:
            def retrieve_plan(self, query, top_k):
                return analysis, plan, {}

        class Pipeline:
            def propose(self, plan, retrieved, reranker):
                return proposal

        response = SemanticSearchEngine(
            Retrieval(),
            object(),
            Pipeline(),
            PromptBuilderService(PromptBuilder()),
            ResponseFormatterService(),
        ).search(QueryRequest("buscar logs", output_format="json"))

        payload = json.loads(response)
        self.assertIn("Comando sugerido: find /var/log -type f", payload["explanation"])


if __name__ == "__main__":
    unittest.main()
