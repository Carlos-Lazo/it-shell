import unittest
import json

from mcp.shared.memory import create_connected_server_and_client_session
from semantic_search_engine.mcp.server import create_server
from semantic_search_engine.mcp.tools import MCPToolService
from semantic_search_engine.shared.contracts import (
    CandidateCommand,
    PlanStep,
    PipelineProposal,
    PipelineStep,
    QueryAnalysis,
    QueryPlan,
    SearchResponse,
)


class FakeRetrieval:
    def __init__(self):
        self.analysis = QueryAnalysis("list files", "list files", "en", False)
        self.step = PlanStep("step_1", "list files")
        self.plan = QueryPlan("list files", (self.step,))
        self.last_top_k = None

    def retrieve_plan(self, query, top_k):
        self.last_top_k = top_k
        return self.analysis, self.plan, {"step_1": [("filesystem:1", 0.9)]}


class FakeResolver:
    def resolve(self, alias, score):
        return CandidateCommand(
            alias, "filesystem", 1, "list files", "find", "find .", score
        )


class FakeReranker:
    def rerank(self, query, candidates):
        return candidates


class FakePipeline:
    def __init__(self):
        from semantic_search_engine.pipeline_builder_service.validator import ProposalValidator

        self.resolver = FakeResolver()
        self.validator = ProposalValidator()

    def propose(self, plan, retrieved, reranker):
        candidate = self.resolver.resolve("filesystem:1", 0.9)
        step = PipelineStep("step_1", "list files", (), candidate, (), "find .")
        return PipelineProposal("list files", (step,))


class FakeEngine:
    def __init__(self):
        self.retrieval = FakeRetrieval()
        self.reranker = FakeReranker()
        self.pipeline = FakePipeline()

    def search_response(self, request):
        proposal = self.pipeline.propose(None, None, None)
        return SearchResponse(
            self.retrieval.analysis,
            self.retrieval.plan,
            proposal,
            explanation="Offline explanation",
        )


class TestMCPToolService(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.tools = MCPToolService(self.engine)

    def test_search_commands_returns_candidates_and_limits_top_k(self):
        result = self.tools.search_commands(" list files ", top_k=5)

        self.assertEqual(result["query"], "list files")
        self.assertEqual(self.engine.retrieval.last_top_k, 5)
        self.assertEqual(result["steps"][0]["candidates"][0]["alias"], "filesystem:1")
        self.assertEqual(result["steps"][0]["candidates"][0]["score"], 0.9)

    def test_plan_operation_returns_structured_response(self):
        result = self.tools.plan_operation("list files", top_k=7)

        self.assertEqual(result["plan"]["goal"], "list files")
        self.assertEqual(result["proposal"]["steps"][0]["command_template"], "find .")
        self.assertFalse(result["proposal"]["executable"])
        self.assertEqual(result["explanation"], "Offline explanation")

    def test_validate_command_reports_risk_without_execution(self):
        result = self.tools.validate_command("sudo rm -rf /tmp/data")

        codes = {issue["code"] for issue in result["issues"]}
        self.assertIn("destructive_command", codes)
        self.assertIn("privileged_command", codes)
        self.assertFalse(result["executable"])

    def test_invalid_queries_commands_and_top_k_are_rejected(self):
        for value in ("", "  ", "x" * 2001):
            with self.subTest(query_length=len(value)):
                with self.assertRaises(ValueError):
                    self.tools.search_commands(value)
                with self.assertRaises(ValueError):
                    self.tools.plan_operation(value)

        for value in ("", " " * 2, "x" * 4097):
            with self.subTest(command_length=len(value)):
                with self.assertRaises(ValueError):
                    self.tools.validate_command(value)

        for top_k in (0, -1, 51, True, 2.5):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    self.tools.search_commands("query", top_k)
                with self.assertRaises(ValueError):
                    self.tools.plan_operation("query", top_k)

        with self.assertRaises(ValueError):
            self.tools.search_commands(None)
        with self.assertRaises(ValueError):
            self.tools.validate_command(None)


class TestMCPServerRegistration(unittest.IsolatedAsyncioTestCase):
    async def test_registers_all_three_tools(self):
        server = create_server(MCPToolService(FakeEngine()))

        registered = await server.list_tools()

        self.assertEqual(
            {tool.name for tool in registered},
            {"search_commands", "plan_operation", "validate_command"},
        )
        for tool in registered:
            self.assertTrue(tool.inputSchema["properties"])
            self.assertEqual(tool.outputSchema["type"], "object")

    async def test_registered_tools_call_service_and_return_structured_data(self):
        server = create_server(MCPToolService(FakeEngine()))

        async with create_connected_server_and_client_session(server) as client:
            registered = await client.list_tools()
            self.assertEqual(
                {tool.name for tool in registered.tools},
                {"search_commands", "plan_operation", "validate_command"},
            )

            search_result = await client.call_tool(
                "search_commands", {"query": "list files", "top_k": 3}
            )
            plan_result = await client.call_tool(
                "plan_operation", {"query": "list files"}
            )
            validation_result = await client.call_tool(
                "validate_command", {"command": "sudo rm -rf /tmp/data"}
            )

        search_payload = json.loads(search_result.content[0].text)
        plan_payload = json.loads(plan_result.content[0].text)
        validation_payload = json.loads(validation_result.content[0].text)
        self.assertFalse(search_result.isError)
        self.assertFalse(plan_result.isError)
        self.assertFalse(validation_result.isError)
        self.assertEqual(search_result.structuredContent, search_payload)
        self.assertEqual(plan_result.structuredContent, plan_payload)
        self.assertEqual(validation_result.structuredContent, validation_payload)
        self.assertEqual(search_payload["steps"][0]["candidates"][0]["alias"], "filesystem:1")
        self.assertEqual(plan_payload["proposal"]["steps"][0]["command_template"], "find .")
        self.assertFalse(plan_payload["proposal"]["executable"])
        self.assertIn(
            "destructive_command",
            {issue["code"] for issue in validation_payload["issues"]},
        )

    async def test_invalid_input_is_returned_as_tool_error(self):
        server = create_server(MCPToolService(FakeEngine()))

        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("plan_operation", {"query": " "})

        self.assertTrue(result.isError)


if __name__ == "__main__":
    unittest.main()
