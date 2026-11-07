import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from semantic_search_engine.llm.ollama_assistant import (
    AnalysisStep,
    CandidateOption,
    LLMQueryAnalysis,
    LLMSelectionResult,
    OllamaAssistant,
    OllamaConnectionError,
    OllamaResponseError,
)


class TestOllamaAssistant(unittest.TestCase):
    def setUp(self):
        self.assistant = OllamaAssistant(model="qwen2.5:1.5b", base_url="http://localhost:11434")

    @patch("semantic_search_engine.llm.ollama_assistant.urlopen")
    def test_is_available_true_when_200(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        self.assertTrue(self.assistant.is_available())

    @patch("semantic_search_engine.llm.ollama_assistant.urlopen")
    def test_is_available_false_on_error(self, mock_urlopen):
        mock_urlopen.side_effect = URLError("Connection refused")
        self.assertFalse(self.assistant.is_available())

    @patch.object(OllamaAssistant, "_chat_request")
    def test_analyze_query_successful(self, mock_chat):
        mock_chat.return_value = {
            "predicted_category": "filesystem",
            "category_confidence": "high",
            "language": "es",
            "steps": [
                {
                    "step_index": 1,
                    "task": "crear carpetas anidadas",
                    "search_query": "mkdir parent directory recursively",
                    "arguments": {"path": "/tmp/a/b/c"},
                    "clarification_needed": False,
                    "clarification_question": None,
                }
            ],
        }

        result = self.assistant.analyze_query("crear carpetas anidadas en /tmp/a/b/c")

        self.assertIsInstance(result, LLMQueryAnalysis)
        self.assertEqual(result.predicted_category, "filesystem")
        self.assertEqual(result.category_confidence, "high")
        self.assertTrue(result.is_confident_category)
        self.assertEqual(result.language, "es")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].search_query, "mkdir parent directory recursively")
        self.assertEqual(result.steps[0].arguments, {"path": "/tmp/a/b/c"})

    @patch.object(OllamaAssistant, "_chat_request")
    def test_analyze_query_fallback_on_error(self, mock_chat):
        mock_chat.side_effect = OllamaConnectionError("Connection timed out")

        result = self.assistant.analyze_query("crear un nuevo directorio")

        self.assertIsInstance(result, LLMQueryAnalysis)
        self.assertEqual(result.predicted_category, "filesystem")  # heuristic fallback
        self.assertEqual(len(result.steps), 1)
        self.assertIn("fallback", result.raw_response)

    @patch.object(OllamaAssistant, "_chat_request")
    def test_analyze_query_disallows_unknown_category(self, mock_chat):
        mock_chat.return_value = {
            "predicted_category": "database",  # not in allowed (filesystem, networking)
            "category_confidence": "high",
            "language": "en",
            "steps": [{"step_index": 1, "task": "query db", "search_query": "sql query"}],
        }

        result = self.assistant.analyze_query("query db", allowed_categories=("filesystem", "networking"))
        self.assertIsNone(result.predicted_category)

    @patch.object(OllamaAssistant, "_chat_request")
    def test_select_and_synthesize_successful(self, mock_chat):
        mock_chat.return_value = {
            "selected_index": 2,
            "confidence": "high",
            "assembled_command": "mkdir -p /tmp/docs/sub",
            "explanation": "Crea el directorio final y todos los directorios padre faltantes.",
            "warnings": [],
        }

        candidates = [
            CandidateOption(index=1, alias="filesystem:10", command="rmdir", intent="remove empty directories"),
            CandidateOption(index=2, alias="filesystem:1871", command="mkdir -p", intent="create directories and parents"),
            CandidateOption(index=3, alias="filesystem:55", command="touch", intent="create empty file"),
        ]

        result = self.assistant.select_and_synthesize(
            query="crear /tmp/docs/sub con padres",
            step_task="crear directorio con padres",
            candidates=candidates,
            arguments={"path": "/tmp/docs/sub"},
            language="es",
        )

        self.assertIsInstance(result, LLMSelectionResult)
        self.assertEqual(result.selected_index, 2)
        self.assertEqual(result.selected_alias, "filesystem:1871")
        self.assertEqual(result.assembled_command, "mkdir -p /tmp/docs/sub")
        self.assertEqual(result.confidence, "high")
        self.assertIn("directorios padre", result.explanation)

    def test_select_and_synthesize_empty_candidates(self):
        result = self.assistant.select_and_synthesize(
            query="test",
            step_task="test",
            candidates=[],
            language="es",
        )
        self.assertIsNone(result.selected_index)
        self.assertEqual(result.confidence, "none")

    @patch.object(OllamaAssistant, "_chat_request")
    def test_select_and_synthesize_fallback_on_error(self, mock_chat):
        mock_chat.side_effect = OllamaResponseError("Invalid JSON")
        candidates = [
            CandidateOption(index=1, alias="filesystem:1871", command="mkdir -p", intent="create directories"),
        ]

        result = self.assistant.select_and_synthesize(
            query="crear dir",
            step_task="crear dir",
            candidates=candidates,
        )

        self.assertEqual(result.selected_index, 1)
        self.assertEqual(result.selected_alias, "filesystem:1871")
        self.assertEqual(result.confidence, "low")
        self.assertTrue(any("Fallback" in w for w in result.issues_or_warnings))


if __name__ == "__main__":
    unittest.main()

