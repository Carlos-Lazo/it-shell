import unittest
from pathlib import Path
from unittest.mock import patch

from semantic_search_engine.cli import main
from semantic_search_engine.shared.contracts import QueryRequest
from semantic_search_engine.shared.paths import default_config_path


class FakeEngine:
    def __init__(self):
        self.request = None

    def search(self, request):
        self.request = request
        return f"result: {request.query}"


class TestCli(unittest.TestCase):
    @patch("semantic_search_engine.cli.create_engine", return_value=FakeEngine())
    def test_main_uses_arguments_and_explicit_config(self, create_engine):
        config = Path("/tmp/it-shell-test.yaml")
        engine = create_engine.return_value

        with patch("builtins.print") as print_result:
            main(
                [
                    "find large files",
                    "--config",
                    str(config),
                    "--format",
                    "json",
                    "--top-k",
                    "7",
                ]
            )

        create_engine.assert_called_once_with(config)
        self.assertEqual(
            engine.request,
            QueryRequest("find large files", retrieval_top_k=7, output_format="json"),
        )
        print_result.assert_called_once_with("result: find large files")

    def test_default_config_selects_repository_config(self):
        self.assertEqual(
            default_config_path(),
            Path(__file__).resolve().parents[1]
            / "semantic_search_engine/config/default.yaml",
        )


if __name__ == "__main__":
    unittest.main()
