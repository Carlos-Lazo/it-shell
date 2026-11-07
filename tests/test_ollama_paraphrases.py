import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.ollama.generate_paraphrases_ import (
    generate_phrases,
    main,
    load_records,
    parse_phrases,
    save_results,
)


class OllamaParaphraseTests(unittest.TestCase):
    def test_load_records_builds_aliases_from_yaml_corpus(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            corpus = Path(temporary_directory) / "filesystem.yaml"
            corpus.write_text(
                'category: "filesystem"\n'
                "documents:\n"
                '  - id: 7\n'
                '    command: "find"\n'
                '    intent: "Search files"\n'
                '    example: "find ."\n',
                encoding="utf-8",
            )

            self.assertEqual(
                load_records(corpus),
                [{"alias": "filesystem:7", "intent": "Search files"}],
            )

    def test_parse_phrases_validates_model_response(self):
        self.assertEqual(
            parse_phrases('{"phrases": ["  find files  ", "locate files"]}', "filesystem:7"),
            ["find files", "locate files"],
        )
        with self.assertRaises(ValueError):
            parse_phrases('{"phrases": ["one", "two", "three", "four", "five"]}', "filesystem:7")

    def test_generation_sends_only_intent_to_model(self):
        response_body = json.dumps({
            "message": {"content": '{"phrases": ["choose a file-list character encoding"]}'}
        }).encode("utf-8")
        record = {
            "alias": "filesystem:7",
            "command": "7za",
            "intent": "set charset for list files",
            "example": "7za -scs{UTF-8|UTF-16LE}",
        }

        with patch(
            "evaluation.ollama.generate_paraphrases_.urlopen",
            return_value=io.BytesIO(response_body),
        ) as mocked_urlopen:
            phrases = generate_phrases("qwen2.5:3b", "http://localhost:11434", record, 10)

        payload = json.loads(mocked_urlopen.call_args.args[0].data)
        self.assertEqual(phrases, ["choose a file-list character encoding"])
        self.assertEqual(
            json.loads(payload["messages"][1]["content"]),
            {"intent": "set charset for list files"},
        )

    def test_generation_retries_invalid_json_response(self):
        response_bodies = [
            json.dumps({"message": {"content": '{"phrases": ["one", "two", "three", "four", "five"]}'}}),
            json.dumps({"message": {"content": '{"phrases": ["choose a file-list encoding"]}'}}),
        ]
        record = {"alias": "filesystem:7", "intent": "set charset for list files"}

        with patch(
            "evaluation.ollama.generate_paraphrases_.urlopen",
            side_effect=[io.BytesIO(body.encode("utf-8")) for body in response_bodies],
        ) as mocked_urlopen:
            phrases = generate_phrases("qwen2.5:3b", "http://localhost:11434", record, 10)

        self.assertEqual(phrases, ["choose a file-list encoding"])
        self.assertEqual(mocked_urlopen.call_count, 2)
        retry_payload = json.loads(mocked_urlopen.call_args_list[1].args[0].data)
        self.assertEqual(retry_payload["messages"][-1]["role"], "user")

    def test_generation_retries_connection_timeout(self):
        response_body = json.dumps({
            "message": {"content": '{"phrases": ["choose a file-list encoding"]}'}
        }).encode("utf-8")
        record = {"alias": "filesystem:7", "intent": "set charset for list files"}

        with (
            patch(
                "evaluation.ollama.generate_paraphrases_.urlopen",
                side_effect=[TimeoutError("timed out"), io.BytesIO(response_body)],
            ) as mocked_urlopen,
            patch("evaluation.ollama.generate_paraphrases_.time.sleep"),
        ):
            phrases = generate_phrases("qwen2.5:3b", "http://localhost:11434", record, 10)

        self.assertEqual(phrases, ["choose a file-list encoding"])
        self.assertEqual(mocked_urlopen.call_count, 2)

    def test_save_results_writes_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "nested" / "phrases.json"
            save_results(output, {"filesystem:7": ["buscar archivos"]})

            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"filesystem:7": ["buscar archivos"]},
            )

    def test_main_writes_one_homonymous_json_per_corpus_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            corpus = root / "corpus"
            corpus.mkdir()
            output_dir = root / "corpus_paraphrases"
            (corpus / "filesystem.yaml").write_text(
                'category: "filesystem"\n'
                "documents:\n"
                '  - id: 7\n'
                '    intent: "set charset for list files"\n',
                encoding="utf-8",
            )
            (corpus / "networking.yaml").write_text(
                'category: "networking"\n'
                "documents:\n"
                '  - id: 2\n'
                '    intent: "send an HTTP request"\n',
                encoding="utf-8",
            )

            with (
                patch(
                    "sys.argv",
                    [
                        "generate_paraphrases_.py",
                        "--corpus", str(corpus),
                        "--output-dir", str(output_dir),
                    ],
                ),
                patch(
                    "evaluation.ollama.generate_paraphrases_.generate_phrases",
                    side_effect=[
                        ValueError("invalid output"),
                        ["make an HTTP request"],
                    ],
                ),
            ):
                main()

            self.assertEqual(
                json.loads((output_dir / "filesystem.json").read_text(encoding="utf-8")),
                {},
            )
            self.assertEqual(
                json.loads((output_dir / "filesystem.errors.json").read_text(encoding="utf-8")),
                {"filesystem:7": "invalid output"},
            )
            self.assertEqual(
                json.loads((output_dir / "networking.json").read_text(encoding="utf-8")),
                {"networking:2": ["make an HTTP request"]},
            )

    def test_resume_restarts_at_first_unfinished_alias(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            corpus = root / "filesystem.yaml"
            corpus.write_text(
                'category: "filesystem"\n'
                "documents:\n"
                '  - id: 1\n'
                '    intent: "first intent"\n'
                '  - id: 2\n'
                '    intent: "second intent"\n',
                encoding="utf-8",
            )
            output_dir = root / "corpus_paraphrases"
            output_file = output_dir / "filesystem.json"

            with (
                patch(
                    "sys.argv",
                    [
                        "generate_paraphrases_.py",
                        "--corpus", str(corpus),
                        "--output-dir", str(output_dir),
                    ],
                ),
                patch(
                    "evaluation.ollama.generate_paraphrases_.generate_phrases",
                    side_effect=[["first paraphrase"], RuntimeError("timed out")],
                ),
                self.assertRaises(RuntimeError),
            ):
                main()

            self.assertEqual(
                json.loads(output_file.read_text(encoding="utf-8")),
                {"filesystem:1": ["first paraphrase"]},
            )

            with (
                patch(
                    "sys.argv",
                    [
                        "generate_paraphrases_.py",
                        "--corpus", str(corpus),
                        "--output-dir", str(output_dir),
                        "--resume",
                    ],
                ),
                patch(
                    "evaluation.ollama.generate_paraphrases_.generate_phrases",
                    return_value=["second paraphrase"],
                ) as mocked_generate,
            ):
                main()

            self.assertEqual(mocked_generate.call_count, 1)
            self.assertEqual(
                mocked_generate.call_args.args[2]["alias"],
                "filesystem:2",
            )
            self.assertEqual(
                json.loads(output_file.read_text(encoding="utf-8")),
                {
                    "filesystem:1": ["first paraphrase"],
                    "filesystem:2": ["second paraphrase"],
                },
            )


if __name__ == "__main__":
    unittest.main()
