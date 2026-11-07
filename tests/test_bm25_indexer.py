from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_builder.indexer.bm25_indexer import (
    MAX_INTENT_CHARS,
    _contents_for_record,
    _load_search_phrases,
    _load_yaml_folder,
    _prepare_input,
)


class TestBM25IndexerRepresentation(unittest.TestCase):
    def test_legacy_representation_indexes_only_the_command(self):
        record = {
            "command": "grep",
            "intent": "search text files",
            "example": "grep ERROR /var/log/syslog",
        }

        self.assertEqual(_contents_for_record(record, "command"), "grep")

    def test_enriched_representation_includes_command_intent_and_example(self):
        record = {
            "command": "grep",
            "intent": "search text files",
            "example": "grep ERROR /var/log/syslog",
        }

        self.assertEqual(
            _contents_for_record(record, "enriched"),
            "grep search text files grep ERROR /var/log/syslog",
        )

    def test_enriched_representation_caps_long_intent_text(self):
        record = {
            "command": "tool",
            "intent": " ".join(f"term{index}" for index in range(100)),
            "example": "tool --help",
        }

        contents = _contents_for_record(record, "enriched")
        indexed_intent = contents.removeprefix("tool ").removesuffix(" tool --help")

        self.assertLessEqual(len(indexed_intent), MAX_INTENT_CHARS)
        self.assertEqual(contents.split()[0], "tool")
        self.assertTrue(contents.endswith("tool --help"))

    def test_curated_representation_adds_only_reviewed_phrases(self):
        record = {
            "id": "filesystem:877",
            "command": "find",
            "intent": "long manual text that should not enter the curated representation",
            "example": "find",
        }

        contents = _contents_for_record(
            record,
            "curated",
            ["search files by name", "locate files under a directory"],
        )

        self.assertEqual(
            contents,
            "find search files by name locate files under a directory",
        )
        self.assertNotIn("manual", contents)

    def test_enriched_paraphrases_adds_phrases_to_enriched_content(self):
        record = {
            "id": "filesystem:877",
            "command": "find",
            "intent": "search files",
            "example": "find /tmp",
        }

        self.assertEqual(
            _contents_for_record(
                record, "enriched_paraphrases", ["locate files", "find by name"]
            ),
            "find search files find /tmp locate files find by name",
        )

    def test_curated_input_preserves_phrases_in_raw_json(self):
        records = [{"id": "filesystem:877", "command": "find", "intent": "search files"}]
        phrases = {"filesystem:877": ["locate files by name"]}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _prepare_input(records, Path(temp_dir), "curated", phrases)
            document = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(document["contents"], "find locate files by name")
        self.assertEqual(document["search_phrases"], ["locate files by name"])

    def test_enriched_paraphrases_input_keeps_empty_aliases_and_raw_phrases(self):
        records = [
            {
                "id": "filesystem:1",
                "command": "find",
                "intent": "search files",
                "example": "find .",
            },
            {
                "id": "filesystem:2",
                "command": "ls",
                "intent": "list files",
                "example": "ls -l",
            },
        ]
        phrases = {"filesystem:1": ["locate files"], "filesystem:2": []}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _prepare_input(
                records, Path(temp_dir), "enriched_paraphrases", phrases
            )
            documents = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            documents[0]["contents"],
            "find search files find . locate files",
        )
        self.assertEqual(documents[0]["search_phrases"], ["locate files"])
        self.assertEqual(
            documents[1]["contents"], "ls list files ls -l"
        )
        self.assertEqual(documents[1]["search_phrases"], [])

    def test_curated_representation_requires_phrase_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "requiere un archivo de frases"):
                _prepare_input(
                    [{"id": "filesystem:877", "command": "find"}],
                    Path(temp_dir),
                    "curated",
                )

    def test_curated_phrase_aliases_must_exist_in_corpus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "Aliases sin documento"):
                _prepare_input(
                    [{"id": "filesystem:877", "command": "find"}],
                    Path(temp_dir),
                    "curated",
                    {"docker:1": ["run a container"]},
                )

    def test_enriched_paraphrases_require_complete_alias_coverage_by_category(self):
        records = [
            {"id": "filesystem:1", "command": "find"},
            {"id": "filesystem:2", "command": "ls"},
            {"id": "networking:1", "command": "curl"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                ValueError, "Aliases sin entrada de paráfrasis: filesystem:2"
            ):
                _prepare_input(
                    records,
                    Path(temp_dir),
                    "enriched_paraphrases",
                    {"filesystem:1": ["locate files"]},
                )

    def test_phrase_loader_rejects_duplicate_phrases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            phrase_file = Path(temp_dir) / "phrases.json"
            phrase_file.write_text(
                json.dumps({"filesystem:877": ["find files", "find files"]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "no deben repetirse"):
                _load_search_phrases(phrase_file)

    def test_enriched_phrase_loader_accepts_abstentions_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            filesystem = root / "filesystem.json"
            networking = root / "networking.json"
            filesystem.write_text(
                json.dumps({
                    "filesystem:1": [" locate files ", "locate files"],
                    "filesystem:2": [],
                }),
                encoding="utf-8",
            )
            networking.write_text(
                json.dumps({"networking:1": ["find a host"]}),
                encoding="utf-8",
            )

            phrases = _load_search_phrases(
                [filesystem, networking], allow_empty=True, deduplicate=True
            )

        self.assertEqual(
            phrases,
            {
                "filesystem:1": ["locate files"],
                "filesystem:2": [],
                "networking:1": ["find a host"],
            },
        )

    def test_enriched_input_preserves_searchable_fields_in_raw_json(self):
        records = [{
            "id": "textprocessing:1",
            "command": "grep",
            "intent": "search text files",
            "example": "grep ERROR /var/log/syslog",
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _prepare_input(records, Path(temp_dir), "enriched")
            document = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(document["contents"], "grep search text files grep ERROR /var/log/syslog")
        self.assertEqual(document["id"], "textprocessing:1")
        self.assertEqual(document["intent"], "search text files")
        self.assertEqual(document["example"], "grep ERROR /var/log/syslog")

    def test_yaml_loader_keeps_example_for_indexing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir)
            (corpus / "textprocessing.yaml").write_text(
                'category: "textprocessing"\n'
                "documents:\n"
                "  - id: 1\n"
                '    command: "grep"\n'
                '    intent: "search text files"\n'
                '    example: "grep ERROR /var/log/syslog"\n',
                encoding="utf-8",
            )

            records = _load_yaml_folder(corpus)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["intent"], "search text files")
        self.assertEqual(records[0]["example"], "grep ERROR /var/log/syslog")

    def test_unknown_representation_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Representación BM25 inválida"):
            _contents_for_record({"command": "grep"}, "unknown")


if __name__ == "__main__":
    unittest.main()
